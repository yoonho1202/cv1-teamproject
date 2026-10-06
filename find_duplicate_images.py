#!/usr/bin/env python3
"""Find identical animal photos and conservative resize candidates; never delete images."""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
RESAMPLERS = tuple(Image.Resampling)
MAX_EXACT_RESIZE_PIXELS = 8_000_000
MIN_COMPARE_SIDE = 64
MAX_COMPARE_SIDE = 256

# pHash is only a candidate filter, never sufficient evidence of a duplicate.
MAX_HASH_DISTANCE = 8
MIN_SSIM = 0.995
MAX_RGB_RMSE = 0.025  # RGB values are normalized to [0, 1].

_indices = np.arange(32, dtype=np.float64)
_dct = np.cos(np.pi * (2 * _indices[None, :] + 1) * _indices[:, None] / 64)


@dataclass(frozen=True)
class Photo:
    path: Path
    width: int
    height: int
    byte_size: int
    file_hash: str
    pixel_hash: str
    phash: int


@dataclass(frozen=True)
class Match:
    kind: str
    hash_distance: int
    ssim: float = 1.0
    rgb_rmse: float = 0.0


@dataclass(frozen=True)
class Duplicate:
    group: int
    keep: Photo
    duplicate: Photo
    match: Match


@dataclass
class ScanResult:
    photos: list[Photo]
    duplicates: list[Duplicate]
    errors: list[tuple[Path, str]]


def load_rgb(path: Path) -> Image.Image:
    """Compare the displayed image; do not search rotations, flips or crops."""
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("animated/multipage images are not supported")
        displayed = ImageOps.exif_transpose(source)
        if "A" in displayed.getbands() or "transparency" in displayed.info:
            rgba = displayed.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            return Image.alpha_composite(background, rgba).convert("RGB")
        return displayed.convert("RGB")


def perceptual_hash(image: Image.Image) -> int:
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
                      dtype=np.float64)
    coefficients = (_dct @ gray @ _dct.T)[:8, :8].ravel()[1:]
    bits = coefficients > np.median(coefficients)
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def inspect_photo(path: Path) -> Photo:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    image = load_rgb(path)
    pixels = hashlib.sha256()
    pixels.update(image.width.to_bytes(8, "big"))
    pixels.update(image.height.to_bytes(8, "big"))
    pixels.update(image.tobytes())
    return Photo(path, image.width, image.height, path.stat().st_size,
                 digest.hexdigest(), pixels.hexdigest(), perceptual_hash(image))


def local_ssim(left: np.ndarray, right: np.ndarray) -> float:
    """Mean SSIM of 8x8 luminance blocks, including padded border pixels."""
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    left_gray, right_gray = left @ weights, right @ weights
    height, width = left_gray.shape
    padding = ((0, (-height) % 8), (0, (-width) % 8))

    def blocks(gray: np.ndarray) -> np.ndarray:
        padded = np.pad(gray, padding, mode="edge")
        h, w = padded.shape
        return padded.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)

    a, b = blocks(left_gray), blocks(right_gray)
    ma, mb = a.mean(axis=1), b.mean(axis=1)
    va, vb = a.var(axis=1), b.var(axis=1)
    covariance = ((a - ma[:, None]) * (b - mb[:, None])).mean(axis=1)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    scores = ((2 * ma * mb + c1) * (2 * covariance + c2)
              / ((ma * ma + mb * mb + c1) * (va + vb + c2)))
    return float(scores.mean())


def compare_photos(left: Photo, right: Photo) -> Match | None:
    distance = (left.phash ^ right.phash).bit_count()
    if left.file_hash == right.file_hash:
        return Match("exact_file", distance)
    if left.pixel_hash == right.pixel_hash:
        return Match("exact_pixels", distance)
    if distance > MAX_HASH_DISTANCE:
        return None

    ratio_error = abs(left.width * right.height - right.width * left.height)
    ratio_error /= max(left.width * right.height, right.width * left.height)
    # A proportional thumbnail can round its short side by half a pixel.
    rounding_tolerance = 0.5 / min(left.width, left.height, right.width, right.height)
    if ratio_error > max(0.005, rounding_tolerance):
        return None
    if min(left.width, left.height, right.width, right.height) < MIN_COMPARE_SIDE:
        return None  # Too little detail for a reliable resized-photo comparison.

    a, b = load_rgb(left.path), load_rgb(right.path)
    if a.size != b.size:
        # Exact reproduction of a supported resize also catches NEAREST and BOX.
        # Both directions are needed: the original may be the smaller image.
        for source, target in ((a, b), (b, a)):
            if target.width * target.height > MAX_EXACT_RESIZE_PIXELS:
                continue
            target_pixels = target.tobytes()
            for resampler in RESAMPLERS:
                resized = source.resize(target.size, resampler)
                if resized.tobytes() == target_pixels:
                    return Match("exact_resize", distance)

    # Unknown resampling/JPEG compression: report only a conservative candidate.
    # Compare colors and local structure at two scales, without image alignment.
    smaller = min((a, b), key=lambda im: im.width * im.height)
    longest = min(MAX_COMPARE_SIDE, max(smaller.size))
    worst_ssim, worst_rmse = 1.0, 0.0
    for edge in (longest, max(32, longest // 2)):
        factor = edge / max(smaller.size)
        size = (max(1, round(smaller.width * factor)),
                max(1, round(smaller.height * factor)))
        aa = np.asarray(a.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255
        bb = np.asarray(b.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255
        rmse = float(np.sqrt(np.mean((aa - bb) ** 2)))
        ssim = local_ssim(aa, bb)
        # A localized difference can hide behind a high mean similarity.
        changed_fraction = float(np.mean(np.max(np.abs(aa - bb), axis=2) > 0.1))
        if ssim < MIN_SSIM or rmse > MAX_RGB_RMSE or changed_fraction > 0.005:
            return None
        worst_ssim, worst_rmse = min(worst_ssim, ssim), max(worst_rmse, rmse)
    return Match("resize_candidate", distance, worst_ssim, worst_rmse)


def scan_directory(directory: Path) -> ScanResult:
    directory = directory.resolve()
    paths = sorted(path for path in directory.rglob("*")
                   if path.is_file() and path.suffix.lower() in EXTENSIONS)
    photos, errors = [], []
    for path in paths:
        try:
            photos.append(inspect_photo(path))
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            errors.append((path, str(error)))
    # Keep the highest resolution; use byte size and path to break ties.
    photos.sort(key=lambda p: (-p.width * p.height, -p.byte_size, str(p.path)))
    representatives: list[Photo] = []
    duplicates: list[Duplicate] = []
    for photo in photos:
        for index, keep in enumerate(representatives):
            try:
                match = compare_photos(keep, photo)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                errors.append((photo.path, str(error)))
                break
            if match:
                duplicates.append(Duplicate(index + 1, keep, photo, match))
                break
        else:
            representatives.append(photo)
    # Compare every match directly to its keeper; never merge through a chain.
    return ScanResult(photos, duplicates, errors)


def write_report(result: ScanResult, report: Path) -> None:
    # Exclusive creation protects a pre-existing report from accidental overwrite.
    with report.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "keep", "duplicate", "match_kind",
                         "phash_distance", "ssim", "rgb_rmse"])
        for row in result.duplicates:
            writer.writerow([row.group, str(row.keep.path), str(row.duplicate.path),
                             row.match.kind, row.match.hash_distance,
                             f"{row.match.ssim:.6f}", f"{row.match.rgb_rmse:.6f}"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="사진 폴더 (하위 폴더 포함)")
    parser.add_argument("-o", "--report", type=Path, default=Path("duplicates.csv"),
                        help="새 CSV 파일 경로 (기본: duplicates.csv)")
    args = parser.parse_args(argv)
    if not args.directory.is_dir():
        parser.error(f"사진 폴더가 없습니다: {args.directory}")
    if args.report.exists():
        parser.error(f"CSV 파일이 이미 있습니다. 다른 -o 경로를 지정하세요: {args.report}")
    result = scan_directory(args.directory)
    if not result.photos:
        print("검사 가능한 사진이 없습니다.", file=sys.stderr)
        for path, error in result.errors:
            print(f"읽기 실패: {path}: {error}", file=sys.stderr)
        return 2
    try:
        write_report(result, args.report)
    except OSError as error:
        print(f"CSV 저장 실패: {error}", file=sys.stderr)
        return 2
    exact = sum(row.match.kind != "resize_candidate" for row in result.duplicates)
    candidates = len(result.duplicates) - exact
    print(f"검사: {len(result.photos)}장 / 확정 중복: {exact}장 / 리사이즈 추정 후보: {candidates}장")
    print(f"CSV: {args.report.resolve()}")
    print("원본 사진은 변경하지 않았습니다. 추정 후보는 삭제 전에 직접 확인하세요.")
    for path, error in result.errors:
        print(f"읽기 실패: {path}: {error}", file=sys.stderr)
    return 2 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
