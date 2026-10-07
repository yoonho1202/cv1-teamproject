#!/usr/bin/env python3
"""Compare train photos against test/val; report train removal candidates only."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from find_duplicate_images import EXTENSIONS, Photo, inspect_photo
from find_similar_images import SimilarPair, extract_embeddings, write_results


@dataclass(frozen=True)
class SplitMatch:
    train: int
    reference: int
    reference_split: str
    similarity: float
    global_similarity: float
    center_similarity: float
    match_kind: str = "near_duplicate_candidate"


def validate_folders(train: Path, test: Path | None, val: Path | None,
                     output_dir: Path) -> dict[str, Path]:
    if test is None and val is None:
        raise ValueError("--test 또는 --val을 하나 이상 지정하세요.")
    roots = {name: path.resolve() for name, path in (("train", train), ("test", test), ("val", val))
             if path is not None}
    for name, path in roots.items():
        if not path.is_dir():
            raise ValueError(f"{name} 사진 폴더가 없습니다: {path}")
    names = list(roots)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if roots[name].is_relative_to(roots[other]) or roots[other].is_relative_to(roots[name]):
                raise ValueError(f"{name}과 {other} 폴더는 같거나 서로 포함될 수 없습니다.")
    output = output_dir.resolve()
    for name, path in roots.items():
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError(f"결과 폴더는 {name} 입력 폴더와 분리해서 지정하세요.")
    if output.exists():
        raise ValueError(f"결과 폴더가 이미 있습니다. 새 -o 이름을 지정하세요: {output}")
    return roots


def _discover(root: Path) -> list[Path]:
    paths = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"입력 폴더 밖으로 연결된 사진입니다: {path}")
            paths.append(path.absolute())
    return paths


def _normalized(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise ValueError("임베딩은 유한한 2차원 배열이어야 합니다.")
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (lengths <= 0).any():
        raise ValueError("0 임베딩은 비교할 수 없습니다.")
    return vectors / lengths


def select_overlaps(train_global, train_center, reference_global, reference_center,
                    reference_splits: list[str], threshold: float = 0.90,
                    top_k: int = 0) -> list[SplitMatch]:
    if not -1 <= threshold <= 1 or top_k < 0:
        raise ValueError("threshold는 -1~1, top-k는 0 이상이어야 합니다.")
    tg, tc, rg, rc = map(_normalized, (train_global, train_center, reference_global, reference_center))
    if len(tg) != len(tc) or len(rg) != len(rc) or len(rg) != len(reference_splits):
        raise ValueError("사진 수와 임베딩·비교 세트 이름 수가 일치하지 않습니다.")
    if tg.shape[1] != rg.shape[1] or tc.shape[1] != rc.shape[1]:
        raise ValueError("비교 임베딩 차원이 일치하지 않습니다.")
    if any(name not in ("test", "val") for name in reference_splits):
        raise ValueError("비교 대상은 test 또는 val이어야 합니다.")
    whole = np.clip(tg @ rg.T, -1, 1)
    center = np.clip(tc @ rc.T, -1, 1)
    score = 0.4 * whole + 0.6 * center
    matches = []
    for i in range(len(tg)):
        indices = np.argsort(-score[i], kind="stable")
        selected = [int(j) for j in indices if score[i, j] >= threshold]
        if top_k:
            selected = selected[:top_k]
        matches.extend(SplitMatch(i, j, reference_splits[j], float(score[i, j]),
                                  float(whole[i, j]), float(center[i, j])) for j in selected)
    return sorted(matches, key=lambda match: (-match.similarity, match.train, match.reference))


def find_exact_overlaps(train_photos: list[Photo], reference_photos: list[Photo],
                        reference_splits: list[str]) -> list[SplitMatch]:
    if len(reference_photos) != len(reference_splits):
        raise ValueError("비교 사진과 세트 이름 수가 일치하지 않습니다.")
    by_file: dict[str, list[int]] = {}
    by_pixels: dict[str, list[int]] = {}
    for j, photo in enumerate(reference_photos):
        by_file.setdefault(photo.file_hash, []).append(j)
        by_pixels.setdefault(photo.pixel_hash, []).append(j)
    matches = []
    for i, photo in enumerate(train_photos):
        identical_files = set(by_file.get(photo.file_hash, []))
        for j in sorted(identical_files | set(by_pixels.get(photo.pixel_hash, []))):
            kind = "exact_file" if j in identical_files else "exact_pixels"
            matches.append(SplitMatch(i, j, reference_splits[j], 1.0, 1.0, 1.0, kind))
    return matches


def write_overlap_results(train_paths: list[Path], reference_paths: list[Path],
                          matches: list[SplitMatch], output_dir: Path, *,
                          incomplete_count: int = 0) -> None:
    paths = train_paths + reference_paths
    labels = {i: "train — 제거 검토 대상" for i in range(len(train_paths))}
    for match in matches:
        labels[len(train_paths) + match.reference] = f"{match.reference_split} — 유지"
    pairs = [SimilarPair(match.train, len(train_paths) + match.reference,
                         match.similarity, match.global_similarity, match.center_similarity)
             for match in matches]
    count = len({match.train for match in matches})
    guidance = f"train 제거 검토 대상 {count}장. 왼쪽 train과 오른쪽 test·val을 비교하세요."
    if incomplete_count:
        guidance += f" 읽기 실패 {incomplete_count}건: 일부 사진이 제외된 미완료 검사입니다."
    descriptions = {"exact_file": "정확한 일치: 파일 내용이 같습니다.",
                    "exact_pixels": "정확한 일치: 표시되는 픽셀이 같습니다.",
                    "near_duplicate_candidate": "유사 후보: 같은 개체·포즈·배경인지 확인하세요."}
    evidence_labels = {(match.train, len(train_paths) + match.reference):
                       descriptions.get(match.match_kind, match.match_kind) for match in matches}
    write_results(paths, pairs, output_dir, image_labels=labels, csv_name=None,
                  pair_labels=evidence_labels,
                  title="train과 test·val 사이의 겹침 검토",
                  guidance=guidance)
    with (output_dir / "split_overlaps.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["train_image", "reference_image", "reference_split", "similarity",
                         "global_similarity", "center_similarity", "match_kind"])
        for match in matches:
            writer.writerow([str(train_paths[match.train]), str(reference_paths[match.reference]),
                             match.reference_split, f"{match.similarity:.6f}",
                             f"{match.global_similarity:.6f}", f"{match.center_similarity:.6f}",
                             match.match_kind])
    # A unique train list makes pair counts distinct from potential removal counts.
    with (output_dir / "train_candidates.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["train_image", "review_required"])
        for index in sorted({match.train for match in matches}):
            writer.writerow([str(train_paths[index]), "yes"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=0, help="train당 유사 후보 제한 (0: 제한 없음)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("split_overlap_results"))
    args = parser.parse_args(argv)
    if not -1 <= args.threshold <= 1 or args.top_k < 0 or args.batch_size < 1:
        parser.error("threshold는 -1~1, top-k는 0 이상, batch-size는 1 이상이어야 합니다.")
    try:
        roots = validate_folders(args.train, args.test, args.val, args.output_dir)
        discovered = {name: _discover(path) for name, path in roots.items()}
        for name, paths in discovered.items():
            if not paths:
                raise ValueError(f"{name} 폴더에 검사할 사진이 없습니다: {roots[name]}")
        all_paths, split_of = [], {}
        for name, paths in discovered.items():
            all_paths.extend(paths)
            split_of.update({path: name for path in paths})
        valid, whole, center, errors = extract_embeddings(all_paths, args.batch_size)
        for path, error in errors:
            print(f"읽기 실패: {path}: {error}", file=sys.stderr)
        train_indices = [i for i, path in enumerate(valid) if split_of[path] == "train"]
        reference_indices = [i for i, path in enumerate(valid) if split_of[path] != "train"]
        if not train_indices or not reference_indices:
            raise ValueError("읽을 수 있는 train 사진과 test·val 사진이 모두 필요합니다.")
        train_paths = [valid[i] for i in train_indices]
        reference_paths = [valid[i] for i in reference_indices]
        reference_splits = [split_of[path] for path in reference_paths]
        matches = select_overlaps(whole[train_indices], center[train_indices],
                                  whole[reference_indices], center[reference_indices],
                                  reference_splits, args.threshold, args.top_k)
        # Inspect decoded/hash data only for images already read successfully.
        train_photos = [inspect_photo(path) for path in train_paths]
        reference_photos = [inspect_photo(path) for path in reference_paths]
        exact = find_exact_overlaps(train_photos, reference_photos, reference_splits)
        merged = {(match.train, match.reference): match for match in matches}
        for match in exact:
            previous = merged.get((match.train, match.reference))
            if previous is not None:
                match = replace(previous, match_kind=match.match_kind)
            merged[match.train, match.reference] = match
        matches = sorted(merged.values(), key=lambda match: (-match.similarity, match.train, match.reference))
        write_overlap_results(train_paths, reference_paths, matches, args.output_dir,
                              incomplete_count=len(errors))
    except (RuntimeError, OSError, ValueError, Image.DecompressionBombError) as error:
        print(f"검사 실패: {error}", file=sys.stderr)
        return 2
    train_count = len({match.train for match in matches})
    print(f"train {len(train_paths)}장 × test·val {len(reference_paths)}장 비교 완료")
    print(f"겹침 검토 후보: {len(matches)}쌍 / 제거 검토 대상 train: {train_count}장")
    print(f"사진 비교: {(args.output_dir / 'index.html').resolve()}")
    print("train·test·val 사진은 변경하지 않았습니다. 삭제 전 후보를 직접 확인하세요.")
    if args.top_k:
        print(f"유사 후보는 train당 최대 {args.top_k}쌍으로 제한했습니다. 정확한 파일·픽셀 일치는 모두 포함합니다.")
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
