#!/usr/bin/env python3
"""Rank visually similar photos for human review, without modifying images."""

from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import dataclass
import html
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps

from find_duplicate_images import EXTENSIONS, load_rgb


@dataclass(frozen=True)
class SimilarPair:
    left: int
    right: int
    similarity: float
    global_similarity: float
    center_similarity: float


def select_pairs(global_vectors: np.ndarray, center_vectors: np.ndarray,
                 threshold: float = 0.90, top_k: int = 5) -> list[SimilarPair]:
    """Union of each image's top-K neighbors; do not form transitive groups."""
    if not -1 <= threshold <= 1 or top_k < 1:
        raise ValueError("threshold must be in [-1, 1] and top_k must be positive")

    def normalized(vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError("embeddings must be a finite two-dimensional matrix")
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (lengths <= 0).any():
            raise ValueError("embeddings cannot contain a zero vector")
        return vectors / lengths

    global_vectors, center_vectors = normalized(global_vectors), normalized(center_vectors)
    if len(global_vectors) != len(center_vectors):
        raise ValueError("embedding matrices must have the same number of images")
    whole = np.clip(global_vectors @ global_vectors.T, -1, 1)
    center = np.clip(center_vectors @ center_vectors.T, -1, 1)
    score = 0.4 * whole + 0.6 * center
    np.fill_diagonal(score, -np.inf)
    selected: set[tuple[int, int]] = set()
    for i in range(len(score)):
        neighbors = np.argsort(-score[i], kind="stable")[:min(top_k, len(score) - 1)]
        for j in neighbors:
            if score[i, j] >= threshold:
                selected.add((min(i, int(j)), max(i, int(j))))
    return sorted((SimilarPair(i, j, float(score[i, j]), float(whole[i, j]), float(center[i, j]))
                   for i, j in selected), key=lambda pair: (-pair.similarity, pair.left, pair.right))


def extract_embeddings(paths: list[Path], batch_size: int = 8):
    """Use official ImageNet V2 ResNet50 weights on whole and centered views."""
    try:
        import torch
        from torchvision.models import ResNet50_Weights, resnet50
    except ImportError as error:
        raise RuntimeError("추가 패키지가 필요합니다. README의 CPU용 PyTorch 설치 명령을 실행하세요.") from error
    os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parent / ".model_cache"))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    print("이미지 모델 준비 중입니다. 처음 실행하면 공식 모델 파일 약 98MB를 내려받습니다.", flush=True)
    # torchvision verifies the official weight file's SHA256 prefix on download.
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    model.fc = torch.nn.Identity()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def views(image):
        # The whole view preserves background; the central view emphasizes content.
        whole = ImageOps.pad(image, (224, 224), method=Image.Resampling.LANCZOS,
                             color=(124, 116, 104))
        side = min(image.size)
        x, y = (image.width - side) // 2, (image.height - side) // 2
        center = image.crop((x, y, x + side, y + side)).resize((224, 224), Image.Resampling.LANCZOS)
        return [(torch.from_numpy(np.asarray(view).copy()).permute(2, 0, 1).float() / 255 - mean) / std
                for view in (whole, center)]

    valid, errors, whole_vectors, center_vectors = [], [], [], []
    for start in range(0, len(paths), batch_size):
        tensors, batch_paths = [], []
        for path in paths[start:start + batch_size]:
            try:
                tensors.extend(views(load_rgb(path)))
                batch_paths.append(path)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                errors.append((path, str(error)))
        if tensors:
            with torch.inference_mode():
                vectors = model(torch.stack(tensors)).numpy()
            whole_vectors.append(vectors[0::2])
            center_vectors.append(vectors[1::2])
            valid.extend(batch_paths)
        print(f"사진 분석: {min(start + batch_size, len(paths))}/{len(paths)}장", flush=True)
    empty = np.empty((0, 2048), dtype=np.float32)
    return (valid, np.concatenate(whole_vectors) if whole_vectors else empty,
            np.concatenate(center_vectors) if center_vectors else empty, errors)


def _thumbnail(path: Path) -> str:
    image = load_rgb(path)
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def write_results(paths: list[Path], pairs: list[SimilarPair], output_dir: Path) -> None:
    # Prepare all thumbnails before creating output, so unreadable images fail cleanly.
    if output_dir.exists():
        raise FileExistsError(f"결과 폴더가 이미 있습니다. 새 --output-dir 이름을 지정하세요: {output_dir}")
    used = sorted({index for pair in pairs for index in (pair.left, pair.right)})
    thumbnails = {str(index): _thumbnail(paths[index]) for index in used}
    cards = []
    for pair in pairs:
        left, right = paths[pair.left], paths[pair.right]
        searchable = html.escape(f"{left} {right}", quote=True)
        cards.append(f'''<article data-score="{pair.similarity:.8f}" data-name="{searchable}">
<p class="score">유사도 {pair.similarity:.3f} · 전체 {pair.global_similarity:.3f} · 중앙 {pair.center_similarity:.3f}</p>
<div class="photos"><figure><img data-image="{pair.left}" loading="lazy" alt="{html.escape(left.name, quote=True)}"><figcaption>{html.escape(left.name)}<small>{html.escape(str(left))}</small></figcaption></figure>
<figure><img data-image="{pair.right}" loading="lazy" alt="{html.escape(right.name, quote=True)}"><figcaption>{html.escape(right.name)}<small>{html.escape(str(right))}</small></figcaption></figure></div>
</article>''')
    images_json = json.dumps(thumbnails, ensure_ascii=True).replace("<", "\\u003c")
    minimum = float(np.floor(min((pair.similarity for pair in pairs), default=0.0) * 1000) / 1000)
    document = f'''<!doctype html>
<html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>유사 사진 검토</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:1050px;margin:24px auto;padding:0 16px;background:#f5f6f8;color:#20242c}}
header{{position:sticky;top:0;background:#f5f6f8;padding:8px 0;z-index:1}}
input{{padding:8px;margin:4px 8px 4px 0}}article{{background:white;border:1px solid #dde1e7;border-radius:10px;padding:12px;margin:16px 0}}
.photos{{display:flex;gap:16px}}figure{{margin:0;flex:1;min-width:0;text-align:center}}
img{{width:100%;height:300px;object-fit:contain;background:#eef0f3}}figcaption{{overflow-wrap:anywhere;margin-top:8px}}
small{{display:block;color:#596474;font-size:12px;margin-top:4px}}
.score{{font-weight:600}}[hidden]{{display:none}}@media(max-width:550px){{img{{height:210px}}}}
</style><h1>유사 사진 검토</h1>
<p>검사 {len(paths)}장 · 후보 {len(pairs)}쌍. 같은 동물·포즈·배경인지 사진을 보고 확인하세요.</p>
<p>모델 유사도는 확률이나 개체 동일성의 확정 판정이 아닙니다. 사진은 삭제하거나 이동하지 않았습니다.</p>
<header><label>파일명 검색 <input id="query" placeholder="예: 613"></label>
<label>최소 유사도 <input id="minimum" type="number" min="-1" max="1" step="0.01" value="{minimum:.3f}"></label>
<span id="count"></span></header>
{''.join(cards) if cards else '<p>설정한 기준 이상의 후보가 없습니다. --threshold를 낮추고 새 결과 폴더로 다시 실행할 수 있습니다.</p>'}
<script type="application/json" id="thumbnails">{images_json}</script>
<script>
const images=JSON.parse(document.getElementById('thumbnails').textContent);
document.querySelectorAll('img[data-image]').forEach(img=>img.src=images[img.dataset.image]);
function filter(){{
 const query=document.getElementById('query').value.toLowerCase();
 const minimum=Number(document.getElementById('minimum').value);let count=0;
 document.querySelectorAll('article').forEach(card=>{{
   card.hidden=Number(card.dataset.score)<minimum||!card.dataset.name.toLowerCase().includes(query);
   if(!card.hidden)count++;
 }});document.getElementById('count').textContent=count+'쌍 표시';
}}
document.getElementById('query').addEventListener('input',filter);
document.getElementById('minimum').addEventListener('input',filter);filter();
</script></html>'''
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "similar_photos.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_a", "image_b", "similarity", "global_similarity",
                         "center_similarity", "match_kind"])
        for pair in pairs:
            writer.writerow([str(paths[pair.left]), str(paths[pair.right]), f"{pair.similarity:.6f}",
                             f"{pair.global_similarity:.6f}", f"{pair.center_similarity:.6f}",
                             "near_duplicate_candidate"])
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="사진 폴더 (하위 폴더 포함)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("similar_results"))
    parser.add_argument("--threshold", type=float, default=0.90, help="최소 유사도 (기본: 0.90)")
    parser.add_argument("--top-k", type=int, default=5, help="사진당 가장 비슷한 후보 수 (기본: 5)")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.directory.is_dir():
        parser.error(f"사진 폴더가 없습니다: {args.directory}")
    if args.output_dir.exists():
        parser.error(f"결과 폴더가 이미 있습니다. 새 -o 이름을 지정하세요: {args.output_dir}")
    if not -1 <= args.threshold <= 1 or args.top_k < 1 or args.batch_size < 1:
        parser.error("threshold는 -1~1, top-k와 batch-size는 1 이상이어야 합니다.")
    paths = sorted(path.resolve() for path in args.directory.rglob("*")
                   if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not paths:
        print("검사할 사진이 없습니다.", file=sys.stderr)
        return 2
    try:
        valid, whole, center, errors = extract_embeddings(paths, args.batch_size)
        if not valid:
            for path, error in errors:
                print(f"읽기 실패: {path}: {error}", file=sys.stderr)
            raise RuntimeError("읽을 수 있는 사진이 없습니다.")
        pairs = select_pairs(whole, center, args.threshold, args.top_k)
        write_results(valid, pairs, args.output_dir)
    except (RuntimeError, OSError, ValueError) as error:
        print(f"검사 실패: {error}", file=sys.stderr)
        return 2
    print(f"검사: {len(valid)}장 / 유사 사진 검토 후보: {len(pairs)}쌍")
    print(f"사진 비교: {(args.output_dir / 'index.html').resolve()}")
    print(f"CSV: {(args.output_dir / 'similar_photos.csv').resolve()}")
    for path, error in errors:
        print(f"읽기 실패: {path}: {error}", file=sys.stderr)
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
