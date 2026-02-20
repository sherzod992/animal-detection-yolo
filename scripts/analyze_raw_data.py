#!/usr/bin/env python3
"""raw_data 구조 분석 - 결과를 analyze_result.txt에 저장."""
import json
from pathlib import Path

def main():
    base = Path(__file__).resolve().parent.parent / "raw_data"
    out = Path(__file__).resolve().parent.parent / "analyze_result.txt"
    lines = []
    if not base.exists():
        lines.append("raw_data not found")
        out.write_text("\n".join(lines), encoding="utf-8")
        return

    # 최상위 폴더 = 클래스 후보
    top_dirs = sorted(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))
    lines.append("=== Top-level folders (classes) ===")
    lines.append(str(top_dirs))
    lines.append("")

    # 이미지/JSON 개수
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    all_imgs = []
    for e in exts:
        all_imgs.extend(base.rglob(e))
    all_jsons = list(base.rglob("*.json"))
    lines.append("=== Counts ===")
    lines.append(f"Total images: {len(all_imgs)}")
    lines.append(f"Total JSONs: {len(all_jsons)}")
    lines.append("")

    # 이미지 중 JSON 있는 것 / 없는 것
    json_stems = {p.stem for p in all_jsons}
    imgs_with_json = [p for p in all_imgs if p.stem in json_stems]
    imgs_no_json = [p for p in all_imgs if p.stem not in json_stems]
    lines.append("Images with matching JSON: " + str(len(imgs_with_json)))
    lines.append("Images WITHOUT JSON: " + str(len(imgs_no_json)))
    lines.append("")

    # 폴더별 이미지/JSON 개수
    lines.append("=== Per-folder (top dir) ===")
    for d in top_dirs:
        folder = base / d
        imgs = []
        for e in exts:
            imgs.extend(folder.rglob(e))
        jsons = list(folder.rglob("*.json"))
        lines.append(f"{d}: images={len(imgs)}, jsons={len(jsons)}")
    lines.append("")

    # 샘플 JSON 구조 하나
    if all_jsons:
        with open(all_jsons[0], encoding="utf-8") as f:
            sample = json.load(f)
        lines.append("=== Sample JSON keys ===")
        lines.append(str(list(sample.keys())))
        if sample.get("annotations"):
            lines.append("First annotation keys: " + str(list(sample["annotations"][0].keys())))
        if sample.get("categories"):
            lines.append("Categories (first 3): " + str((sample["categories"] or [])[:3]))

    out.write_text("\n".join(lines), encoding="utf-8")
    print("Written to", out)

if __name__ == "__main__":
    main()
