#!/usr/bin/env python3
"""raw_data JSON 구조 분석 - category_name vs category_id 등."""
import json
import sys
from pathlib import Path

def main():
    # 프로젝트 루트 = 스크립트 기준 상위
    project_root = Path(__file__).resolve().parent.parent
    base = project_root / "raw_data"
    # cwd가 프로젝트 루트일 때 결과 파일
    out_path = project_root / "inspect_raw_data_result.txt"
    if not base.exists():
        with open(out_path, "w", encoding="utf-8") as out:
            out.write("raw_data not found\n")
        return
    all_names = set()
    all_ids = set()
    has_categories = False
    categories_list = None
    sample_ann = None
    jsons = list(base.rglob("*.json"))
    for j in jsons[:200]:
            try:
                with open(j, encoding="utf-8") as f:
                    d = json.load(f)
                for ann in d.get("annotations", []):
                    if "category_name" in ann:
                        all_names.add(ann["category_name"])
                    if "category_id" in ann:
                        all_ids.add(ann["category_id"])
                if d.get("categories") is not None:
                    has_categories = True
                    categories_list = d["categories"]
                if sample_ann is None and d.get("annotations"):
                    sample_ann = d["annotations"][0]
            except Exception as e:
                pass
    lines = []
    lines.append("=== category_name (string) values ===")
    lines.append(str(sorted(all_names)))
    lines.append("=== category_id values ===")
    lines.append(str(sorted(all_ids)))
    lines.append("=== Top-level 'categories' in JSON? ===")
    lines.append(str(has_categories))
    if categories_list is not None:
        lines.append("categories sample: " + str(categories_list[:3] if isinstance(categories_list, list) else categories_list))
    lines.append("=== Sample annotation keys ===")
    if sample_ann is not None:
        lines.append(str(list(sample_ann.keys())))
        lines.append("sample: " + str(sample_ann))
    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(text)
    print(text)

if __name__ == "__main__":
    main()
