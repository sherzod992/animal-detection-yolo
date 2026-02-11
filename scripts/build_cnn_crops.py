"""
YOLO 라벨 기반 CNN용 crop 데이터셋 생성
- Stage1: 오탐 제거용 → animal/ (YOLO 정답 bbox crop)
- Stage2: 종 분류용 → 클래스별 폴더 (개, 고라니, ...)
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: Path) -> dict:
    """config/cnn_stages.yaml 로드."""
    if yaml is None:
        raise ImportError("PyYAML이 필요합니다: pip install pyyaml")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def yolo_line_to_bbox(line: str, img_w: int, img_h: int, padding_ratio: float = 0.0):
    """
    YOLO 한 줄 (class_id cx cy w h) → 픽셀 bbox (x1, y1, x2, y2), class_id.
    padding_ratio: bbox 확장 비율 (0.1 = 10%)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None, None
    try:
        cid = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except (ValueError, IndexError):
        return None, None
    xc = cx * img_w
    yc = cy * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2
    if padding_ratio > 0:
        pad_w = bw * padding_ratio / 2
        pad_h = bh * padding_ratio / 2
        x1 -= pad_w
        y1 -= pad_h
        x2 += pad_w
        y2 += pad_h
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_w, int(x2))
    y2 = min(img_h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None, None
    return (x1, y1, x2, y2), cid


def process_split(
    images_dir: Path,
    labels_dir: Path,
    stage1_dir: Path,
    stage2_dir: Path,
    class_names: list,
    split: str,
    padding_ratio: float,
    extensions: tuple = (".jpg", ".jpeg", ".png"),
):
    """한 split(train/val)에 대해 이미지별 라벨 읽고 crop 저장."""
    from PIL import Image

    stage1_animal = stage1_dir / split / "animal"
    stage1_animal.mkdir(parents=True, exist_ok=True)
    for name in class_names:
        (stage2_dir / split / name).mkdir(parents=True, exist_ok=True)

    total_crops = 0
    skipped_no_label = 0
    skipped_invalid = 0

    image_files = []
    for ext in extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))

    for img_path in sorted(image_files):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped_no_label += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped_invalid += 1
            continue
        w, h = img.size
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        for idx, line in enumerate(lines):
            bbox, cid = yolo_line_to_bbox(line, w, h, padding_ratio)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            crop = img.crop((x1, y1, x2, y2))
            stem_idx = f"{img_path.stem}_{idx}"
            # Stage1: animal
            crop.save(stage1_animal / f"{stem_idx}.jpg", quality=95)
            # Stage2: species
            if 0 <= cid < len(class_names):
                out_path = stage2_dir / split / class_names[cid] / f"{stem_idx}.jpg"
                crop.save(out_path, quality=95)
            total_crops += 1

    return total_crops, skipped_no_label, skipped_invalid


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 라벨로 CNN용 crop 데이터셋 생성 (stage1 animal, stage2 species)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "cnn_stages.yaml",
        help="cnn_stages.yaml 경로",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
        help="bbox 확장 비율 (0.1 = 10%%, 기본 0.1)",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        print(f"설정 파일 없음: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    yolo = config["yolo_dataset"]
    stage1_dir = PROJECT_ROOT / config["stage1"]["dataset_dir"]
    stage2_dir = PROJECT_ROOT / config["stage2"]["dataset_dir"]
    class_names = config["stage2"]["class_names"]

    images_train = PROJECT_ROOT / yolo["images_train"]
    images_val = PROJECT_ROOT / yolo["images_val"]
    labels_train = PROJECT_ROOT / yolo["labels_train"]
    labels_val = PROJECT_ROOT / yolo["labels_val"]

    for d in (images_train, images_val, labels_train, labels_val):
        if not d.exists():
            print(f"경로 없음: {d}")
            sys.exit(1)

    print("=" * 60)
    print("CNN crop 데이터셋 생성")
    print(f"Stage1: {stage1_dir}")
    print(f"Stage2: {stage2_dir}")
    print(f"Padding: {args.padding * 100:.0f}%")
    print("=" * 60)

    for split, im_dir, lb_dir in [
        ("train", images_train, labels_train),
        ("val", images_val, labels_val),
    ]:
        n_crops, n_no_label, n_invalid = process_split(
            im_dir, lb_dir, stage1_dir, stage2_dir, class_names, split, args.padding
        )
        print(f"[{split}] crop 수: {n_crops}, 라벨 없음 스킵: {n_no_label}, 이미지 오류 스킵: {n_invalid}")

    print("=" * 60)
    print("완료.")
    print(f"Stage1 animal: {stage1_dir}/train/animal, {stage1_dir}/val/animal")
    print(f"Stage2 species: {stage2_dir}/train|<class>/", f"{stage2_dir}/val/<class>/")
    print("=" * 60)


if __name__ == "__main__":
    main()
