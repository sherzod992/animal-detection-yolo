"""
3단계: 1단계 CNN용 non_animal(오탐) 샘플 수집
YOLO 추론 결과 중 GT와 겹치지 않는 예측을 오탐으로 간주해 crop하여 non_animal/에 저장합니다.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: Path) -> dict:
    if yaml is None:
        raise ImportError("PyYAML이 필요합니다: pip install pyyaml")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_yolo_label_to_xyxy(line: str, img_w: int, img_h: int):
    """YOLO 한 줄 → (x1,y1,x2,y2) 픽셀 좌표. 유효하지 않으면 None."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except (ValueError, IndexError):
        return None
    xc = cx * img_w
    yc = cy * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = max(0, int(xc - bw / 2))
    y1 = max(0, int(yc - bh / 2))
    x2 = min(img_w, int(xc + bw / 2))
    y2 = min(img_h, int(yc + bh / 2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def box_iou(box_a: tuple, box_b: tuple) -> float:
    """(x1,y1,x2,y2) 두 개 → IoU."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def get_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list:
    """라벨 파일에서 모든 GT bbox (x1,y1,x2,y2) 리스트 반환."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        b = parse_yolo_label_to_xyxy(line, img_w, img_h)
        if b is not None:
            boxes.append(b)
    return boxes


def process_split(
    images_dir: Path,
    labels_dir: Path,
    non_animal_dir: Path,
    model,
    split: str,
    conf_thresh: float,
    iou_thresh: float,
    max_per_image: int,
    padding_ratio: float,
    extensions: tuple = (".jpg", ".jpeg", ".png"),
):
    from PIL import Image

    out_dir = non_animal_dir / split / "non_animal"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))

    total = 0
    skipped_no_label = 0
    skipped_no_pred = 0

    for img_path in sorted(image_files):
        label_path = labels_dir / (img_path.stem + ".txt")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        gt_boxes = get_gt_boxes(label_path, w, h)

        results = model.predict(str(img_path), verbose=False)
        if not results or len(results) == 0:
            skipped_no_pred += 1
            continue
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        collected = 0
        for idx, (box, c) in enumerate(zip(xyxy, conf)):
            if c < conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            pred_box = (x1, y1, x2, y2)
            max_iou = 0.0
            for gt in gt_boxes:
                max_iou = max(max_iou, box_iou(pred_box, gt))
            if max_iou >= iou_thresh:
                continue
            if max_per_image > 0 and collected >= max_per_image:
                break
            if padding_ratio > 0:
                bw = x2 - x1
                bh = y2 - y1
                pad_w = bw * padding_ratio / 2
                pad_h = bh * padding_ratio / 2
                x1 = max(0, int(x1 - pad_w))
                y1 = max(0, int(y1 - pad_h))
                x2 = min(w, int(x2 + pad_w))
                y2 = min(h, int(y2 + pad_h))
            crop = img.crop((x1, y1, x2, y2))
            crop.save(out_dir / f"{img_path.stem}_fp{idx}.jpg", quality=95)
            collected += 1
            total += 1
        if not label_path.exists():
            skipped_no_label += 1

    return total, skipped_no_label, skipped_no_pred


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 오탐(FP) crop을 non_animal로 수집 (Stage1 CNN용)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "cnn_stages.yaml",
        help="cnn_stages.yaml 경로",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="YOLO 가중치 경로 (.pt). 비우면 config의 yolo_model_pt 사용",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="오탐 후보로 쓸 예측 최소 confidence (미지정 시 config collect_non_animal.conf)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="GT와 IoU가 이 값 미만이면 오탐으로 간주 (미지정 시 config collect_non_animal.iou)",
    )
    parser.add_argument(
        "--max-per-image",
        type=int,
        default=None,
        help="이미지당 최대 non_animal crop 수 (미지정 시 config collect_non_animal.max_per_image)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=None,
        help="crop bbox 확장 비율 (미지정 시 config collect_non_animal.padding)",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"설정 파일 없음: {config_path}")
        sys.exit(1)
    config = load_config(config_path)
    cna = config.get("collect_non_animal", {})
    conf = args.conf if args.conf is not None else cna.get("conf", 0.25)
    iou = args.iou if args.iou is not None else cna.get("iou", 0.3)
    max_per_image = args.max_per_image if args.max_per_image is not None else cna.get("max_per_image", 5)
    padding = args.padding if args.padding is not None else cna.get("padding", 0.1)

    yolo_cfg = config["yolo_dataset"]
    stage1_dir = PROJECT_ROOT / config["stage1"]["dataset_dir"]
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = PROJECT_ROOT / config.get(
            "yolo_model_pt", yolo_cfg.get("yolo_model_pt", "runs/detect/runs/detect/train2/weights/best.pt")
        )
    images_train = PROJECT_ROOT / yolo_cfg["images_train"]
    images_val = PROJECT_ROOT / yolo_cfg["images_val"]
    labels_train = PROJECT_ROOT / yolo_cfg["labels_train"]
    labels_val = PROJECT_ROOT / yolo_cfg["labels_val"]

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        print(f"모델 파일 없음: {model_path}")
        sys.exit(1)

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    print("=" * 60)
    print("non_animal(오탐) crop 수집")
    print(f"Stage1 디렉터리: {stage1_dir}")
    print(f"YOLO 모델: {model_path}")
    print(f"conf >= {conf}, IoU < {iou} → 오탐, max_per_image={max_per_image}, padding={padding}")
    print("=" * 60)

    for split, im_dir, lb_dir in [
        ("train", images_train, labels_train),
        ("val", images_val, labels_val),
    ]:
        n, skip_label, skip_pred = process_split(
            im_dir, lb_dir, stage1_dir, model, split,
            conf, iou, max_per_image, padding,
        )
        print(f"[{split}] non_animal crop 수: {n}, (라벨 없음 스킵: {skip_label}, 예측 없음 스킵: {skip_pred})")

    print("=" * 60)
    print("완료. Stage1 train/val 아래 non_animal/ 폴더를 확인하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
