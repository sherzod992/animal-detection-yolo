"""
5단계: YOLO → Stage1 CNN(오탐 제거) → Stage2 CNN(종 분류) 2단계 추론
이미지/폴더 입력, 최종 탐지 결과(박스+종) 출력 및 시각화
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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


def load_models(yolo_path: Path, cnn1_path: Path, cnn2_path: Path, cnn_img_size: int = 224):
    """YOLO, Stage1 CNN, Stage2 CNN 로드. Stage2 class_names.json도 로드."""
    from ultralytics import YOLO
    yolo = YOLO(str(yolo_path))
    cnn1 = None
    cnn2 = None
    stage2_class_names = []
    if cnn1_path.exists():
        import tensorflow as tf
        cnn1 = tf.keras.models.load_model(str(cnn1_path))
    if cnn2_path.exists():
        import tensorflow as tf
        cnn2 = tf.keras.models.load_model(str(cnn2_path))
        names_file = cnn2_path.parent / "class_names.json"
        if names_file.exists():
            with open(names_file, encoding="utf-8") as f:
                stage2_class_names = json.load(f)
    return yolo, cnn1, cnn2, stage2_class_names, cnn_img_size


def preprocess_crop_for_cnn(crop_bgr_or_rgb: np.ndarray, size: int) -> np.ndarray:
    """Crop (H,W,3) 0-255 → (1, size, size, 3) float32. 학습 시 image_dataset_from_directory는 rescale 없음(0-255)."""
    import tensorflow as tf
    img = tf.image.resize(crop_bgr_or_rgb, (size, size))
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)


def run_two_stage(
    image_path: Path,
    yolo,
    cnn1,
    cnn2,
    stage2_class_names: list,
    cnn_img_size: int,
    yolo_conf: float = 0.25,
    padding_ratio: float = 0.0,
):
    """
    한 장 이미지에 대해 YOLO → crop → Stage1 → Stage2 실행.
    반환: list of dict { "box": [x1,y1,x2,y2], "animal": bool, "species": str or None, "yolo_conf": float, "cnn1_prob": float }
    """
    from PIL import Image
    import tensorflow as tf
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]

    results = yolo.predict(str(image_path), conf=yolo_conf, verbose=False)
    if not results or len(results) == 0:
        return []
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    out = []
    for box, c in zip(xyxy, conf):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        if padding_ratio > 0:
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - bw * padding_ratio / 2))
            y1 = max(0, int(y1 - bh * padding_ratio / 2))
            x2 = min(w, int(x2 + bw * padding_ratio / 2))
            y2 = min(h, int(y2 + bh * padding_ratio / 2))
        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        X = preprocess_crop_for_cnn(crop, cnn_img_size)
        is_animal = True
        cnn1_prob = 1.0
        species = None
        if cnn1 is not None:
            p1 = cnn1.predict(X, verbose=0)[0]
            cnn1_prob = float(p1[0])
            is_animal = np.argmax(p1) == 0
        if is_animal and cnn2 is not None and len(stage2_class_names) > 0:
            p2 = cnn2.predict(X, verbose=0)[0]
            idx = int(np.argmax(p2))
            species = stage2_class_names[idx] if idx < len(stage2_class_names) else None
        out.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "animal": is_animal,
            "species": species,
            "yolo_conf": float(c),
            "cnn1_prob": cnn1_prob,
        })
    return out


def draw_detections(image_path: Path, detections: list, out_path: Path, stage2_class_names: list):
    """탐지 결과를 이미지에 그려 저장. animal인 경우에만 species 라벨."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        if d["animal"] and d.get("species"):
            label = d["species"]
            color = (0, 255, 0)
        else:
            label = "non_animal"
            color = (255, 100, 100)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1 - 18), label, fill=color, font=font)
    img.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="YOLO → Stage1 CNN → Stage2 CNN 2단계 추론"
    )
    parser.add_argument(
        "source",
        type=Path,
        help="입력 이미지 경로 또는 이미지가 있는 폴더",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "cnn_stages.yaml",
        help="cnn_stages.yaml 경로",
    )
    parser.add_argument(
        "--yolo",
        type=Path,
        default=None,
        help="YOLO 모델 경로 (.pt 또는 .onnx). 비우면 config의 yolo_model_pt 사용",
    )
    parser.add_argument(
        "--cnn-stage1",
        type=Path,
        default=None,
        help="Stage1 CNN 모델 (best.keras). 비우면 runs/cnn/stage1/best.keras",
    )
    parser.add_argument(
        "--cnn-stage2",
        type=Path,
        default=None,
        help="Stage2 CNN 모델 (best.keras). 비우면 runs/cnn/stage2/best.keras",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO 최소 confidence",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
        help="crop bbox 확장 비율",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="출력 이미지 저장 폴더 (지정 시 시각화 결과 저장)",
    )
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="시각화 이미지 저장 안 함",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"설정 파일 없음: {config_path}")
        sys.exit(1)
    config = load_config(config_path)
    inf = config.get("inference", {})
    yolo_model_pt = inf.get("yolo") or config.get("yolo_model_pt", "runs/detect/runs/detect/train2/weights/best.pt")
    yolo_path = args.yolo or (PROJECT_ROOT / yolo_model_pt)
    cnn1_path = args.cnn_stage1 or (PROJECT_ROOT / inf.get("cnn_stage1", "runs/cnn/stage1/best.keras"))
    cnn2_path = args.cnn_stage2 or (PROJECT_ROOT / inf.get("cnn_stage2", "runs/cnn/stage2/best.keras"))
    if not yolo_path.is_absolute():
        yolo_path = PROJECT_ROOT / yolo_path
    if not cnn1_path.is_absolute():
        cnn1_path = PROJECT_ROOT / cnn1_path
    if not cnn2_path.is_absolute():
        cnn2_path = PROJECT_ROOT / cnn2_path

    if not yolo_path.exists():
        print(f"YOLO 모델 없음: {yolo_path}")
        sys.exit(1)
    stage2_class_names = config["stage2"]["class_names"]
    train_defaults = config.get("train_defaults", {})
    cnn_img_size = train_defaults.get("img_size", 224)

    print("모델 로딩...")
    yolo, cnn1, cnn2, loaded_class_names, cnn_img_size = load_models(
        yolo_path, cnn1_path, cnn2_path, cnn_img_size
    )
    if loaded_class_names:
        stage2_class_names = loaded_class_names
    if cnn1 is None:
        print("경고: Stage1 CNN 없음. YOLO 결과만 사용 (모두 animal로 간주).")
    if cnn2 is None:
        print("경고: Stage2 CNN 없음. 종 분류 생략.")

    source = args.source if args.source.is_absolute() else PROJECT_ROOT / args.source
    if not source.exists():
        print(f"입력 없음: {source}")
        sys.exit(1)
    if source.is_file():
        image_paths = [source]
    else:
        image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(source.glob(ext))
        image_paths = sorted(image_paths)
    if not image_paths:
        print("이미지 파일 없음.")
        sys.exit(1)

    out_dir = args.out_dir
    if out_dir is not None and not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    if out_dir is not None and not args.no_draw:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("2단계 추론 시작")
    print(f"입력: {len(image_paths)}장, YOLO conf>={args.conf}")
    print("=" * 60)
    for img_path in image_paths:
        detections = run_two_stage(
            img_path, yolo, cnn1, cnn2, stage2_class_names, cnn_img_size,
            yolo_conf=args.conf, padding_ratio=args.padding,
        )
        n_animal = sum(1 for d in detections if d["animal"])
        print(f"{img_path.name}: YOLO 박스 {len(detections)}개 → 동물 {n_animal}개")
        if out_dir and not args.no_draw:
            out_path = out_dir / f"{img_path.stem}_out{img_path.suffix}"
            draw_detections(img_path, detections, out_path, stage2_class_names)
            print(f"  저장: {out_path}")
    print("=" * 60)
    print("완료.")


if __name__ == "__main__":
    main()
