"""
Stage1 / Stage2 CNN을 TFLite로 내보내기 (Jetson 등 엣지 배포용)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_keras_to_tflite(
    keras_path: Path,
    tflite_path: Path,
    quantize: str = "none",
    img_size: int = 224,
):
    """
    keras_path: best.keras 또는 final.keras
    tflite_path: 출력 .tflite 경로
    quantize: "none" | "float16" | "int8"
    """
    import tensorflow as tf
    model = tf.keras.models.load_model(str(keras_path))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    print(f"저장: {tflite_path}")
    return tflite_path


def main():
    parser = argparse.ArgumentParser(description="Stage1/Stage2 CNN → TFLite 내보내기")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
        help="1=animal vs non_animal, 2=8종",
    )
    parser.add_argument(
        "--keras",
        type=Path,
        default=None,
        help="입력 .keras 경로. 비우면 runs/cnn/stageN/best.keras",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="출력 .tflite 경로. 비우면 runs/cnn/stageN/best_float32.tflite",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "float16", "int8"],
        default="none",
        help="양자화: none, float16, int8",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="입력 크기 (기록용, 변환에는 미사용)",
    )
    args = parser.parse_args()

    if args.stage == 1:
        default_keras = PROJECT_ROOT / "runs" / "cnn" / "stage1" / "best.keras"
        default_out = PROJECT_ROOT / "runs" / "cnn" / "stage1" / "best_float32.tflite"
    else:
        default_keras = PROJECT_ROOT / "runs" / "cnn" / "stage2" / "best.keras"
        default_out = PROJECT_ROOT / "runs" / "cnn" / "stage2" / "best_float32.tflite"

    keras_path = args.keras or default_keras
    if not keras_path.is_absolute():
        keras_path = PROJECT_ROOT / keras_path
    if not keras_path.exists():
        print(f"Keras 모델 없음: {keras_path}")
        print("먼저 train_cnn.py --stage", args.stage, "로 학습하세요.")
        sys.exit(1)

    out_path = args.out or default_out
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    if args.quantize == "float16":
        out_path = out_path.parent / (out_path.stem.replace("float32", "float16") + ".tflite")
    elif args.quantize == "int8":
        out_path = out_path.parent / (out_path.stem.replace("float32", "int8") + ".tflite")

    print("=" * 60)
    print(f"Stage{args.stage} CNN → TFLite (quantize={args.quantize})")
    print("=" * 60)
    export_keras_to_tflite(keras_path, out_path, quantize=args.quantize, img_size=args.img_size)
    print("완료.")


if __name__ == "__main__":
    main()
