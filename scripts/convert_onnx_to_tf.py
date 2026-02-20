"""
ONNX → TensorFlow SavedModel 변환 스크립트
YOLOv8 ONNX 모델을 TensorFlow SavedModel 형식으로 변환
"""

import argparse
from pathlib import Path

import onnx
from onnx_tf.backend import prepare


def convert_onnx_to_tf(
    onnx_path: str,
    output_dir: str,
    device: str = "CPU"
) -> str:
    """
    ONNX 모델을 TensorFlow SavedModel로 변환

    Args:
        onnx_path: ONNX 모델 파일 경로 (.onnx)
        output_dir: SavedModel 저장 경로
        device: 변환 시 사용할 디바이스 ("CPU" 또는 "CUDA")

    Returns:
        SavedModel이 저장된 경로
    """
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX 파일을 찾을 수 없습니다: {onnx_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ONNX → TensorFlow 변환 시작")
    print(f"입력: {onnx_file.resolve()}")
    print(f"출력: {output_path.resolve()}")
    print("=" * 60)

    # ONNX 모델 로드
    onnx_model = onnx.load(str(onnx_file))

    # TensorFlow 백엔드로 변환
    tf_rep = prepare(onnx_model, device=device)

    # SavedModel으로 저장
    tf_rep.export_graph(str(output_path))

    print("=" * 60)
    print("변환 완료!")
    print(f"SavedModel 저장 위치: {output_path.resolve()}")
    print("=" * 60)

    return str(output_path)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="ONNX 모델을 TensorFlow SavedModel로 변환"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="runs/detect/runs/detect/train/weights/best.onnx",
        help="ONNX 모델 파일 경로 (기본값: runs/detect/runs/detect/train/weights/best.onnx)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/saved_model",
        help="SavedModel 저장 디렉토리 (기본값: output/saved_model)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        choices=["CPU", "CUDA"],
        help="변환 시 사용할 디바이스 (기본값: CPU)"
    )

    args = parser.parse_args()

    convert_onnx_to_tf(
        onnx_path=args.onnx,
        output_dir=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()
