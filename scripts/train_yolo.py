"""
YOLOv8 학습 스크립트
야생동물 탐지 모델 학습
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_yolo(
    data_yaml: str,
    model: str = "yolov8s.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "train"
):
    """
    YOLOv8 모델 학습
    
    Args:
        data_yaml: data.yaml 파일 경로
        model: 사용할 모델 (yolov8s.pt, yolov8n.pt 등)
        epochs: 학습 에포크 수
        imgsz: 이미지 크기
        batch: 배치 크기
        device: 사용할 디바이스 ("0" = GPU 0, "cpu" = CPU)
        project: 프로젝트 디렉토리
        name: 실행 이름
    """
    # data.yaml 경로 확인
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml 파일을 찾을 수 없습니다: {data_yaml}")
    
    # 절대 경로로 변환
    data_yaml_abs = str(data_path.resolve())
    
    print("=" * 60)
    print("YOLOv8 학습 시작")
    print(f"데이터셋: {data_yaml_abs}")
    print(f"모델: {model}")
    print(f"에포크: {epochs}")
    print(f"이미지 크기: {imgsz}")
    print(f"배치 크기: {batch}")
    print(f"디바이스: {device}")
    print("=" * 60)
    
    # 모델 로드
    model = YOLO(model)
    
    # 학습 실행
    results = model.train(
        data=data_yaml_abs,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        save=True,
        plots=True
    )
    
    print("=" * 60)
    print("학습 완료!")
    print(f"결과 저장 위치: {project}/{name}")
    print(f"최적 모델: {project}/{name}/weights/best.pt")
    print("=" * 60)
    
    return results


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="YOLOv8 야생동물 탐지 모델 학습"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="output/dataset/data.yaml",
        help="data.yaml 파일 경로 (기본값: output/dataset/data.yaml)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="사용할 모델 (기본값: yolov8s.pt)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="학습 에포크 수 (기본값: 100)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="이미지 크기 (기본값: 640)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="배치 크기 (기본값: 16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="사용할 디바이스 (기본값: 0, GPU 0번 사용)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="프로젝트 디렉토리 (기본값: runs/detect)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="실행 이름 (기본값: train)"
    )
    
    args = parser.parse_args()
    
    train_yolo(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )


if __name__ == "__main__":
    main()