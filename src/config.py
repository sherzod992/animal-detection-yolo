"""
    설정 관리 모듈
    경로, 비율, 시드 등 전역 설정 관리합니다.
"""


import argparse 
from pathlib import Path
from typing import Optional 

class Config:
    """프로젝트 설정 클래스"""
    def __init__(
        self,
        raw_dir: str = "raw_data",
        out_dir: str = "output",
        val_ratio: float = 0.2,
        seed: int = 42
    ):
        self.raw_dir = Path(raw_dir).resolve()
        self.out_dir = Path(out_dir).resolve()
        self.val_ratio = val_ratio
        self.seed = seed

        #출력 경로
        self.dataset_dir = self.out_dir /"dataset"
        self.images_train_dir = self.dataset_dir /"images" /"train"
        self.images_val_dir = self.dataset_dir /"images" /"val"
        self.labels_train_dir = self.dataset_dir /"labels" /"train"
        self.labels_val_dir = self.dataset_dir /"labels" /"val"
        self.classes_file = self.dataset_dir /"classes.txt"
        self.yaml_file = self.dataset_dir /"data.yaml"

    def validate(self) -> None:
        """설정 검증"""
        if not self.raw_dir.exists():
            raise ValueError(f"입력 디렉토리가 존재하지 않습니다: {self.raw_dir}")
        if not (0 < self.val_ratio < 1):
            raise ValueError(f"검증 비율은 0과 1 사이여야 합니다: {self.val_ratio}")
    @classmethod
    def from_args(cls) -> "Config":
        """명령줄 인자로부터 Config 생성"""
        parser = argparse.ArgumentParser(
            description="raw_data를 YOLO 학습용 데이터셋으로 변환"
        )
        parser.add_argument(
            "--raw_dir",
            type=str,
            default="raw_data",
            help="입력 raw_data 디렉토리 경로 (기본값: raw_data)"
        )
        parser.add_argument(
            "--out_dir",
            type=str,
            default="output",
            help="출력 디렉토리 경로 (기본값: output)"
        )
        parser.add_argument(
            "--val_ratio",
            type=float,
            default=0.2,
            help="검증 세트 비율 (기본값: 0.2)"
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="랜덤 시드 (기본값: 42)"
        )
        
        args = parser.parse_args()
        config = cls(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
        config.validate()
        return config