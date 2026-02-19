"""
데이터 분할 모듈
이미지와 라벨을 train/val로 분할합니다.
"""

import random
from pathlib import Path
from typing import List, Tuple


def split_data(
    file_pairs: List[Tuple[Path, Path]],
    val_ratio: float,
    seed: int = 42
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    파일 쌍을 train/val로 분할합니다.
    
    Args:
        file_pairs: (이미지 경로, JSON 경로) 튜플 리스트
        val_ratio: 검증 세트 비율
        seed: 랜덤 시드
        
    Returns:
        (train_pairs, val_pairs) 튜플
    """
    if seed is not None:
        random.seed(seed)
    
    # 셔플
    shuffled = file_pairs.copy()
    random.shuffle(shuffled)
    
    # 분할
    val_size = int(len(shuffled) * val_ratio)
    val_pairs = shuffled[:val_size]
    train_pairs = shuffled[val_size:]
    
    return train_pairs, val_pairs