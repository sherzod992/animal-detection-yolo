"""
입출력 유틸리티 모듈
파일 탐색, 복사, 저장, 로깅 기능을 제공합니다.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def find_image_json_pairs(raw_dir: Path) -> List[Tuple[Path, Path]]:
    """
    raw_data 디렉토리에서 이미지와 JSON 파일 쌍을 찾습니다.
    
    재귀적으로 탐색하여 같은 이름의 .jpg와 .json 파일을 매칭합니다.
    
    Args:
        raw_dir: raw_data 디렉토리 경로
        
    Returns:
        (이미지 경로, JSON 경로) 튜플 리스트
    """
    pairs = []
    json_files = {}
    image_files = {}
    
    # 모든 JSON 파일 찾기
    for json_path in raw_dir.rglob("*.json"):
        stem = json_path.stem  # 확장자 제외한 파일명
        json_files[stem] = json_path
    
    # 모든 이미지 파일 찾기 (.jpg, .jpeg, .png)
    for ext in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]:
        for img_path in raw_dir.rglob(ext):
            stem = img_path.stem
            image_files[stem] = img_path
    
    # 매칭
    matched_stems = set(json_files.keys()) & set(image_files.keys())
    
    for stem in matched_stems:
        pairs.append((image_files[stem], json_files[stem]))
    
    logger.info(f"총 {len(pairs)}개의 이미지-JSON 쌍을 찾았습니다.")
    return pairs


def load_class_mapping(classes_file: Path) -> Optional[Dict[str, int]]:
    """
    기존 classes.txt에서 클래스 매핑을 로드합니다.
    
    Args:
        classes_file: classes.txt 파일 경로
        
    Returns:
        {category_name: class_id} 딕셔너리 또는 None (파일이 없는 경우)
    """
    if not classes_file.exists():
        return None
    
    mapping = {}
    with open(classes_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            class_name = line.strip()
            if class_name:
                mapping[class_name] = idx
    
    logger.info(f"기존 클래스 매핑을 로드했습니다: {len(mapping)}개 클래스")
    return mapping


def save_class_mapping(classes_file: Path, class_names: List[str]) -> Dict[str, int]:
    """
    클래스 매핑을 classes.txt에 저장합니다.
    
    Args:
        classes_file: 저장할 파일 경로
        class_names: 클래스 이름 리스트 (인덱스 순서대로)
        
    Returns:
        {category_name: class_id} 딕셔너리
    """
    classes_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(classes_file, 'w', encoding='utf-8') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    mapping = {name: idx for idx, name in enumerate(class_names)}
    logger.info(f"클래스 매핑을 저장했습니다: {len(mapping)}개 클래스")
    return mapping


def build_class_mapping(
    all_category_names: Set[str],
    classes_file: Path
) -> Dict[str, int]:
    """
    클래스 매핑을 구축합니다.
    기존 파일이 있으면 재사용하고, 없으면 새로 생성합니다.
    
    Args:
        all_category_names: 모든 category_name 집합
        classes_file: classes.txt 파일 경로
        
    Returns:
        {category_name: class_id} 딕셔너리
    """
    # 기존 매핑 로드
    existing_mapping = load_class_mapping(classes_file)
    
    if existing_mapping is not None:
        new_classes = all_category_names - set(existing_mapping.keys())
        if new_classes:
            # 기존 순서 유지 + 새 클래스를 알파벳 순으로 뒤에 추가
            existing_names_ordered = sorted(
                existing_mapping.keys(),
                key=lambda name: existing_mapping[name]
            )
            merged_names = existing_names_ordered + sorted(new_classes)
            logger.info(
                f"새 클래스 추가: {sorted(new_classes)}. "
                f"전체 클래스 수: {len(merged_names)}"
            )
            return save_class_mapping(classes_file, merged_names)
        return existing_mapping
    
    # 새 매핑 생성 (알파벳 순서로 정렬하여 일관성 유지)
    sorted_names = sorted(all_category_names)
    return save_class_mapping(classes_file, sorted_names)


def copy_image(src: Path, dst: Path) -> bool:
    """
    이미지 파일을 복사합니다.
    
    Args:
        src: 원본 경로
        dst: 대상 경로
        
    Returns:
        성공 여부
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        logger.error(f"이미지 복사 실패 ({src} -> {dst}): {e}")
        return False


def save_yolo_label(label_path: Path, lines: List[str]) -> bool:
    """
    YOLO 형식 라벨 파일을 저장합니다.
    
    Args:
        label_path: 저장할 파일 경로
        lines: 각 줄의 문자열 리스트
        
    Returns:
        성공 여부
    """
    try:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            if lines:  # 마지막 줄에 개행 추가
                f.write('\n')
        return True
    except Exception as e:
        logger.error(f"라벨 저장 실패 ({label_path}): {e}")
        return False