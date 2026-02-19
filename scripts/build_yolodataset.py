"""
YOLO 데이터셋 빌드 엔트리포인트
전체 변환 파이프라인을 실행합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.schemas import (
    load_json,
    validate_json_structure,
    extract_image_info,
    extract_annotations,
    get_bbox_from_annotation,
    get_category_name,
    JSONSchemaError
)
from src.converter import convert_annotation_to_yolo_line
from src.splitter import split_data
from src.io_utils import (
    find_image_json_pairs,
    build_class_mapping,
    copy_image,
    save_yolo_label,
    logger
)
from src.generate_yaml import generate_yaml


def process_single_file(
    img_path: Path,
    json_path: Path,
    class_mapping: dict,
    width: int,
    height: int
) -> tuple[list[str], set[str]]:
    """
    단일 파일을 처리하여 YOLO 라벨 라인을 생성합니다.
    
    Args:
        img_path: 이미지 파일 경로
        json_path: JSON 파일 경로
        class_mapping: 클래스 매핑 딕셔너리
        width: 이미지 너비
        height: 이미지 높이
        
    Returns:
        (yolo_lines, category_names) 튜플
    """
    yolo_lines = []
    category_names = set()
    
    try:
        data = load_json(json_path)
        validate_json_structure(data, json_path)
        annotations = extract_annotations(data)
        
        for ann in annotations:
            category_name = get_category_name(ann)
            if category_name is None:
                continue
            
            category_names.add(category_name)
            
            if category_name not in class_mapping:
                logger.warning(
                    f"클래스 매핑에 없는 category_name: {category_name} "
                    f"({json_path})"
                )
                continue
            
            class_id = class_mapping[category_name]
            bbox = get_bbox_from_annotation(ann)
            
            if bbox is None:
                continue
            
            line = convert_annotation_to_yolo_line(class_id, bbox, width, height)
            if line is not None:
                yolo_lines.append(line)
        
    except JSONSchemaError as e:
        logger.error(f"JSON 스키마 오류 ({json_path}): {e}")
    except Exception as e:
        logger.error(f"파일 처리 오류 ({json_path}): {e}")
    
    return yolo_lines, category_names


def main():
    """메인 실행 함수"""
    # 설정 로드
    config = Config.from_args()
    
    logger.info("=" * 60)
    logger.info("YOLO 데이터셋 변환 시작")
    logger.info(f"입력 디렉토리: {config.raw_dir}")
    logger.info(f"출력 디렉토리: {config.out_dir}")
    logger.info(f"검증 비율: {config.val_ratio}")
    logger.info(f"랜덤 시드: {config.seed}")
    logger.info("=" * 60)
    
    # 1. 이미지-JSON 쌍 찾기
    file_pairs = find_image_json_pairs(config.raw_dir)
    if len(file_pairs) == 0:
        logger.error("이미지-JSON 쌍을 찾을 수 없습니다.")
        return
    
    # 2. 모든 category_name 수집
    all_category_names = set()
    valid_pairs = []
    
    logger.info("JSON 파일 분석 중...")
    for img_path, json_path in file_pairs:
        try:
            data = load_json(json_path)
            validate_json_structure(data, json_path)
            annotations = extract_annotations(data)
            
            # annotations가 비어있으면 스킵
            if len(annotations) == 0:
                logger.debug(f"annotations가 비어있어 스킵: {json_path}")
                continue
            
            # category_name 수집
            for ann in annotations:
                category_name = get_category_name(ann)
                if category_name:
                    all_category_names.add(category_name)
            
            # 이미지 정보 추출
            file_name, width, height = extract_image_info(data)
            
            # 이미지 파일 존재 확인
            if not img_path.exists():
                logger.warning(f"이미지 파일이 없습니다: {img_path}")
                continue
            
            valid_pairs.append((img_path, json_path, width, height))
            
        except JSONSchemaError as e:
            logger.warning(f"JSON 스키마 오류로 스킵: {json_path} - {e}")
        except Exception as e:
            logger.warning(f"파일 분석 실패로 스킵: {json_path} - {e}")
    
    logger.info(f"유효한 파일 쌍: {len(valid_pairs)}개")
    logger.info(f"발견된 클래스: {sorted(all_category_names)}")
    
    if len(valid_pairs) == 0:
        logger.error("처리할 유효한 파일이 없습니다.")
        return
    
    # 3. 클래스 매핑 구축
    class_mapping = build_class_mapping(all_category_names, config.classes_file)
    logger.info(f"클래스 매핑: {class_mapping}")
    
    # 4. 데이터 분할
    pairs_for_split = [(img, json_path) for img, json_path, _, _ in valid_pairs]
    train_pairs, val_pairs = split_data(
        pairs_for_split,
        config.val_ratio,
        config.seed
    )
    
    logger.info(f"Train: {len(train_pairs)}개, Val: {len(val_pairs)}개")
    
    # 5. 파일 처리 및 복사
    stats = {
        'train_processed': 0,
        'train_skipped': 0,
        'val_processed': 0,
        'val_skipped': 0
    }
    
    # Train 처리
    logger.info("Train 세트 처리 중...")
    valid_pairs_dict = {(img, json): (w, h) for img, json, w, h in valid_pairs}
    
    for img_path, json_path in train_pairs:
        width, height = valid_pairs_dict[(img_path, json_path)]
        yolo_lines, _ = process_single_file(
            img_path, json_path, class_mapping, width, height
        )
        
        if len(yolo_lines) == 0:
            stats['train_skipped'] += 1
            continue
        
        # 이미지 복사
        img_name = img_path.name
        dst_img = config.images_train_dir / img_name
        if not copy_image(img_path, dst_img):
            stats['train_skipped'] += 1
            continue
        
        # 라벨 저장
        label_name = img_path.stem + ".txt"
        dst_label = config.labels_train_dir / label_name
        if save_yolo_label(dst_label, yolo_lines):
            stats['train_processed'] += 1
        else:
            stats['train_skipped'] += 1
    
    # Val 처리
    logger.info("Val 세트 처리 중...")
    for img_path, json_path in val_pairs:
        width, height = valid_pairs_dict[(img_path, json_path)]
        yolo_lines, _ = process_single_file(
            img_path, json_path, class_mapping, width, height
        )
        
        if len(yolo_lines) == 0:
            stats['val_skipped'] += 1
            continue
        
        # 이미지 복사
        img_name = img_path.name
        dst_img = config.images_val_dir / img_name
        if not copy_image(img_path, dst_img):
            stats['val_skipped'] += 1
            continue
        
        # 라벨 저장
        label_name = img_path.stem + ".txt"
        dst_label = config.labels_val_dir / label_name
        if save_yolo_label(dst_label, yolo_lines):
            stats['val_processed'] += 1
        else:
            stats['val_skipped'] += 1
    
    # 6. data.yaml 생성
    class_names = sorted(class_mapping.keys())
    generate_yaml(config.yaml_file, config.dataset_dir, class_names)
    
    # 7. 결과 출력
    logger.info("=" * 60)
    logger.info("변환 완료!")
    logger.info(f"Train: {stats['train_processed']}개 처리, {stats['train_skipped']}개 스킵")
    logger.info(f"Val: {stats['val_processed']}개 처리, {stats['val_skipped']}개 스킵")
    logger.info(f"출력 디렉토리: {config.dataset_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()