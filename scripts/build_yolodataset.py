"""
YOLO 데이터셋 빌드 (raw_data 기준 전면 작성)
- raw_data 아래 모든 이미지를 포함 (JSON 유무 무관).
- JSON 없으면 해당 이미지의 폴더명을 클래스로 하고, 전체 이미지 bbox(0.5,0.5,1,1)로 라벨 생성.
- 클래스 = raw_data 직하위 폴더 이름 (14개 클래스).
"""

import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# 프로젝트 루트
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.schemas import (
    load_json,
    validate_json_structure,
    get_bbox_from_annotation,
    get_category_name,
    JSONSchemaError,
)
from src.converter import convert_annotation_to_yolo_line
from src.io_utils import (
    copy_image,
    save_yolo_label,
    save_class_mapping,
    logger,
)
from src.generate_yaml import generate_yaml


# 이미지 확장자
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]


def get_top_level_class_folders(raw_dir: Path) -> List[str]:
    """raw_data 직하위 폴더 이름을 클래스로 사용 (순서 고정)."""
    if not raw_dir.exists():
        return []
    dirs = [
        p.name
        for p in sorted(raw_dir.iterdir())
        if p.is_dir() and not p.name.startswith(".")
    ]
    return dirs


def get_class_from_path(img_path: Path, raw_dir: Path) -> Optional[str]:
    """이미지 경로가 raw_dir 아래 어떤 직하위 폴더에 있는지 반환. 예: raw_data/duck/... -> duck"""
    try:
        rel = img_path.resolve().relative_to(raw_dir.resolve())
        return rel.parts[0] if rel.parts else None
    except (ValueError, IndexError):
        return None


def find_all_images_and_optional_json(raw_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
    """
    raw_data 아래 모든 이미지를 나열하고, 같은 직하위 폴더 안에서 같은 stem의 JSON이 있으면 매칭.
    JSON 없어도 이미지는 반드시 포함. (이미지 경로, JSON 경로 또는 None)
    """
    # (직하위폴더명, stem) -> json_path (한 폴더 안에서 stem당 하나)
    try:
        raw_resolved = raw_dir.resolve()
    except Exception:
        raw_resolved = raw_dir
    json_by_folder_stem = {}
    for json_path in raw_dir.rglob("*.json"):
        try:
            rel = json_path.resolve().relative_to(raw_resolved)
            top = rel.parts[0] if rel.parts else ""
            json_by_folder_stem[(top, json_path.stem)] = json_path
        except ValueError:
            pass

    entries = []
    for ext in IMAGE_EXTENSIONS:
        for img_path in raw_dir.rglob(ext):
            try:
                rel = img_path.resolve().relative_to(raw_resolved)
                top = rel.parts[0] if rel.parts else ""
            except ValueError:
                top = ""
            json_path = json_by_folder_stem.get((top, img_path.stem))
            entries.append((img_path, json_path))

    logger.info(f"총 {len(entries)}개 이미지 (JSON 있음: {sum(1 for _, j in entries if j is not None)}개, JSON 없음: {sum(1 for _, j in entries if j is None)}개)")
    return entries


def get_image_size(img_path: Path, json_data: Optional[dict]) -> Tuple[int, int]:
    """이미지 크기: JSON이 있으면 images[0]에서, 없으면 PIL로 읽기."""
    if json_data and json_data.get("images"):
        img_info = json_data["images"][0]
        return int(img_info["width"]), int(img_info["height"])
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.size[0], im.size[1]
    except Exception as e:
        logger.warning(f"이미지 크기 읽기 실패 ({img_path}): {e}, 1x1 사용")
        return 1, 1


def unique_output_stem(img_path: Path, raw_dir: Path) -> str:
    """출력 파일명 충돌 방지: raw_data 기준 상대경로를 _ 로 이어붙임."""
    try:
        rel = img_path.resolve().relative_to(raw_dir.resolve())
        stem = rel.with_suffix("")
        return str(stem).replace("\\", "_").replace("/", "_")
    except ValueError:
        return img_path.stem


def process_one_image(
    img_path: Path,
    json_path: Optional[Path],
    raw_dir: Path,
    class_mapping: dict,
    top_folders: List[str],
) -> Tuple[List[str], int, int]:
    """
    한 장 처리: YOLO 라벨 라인 리스트, width, height 반환.
    JSON 없으면 클래스는 경로(폴더명), bbox는 전체 이미지(0.5,0.5,1,1).
    """
    json_data = None
    if json_path and json_path.exists():
        try:
            json_data = load_json(json_path)
            validate_json_structure(json_data, json_path)
        except (JSONSchemaError, Exception) as e:
            logger.debug(f"JSON 로드/검증 실패 ({json_path}): {e}, 경로 기반으로 처리")

    width, height = get_image_size(img_path, json_data)
    path_class = get_class_from_path(img_path, raw_dir)
    if path_class not in class_mapping and path_class is not None:
        # 클래스 매핑에 없는 폴더면 스킵하지 말고, 매핑에 추가되어 있어야 함. 이미 14개로 만들었으므로 있어야 함.
        pass

    yolo_lines = []

    if json_data and json_data.get("annotations"):
        for ann in json_data["annotations"]:
            category_name = get_category_name(ann, json_data) or path_class
            if not category_name or category_name not in class_mapping:
                continue
            bbox = get_bbox_from_annotation(ann)
            if bbox is None:
                continue
            class_id = class_mapping[category_name]
            line = convert_annotation_to_yolo_line(class_id, bbox, width, height)
            if line:
                yolo_lines.append(line)

    # JSON이 없거나 annotations가 비어 있으면 전체 이미지를 하나의 객체로
    if not yolo_lines and path_class and path_class in class_mapping:
        class_id = class_mapping[path_class]
        # YOLO 정규화: 중심 0.5, 0.5, 너비 1, 높이 1
        yolo_lines.append(f"{class_id} 0.5 0.5 1.0 1.0")

    return yolo_lines, width, height


def main():
    config = Config.from_args()
    raw_dir = config.raw_dir

    logger.info("=" * 60)
    logger.info("YOLO 데이터셋 변환 (모든 이미지 포함, JSON 없어도 폴더 기준 클래스 적용)")
    logger.info(f"입력: {raw_dir}")
    logger.info(f"출력: {config.dataset_dir}")
    logger.info("=" * 60)

    # 1) 클래스 = raw_data 직하위 폴더 이름 (14개)
    top_folders = get_top_level_class_folders(raw_dir)
    if not top_folders:
        logger.error("raw_data 직하위 폴더가 없습니다.")
        return
    logger.info(f"클래스(폴더) 개수: {len(top_folders)} — {top_folders}")

    # 2) 모든 이미지 + 선택적 JSON
    entries = find_all_images_and_optional_json(raw_dir)
    if not entries:
        logger.error("이미지를 찾을 수 없습니다.")
        return

    # 3) 클래스 매핑: raw_data 직하위 14개 폴더 이름만 사용 (classes.txt 덮어씀)
    class_names_ordered = sorted(top_folders)
    class_mapping = save_class_mapping(config.classes_file, class_names_ordered)
    logger.info(f"클래스 매핑: {class_mapping}")

    # 4) train/val 분할 (이미지 경로만으로 분할, JSON 유무 무관)
    shuffled = entries.copy()
    random.seed(config.seed)
    random.shuffle(shuffled)
    val_size = int(len(shuffled) * config.val_ratio)
    val_entries = shuffled[:val_size]
    train_entries = shuffled[val_size:]
    logger.info(f"Train: {len(train_entries)}개, Val: {len(val_entries)}개")

    # 5) 출력 디렉터리
    images_train = config.images_train_dir
    images_val = config.images_val_dir
    labels_train = config.labels_train_dir
    labels_val = config.labels_val_dir
    for d in (images_train, images_val, labels_train, labels_val):
        d.mkdir(parents=True, exist_ok=True)

    stats = {"train_ok": 0, "train_skip": 0, "val_ok": 0, "val_skip": 0}

    def process_split(entries_list, images_dir, labels_dir, stat_ok, stat_skip):
        for img_path, json_path in entries_list:
            path_class = get_class_from_path(img_path, raw_dir)
            if path_class not in class_mapping:
                stats[stat_skip] += 1
                continue
            yolo_lines, w, h = process_one_image(
                img_path, json_path, raw_dir, class_mapping, top_folders
            )
            unique = unique_output_stem(img_path, raw_dir)
            dst_img = images_dir / (unique + img_path.suffix)
            dst_label = labels_dir / (unique + ".txt")
            if not copy_image(img_path, dst_img):
                stats[stat_skip] += 1
                continue
            if save_yolo_label(dst_label, yolo_lines):
                stats[stat_ok] += 1
            else:
                stats[stat_skip] += 1

    logger.info("Train 세트 처리 중...")
    process_split(train_entries, images_train, labels_train, "train_ok", "train_skip")
    logger.info("Val 세트 처리 중...")
    process_split(val_entries, images_val, labels_val, "val_ok", "val_skip")

    # 6) data.yaml 생성
    generate_yaml(config.yaml_file, config.dataset_dir, class_names_ordered)

    logger.info("=" * 60)
    logger.info("변환 완료")
    logger.info(f"Train: {stats['train_ok']} 처리, {stats['train_skip']} 스킵")
    logger.info(f"Val: {stats['val_ok']} 처리, {stats['val_skip']} 스킵")
    logger.info(f"출력: {config.dataset_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
