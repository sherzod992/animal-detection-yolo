
"""
JSON 스키마 검증 및 타입 정의 모듈
입력 JSON 파일의 구조를 검증하고 파싱합니다.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class JSONSchemaError(Exception):
    """JSON 스키마 검증 오류"""
    pass


def load_json(json_path: Path) -> Dict[str, Any]:
    """
    JSON 파일을 로드합니다.
    
    Args:
        json_path: JSON 파일 경로
        
    Returns:
        파싱된 JSON 딕셔너리
        
    Raises:
        JSONSchemaError: 파일 읽기 또는 파싱 실패 시
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        raise JSONSchemaError(f"JSON 파싱 실패 ({json_path}): {e}")
    except Exception as e:
        raise JSONSchemaError(f"파일 읽기 실패 ({json_path}): {e}")


def validate_json_structure(data: Dict[str, Any], json_path: Path) -> None:
    """
    JSON 구조를 검증합니다.
    
    Args:
        data: 파싱된 JSON 데이터
        json_path: JSON 파일 경로 (에러 메시지용)
        
    Raises:
        JSONSchemaError: 필수 필드가 없거나 형식이 잘못된 경우
    """
    # images 필드 검증
    if "images" not in data:
        raise JSONSchemaError(f"필수 필드 'images'가 없습니다: {json_path}")
    
    if not isinstance(data["images"], list) or len(data["images"]) == 0:
        raise JSONSchemaError(f"'images'는 비어있지 않은 리스트여야 합니다: {json_path}")
    
    image = data["images"][0]
    required_image_fields = ["file_name", "width", "height"]
    for field in required_image_fields:
        if field not in image:
            raise JSONSchemaError(f"이미지에 필수 필드 '{field}'가 없습니다: {json_path}")
    
    # annotations 필드 검증 (없어도 됨, 빈 배열 가능)
    if "annotations" not in data:
        data["annotations"] = []
    
    if not isinstance(data["annotations"], list):
        raise JSONSchemaError(f"'annotations'는 리스트여야 합니다: {json_path}")


def extract_image_info(data: Dict[str, Any]) -> Tuple[str, int, int]:
    """
    JSON에서 이미지 정보를 추출합니다.
    
    Args:
        data: 파싱된 JSON 데이터
        
    Returns:
        (file_name, width, height) 튜플
        
    Raises:
        JSONSchemaError: 필수 필드가 없는 경우
    """
    validate_json_structure(data, Path(""))  # 경로는 검증용이므로 빈 경로 사용
    
    image = data["images"][0]
    file_name = str(image["file_name"])
    width = int(image["width"])
    height = int(image["height"])
    
    return file_name, width, height


def extract_annotations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    JSON에서 annotations 배열을 추출합니다.
    
    Args:
        data: 파싱된 JSON 데이터
        
    Returns:
        annotations 리스트
    """
    return data.get("annotations", [])


def get_bbox_from_annotation(annotation: Dict[str, Any]) -> Optional[List[List[float]]]:
    """
    annotation에서 bbox를 추출합니다.
    
    Args:
        annotation: annotation 딕셔너리
        
    Returns:
        bbox [[x1, y1], [x2, y2]] 또는 None
    """
    if "bbox" not in annotation:
        return None
    
    bbox = annotation["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 2:
        return None
    
    if not all(isinstance(point, list) and len(point) == 2 for point in bbox):
        return None
    
    return bbox


def _build_category_id_to_name(data: Dict[str, Any]) -> Dict[int, str]:
    """
    JSON 최상위 'categories' 배열에서 id -> name 매핑을 만듭니다.
    형식: [{"id": 1, "name": "dog"}, ...] 또는 [{"category_id": 1, "category_name": "dog"}, ...]
    """
    mapping = {}
    for item in data.get("categories") or []:
        if not isinstance(item, dict):
            continue
        cid = item.get("id") or item.get("category_id")
        name = item.get("name") or item.get("category_name")
        if cid is not None and name is not None:
            mapping[int(cid)] = str(name)
    return mapping


def get_category_name(
    annotation: Dict[str, Any],
    data: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    annotation에서 category_name을 추출합니다.
    category_name이 없고 category_id만 있으면, data의 'categories'에서 이름을 찾습니다.
    
    Args:
        annotation: annotation 딕셔너리
        data: 전체 JSON 데이터 (categories 포함 시 category_id 해석용)
        
    Returns:
        category_name 또는 None
    """
    name = (
        annotation.get("category_name")
        or annotation.get("category")
        or annotation.get("name")
    )
    if name is not None and name != "":
        return str(name)
    cid = annotation.get("category_id")
    if cid is not None and data is not None:
        id_to_name = _build_category_id_to_name(data)
        return id_to_name.get(int(cid))
    return None