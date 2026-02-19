"""
JSON → YOLO 형식 변환 모듈
bbox 좌표를 YOLO 정규화 형식으로 변환합니다.
"""

from typing import List, Optional, Tuple


def normalize_bbox(
    bbox: List[List[float]],
    width: int,
    height: int
) -> Optional[Tuple[float, float, float, float]]:
    """
    bbox를 정규화하고 YOLO 형식으로 변환합니다.
    
    입력: [[x1, y1], [x2, y2]]
    처리:
    1. 좌표 보정 (min/max)
    2. clamp (0~W, 0~H)
    3. YOLO 정규화 (cx, cy, w, h)
    
    Args:
        bbox: [[x1, y1], [x2, y2]] 형식의 bbox
        width: 이미지 너비
        height: 이미지 높이
        
    Returns:
        (cx, cy, w, h) 정규화된 좌표 또는 None (유효하지 않은 경우)
    """
    if len(bbox) != 2 or len(bbox[0]) != 2 or len(bbox[1]) != 2:
        return None
    
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    
    # 보정: min/max로 정렬
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # clamp: 이미지 범위 내로 제한
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    
    # 너비/높이가 0이면 유효하지 않음
    if x2 <= x1 or y2 <= y1:
        return None
    
    # YOLO 정규화
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    
    # 정규화 값이 0~1 범위를 벗어나면 clamp (반올림 오차 대비)
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    return cx, cy, w, h


def convert_annotation_to_yolo_line(
    class_id: int,
    bbox: List[List[float]],
    width: int,
    height: int
) -> Optional[str]:
    """
    annotation을 YOLO 형식 라인으로 변환합니다.
    
    Args:
        class_id: 클래스 ID
        bbox: [[x1, y1], [x2, y2]] 형식의 bbox
        width: 이미지 너비
        height: 이미지 높이
        
    Returns:
        "<class_id> <cx> <cy> <w> <h>" 형식의 문자열 또는 None
    """
    normalized = normalize_bbox(bbox, width, height)
    if normalized is None:
        return None
    
    cx, cy, w, h = normalized
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"