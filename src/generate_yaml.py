"""
YOLO data.yaml 생성 모듈
YOLO 학습에 필요한 data.yaml 파일을 생성합니다.
"""

from pathlib import Path
from typing import List


def generate_yaml(
    yaml_path: Path,
    dataset_dir: Path,
    class_names: List[str]
) -> None:
    """
    YOLO data.yaml 파일을 생성합니다.
    
    Args:
        yaml_path: 저장할 yaml 파일 경로
        dataset_dir: dataset 디렉토리 경로 (상대 경로 계산용)
        class_names: 클래스 이름 리스트
    """
    # 상대 경로 계산 (yaml 파일 기준)
    yaml_dir = yaml_path.parent
    train_path = (yaml_dir / "images" / "train").relative_to(yaml_dir)
    val_path = (yaml_dir / "images" / "val").relative_to(yaml_dir)
    
    # 경로를 문자열로 변환 (Windows 호환)
    train_path_str = str(train_path).replace('\\', '/')
    val_path_str = str(val_path).replace('\\', '/')
    
    content = f"""# YOLO 데이터셋 설정 파일
# 자동 생성됨

# 클래스 개수
nc: {len(class_names)}

# 클래스 이름
names:
"""
    for idx, name in enumerate(class_names):
        content += f"  {idx}: {name}\n"
    
    content += f"""
# 경로 (상대 경로)
train: {train_path_str}
val: {val_path_str}
"""
    
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(content)