"""
YOLO 모델에 클래스 이름 설정 스크립트
학습이 끝난 모델에 클래스 이름을 추가합니다.
"""

from pathlib import Path
from ultralytics import YOLO
import torch

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 모델 경로
MODEL_PATH = PROJECT_ROOT / "runs/detect/runs/detect/train/weights/best.pt"

# data.yaml 경로
DATA_YAML_PATH = PROJECT_ROOT / "output/dataset/data.yaml"

# 클래스 이름 정의 (data.yaml과 동일한 순서)
CLASS_NAMES = {
    0: 'bird',
    1: 'cat',
    2: 'dog',
    3: 'duck',
    4: 'fox',
    5: 'magpie',
    6: 'magpie2_500',
    7: 'pheasant',
    8: 'rabbit',
    9: 'raccoon_dog',
    10: 'roe_deer',
    11: 'water_deer',
    12: 'weasel',
    13: 'wild_boar'
}


def set_class_names(model_path: Path = None, class_names: dict = None, data_yaml: Path = None):
    """
    YOLO 모델에 클래스 이름을 설정합니다.
    
    Args:
        model_path: 모델 파일 경로 (기본값: best.pt)
        class_names: 클래스 이름 딕셔너리 (기본값: CLASS_NAMES)
        data_yaml: data.yaml 파일 경로 (선택사항)
    """
    if model_path is None:
        model_path = MODEL_PATH
    if class_names is None:
        class_names = CLASS_NAMES
    if data_yaml is None:
        data_yaml = DATA_YAML_PATH
    
    # 모델 파일 존재 확인
    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    
    print("=" * 60)
    print("YOLO 모델 클래스 이름 설정")
    print(f"모델 경로: {model_path}")
    print("=" * 60)
    
    # 모델 로드
    print("모델 로딩 중...")
    model = YOLO(str(model_path))
    
    # 방법 1: 모델 체크포인트 파일 직접 수정 (가장 확실한 방법)
    print("모델 체크포인트에 클래스 이름 저장 중...")
    try:
        ckpt = torch.load(str(model_path), map_location='cpu')
        
        # names를 최상위 레벨에 저장
        ckpt['names'] = class_names
        
        # model 키가 있는 경우
        if 'model' in ckpt:
            # 모델 객체인 경우
            if hasattr(ckpt['model'], 'names'):
                ckpt['model'].names = class_names
            # dict 형태인 경우
            elif isinstance(ckpt['model'], dict) and 'names' in ckpt['model']:
                ckpt['model']['names'] = class_names
        
        # 모델 저장
        torch.save(ckpt, str(model_path))
        print("✅ 모델 파일에 클래스 이름 저장 완료")
        
        # 모델 다시 로드하여 변경사항 적용
        model = YOLO(str(model_path))
        
    except Exception as e:
        print(f"⚠️ 모델 파일 수정 실패: {e}")
        print("내부 속성에 직접 설정 시도...")
        
        # 방법 2: 내부 속성에 직접 접근
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'names'):
                # 내부 모델의 names 속성 수정
                model.model.names = class_names
                print("✅ 내부 모델 속성에 클래스 이름 설정 완료")
            elif hasattr(model, 'overrides'):
                # overrides를 통해 설정
                if model.overrides is None:
                    model.overrides = {}
                model.overrides['names'] = class_names
                print("✅ overrides에 클래스 이름 설정 완료")
            else:
                print("⚠️ 클래스 이름을 설정할 수 없습니다.")
                print("추론 시 클래스 이름을 수동으로 매핑해야 합니다.")
        except Exception as e2:
            print(f"⚠️ 내부 속성 설정 실패: {e2}")
    
    # 현재 설정된 클래스 이름 확인
    current_names = model.names if hasattr(model, 'names') else None
    if current_names:
        print("\n✅ 클래스 이름 설정 완료!")
        print("\n설정된 클래스 이름:")
        for idx, name in sorted(current_names.items()):
            print(f"  {idx:2d}: {name}")
    else:
        print("\n⚠️ 클래스 이름을 확인할 수 없습니다.")
        print("추론 시 클래스 이름 매핑을 사용하세요.")
    
    print("\n" + "=" * 60)
    print("이제 추론 시 클래스 이름이 표시됩니다.")
    print("사용 예시:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{model_path}')")
    print(f"  model.predict('이미지경로.jpg', save=True)")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    # 스크립트 실행
    model = set_class_names()
    
    if model is not None:
        print("\n💡 팁:")
        print("   - 모델을 로드할 때 data.yaml을 지정하면 클래스 이름이 자동으로 적용됩니다.")
        print("   - 예: model = YOLO('best.pt', data='output/dataset/data.yaml')")