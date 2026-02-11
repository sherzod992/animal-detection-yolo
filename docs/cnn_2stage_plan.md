# CNN 2단계 파이프라인 계획

## 1단계 (완료): CNN 목적 결정

**선택: 오탐 제거 → 종 분류 (2단계)**

- **1단계 CNN**: 오탐 제거 — YOLO가 잡은 영역이 “동물 O” vs “비동물(오탐) X” 이진 분류
- **2단계 CNN**: 종 분류 — 동물로 판별된 crop을 8종(개, 고라니, 고양이, 너구리, 노루, 멧돼지, 조류, 족제비)으로 분류

Jetson 추론 흐름: **YOLO bbox → 1단계 CNN(동물 여부) → 2단계 CNN(종 분류)**.

---

## 설정 파일

- `config/cnn_stages.yaml` — 스테이지별 목적, 클래스, 데이터셋 경로 정의

---

## 데이터셋 구조 (예정)

### 1단계 CNN (오탐 제거)

```
cnn_dataset/stage1_animal_vs_background/
  train/
    animal/      ← YOLO 정답 bbox crop
    non_animal/  ← 오탐 샘플 (별도 수집)
  val/
    animal/
    non_animal/
```

### 2단계 CNN (종 분류)

```
cnn_dataset/stage2_species/
  train/
    개/
    고라니/
    고양이/
    너구리/
    노루/
    멧돼지/
    조류/
    족제비/
  val/
    (동일 8개 클래스 폴더)
```

---

## 작업 단계

| 단계 | 내용 | 비고 |
|------|------|------|
| **2** | YOLO 라벨로 crop 생성 → stage1 `animal/`, stage2 클래스별 폴더 | `scripts/build_cnn_crops.py` |
| **3** | 1단계: `non_animal` 샘플 수집 후 CNN 데이터셋 완성 | `scripts/collect_non_animal_crops.py` |
| **4** | 1단계 CNN 학습 → 2단계 CNN 학습 | `scripts/train_cnn.py` |
| **5** | Jetson에서 YOLO → 1단계 CNN → 2단계 CNN 2단계 추론 | `scripts/run_two_stage_inference.py`, `export_cnn_to_tflite.py` |

---

## 5단계: 2단계 추론 및 Jetson 배포

- **PC에서 2단계 추론** (`scripts/run_two_stage_inference.py`):
  - 입력: 이미지 파일 또는 이미지 폴더
  - 흐름: YOLO bbox → crop → Stage1 CNN(동물 여부) → 동물이면 Stage2 CNN(종) → 결과 시각화
  - 실행 예:
    ```bash
    .venv\Scripts\python scripts\run_two_stage_inference.py output\dataset\images\val --out-dir runs\inference_out --conf 0.25
    ```
  - Stage1/Stage2 CNN이 없으면 YOLO 결과만 사용(모두 동물로 간주).
- **CNN → TFLite 내보내기** (Jetson 등 엣지용, `scripts/export_cnn_to_tflite.py`):
  ```bash
  .venv\Scripts\python scripts\export_cnn_to_tflite.py --stage 1 --quantize float16
  .venv\Scripts\python scripts\export_cnn_to_tflite.py --stage 2 --quantize float16
  ```
  - 출력: `runs/cnn/stage1/best_float32.tflite` (또는 float16/int8)
- **설정**: `config/cnn_stages.yaml`의 `inference` 섹션에서 YOLO/CNN 경로 지정 가능. Jetson에서는 TFLite 경로로 바꿔 사용.

---

## 4단계: CNN 학습

- **스크립트**: `scripts/train_cnn.py` (Keras/TensorFlow, MobileNetV2 기반)
- **Stage1** (animal vs non_animal, 이진 분류):
  ```bash
  .venv\Scripts\python scripts\train_cnn.py --stage 1 --epochs 20 --batch 32
  ```
- **Stage2** (8종 분류):
  ```bash
  .venv\Scripts\python scripts\train_cnn.py --stage 2 --epochs 30 --batch 32
  ```
- **저장 경로**: `runs/cnn/stage1/`, `runs/cnn/stage2/`  
  - `best.keras`, `final.keras`, `class_names.json`
- **참고**: val에 `non_animal`이 없으면 Stage1 검증 시 "1 class" 경고가 나올 수 있음. `collect_non_animal_crops.py`로 val용 오탐도 수집하면 해소됨.

---

## 3단계: non_animal 수집

- **스크립트**: `scripts/collect_non_animal_crops.py`
- **방법**: YOLO로 train/val 이미지 추론 → GT와 IoU가 낮은 예측을 오탐으로 간주해 crop → `stage1_animal_vs_background/train|val/non_animal/`에 저장
- **실행 예** (이미지당 최대 5개, conf≥0.25, IoU<0.3이면 오탐):
  ```bash
  .venv\Scripts\python scripts\collect_non_animal_crops.py --conf 0.25 --iou 0.3 --max-per-image 5
  ```
- **설정**: `config/cnn_stages.yaml`의 `yolo_model_pt`로 YOLO 가중치 경로 지정 가능

---

## 다음 단계 실행 순서 (한 번에 진행 시)

1. **Stage1 CNN 학습** (이미 있으면 생략):  
   `python scripts/train_cnn.py --stage 1 --epochs 20 --batch 32`
2. **Stage2 CNN 학습**:  
   `python scripts/train_cnn.py --stage 2 --epochs 30 --batch 32`
3. **CNN → TFLite 내보내기**:  
   `python scripts/export_cnn_to_tflite.py --stage 1 --quantize float16`  
   `python scripts/export_cnn_to_tflite.py --stage 2 --quantize float16`
4. **2단계 추론 테스트**:  
   `python scripts/run_two_stage_inference.py "이미지경로" --out-dir runs/inference_out`

---

## 다음 작업

- Jetson 보드에서 TFLite(YOLO + Stage1/Stage2 CNN) 로드 후 동일 파이프라인 구현 (플랫폼별 런타임 연동).
