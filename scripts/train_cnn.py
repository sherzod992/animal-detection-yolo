"""
4단계: Stage1(animal vs non_animal) / Stage2(14종) CNN 학습
Keras(TensorFlow) + MobileNetV2 기반, Jetson·TFLite 배포 고려
Stage2는 YOLO와 동일한 14개 클래스로 분류합니다.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _is_rtx50_series_gpu() -> bool:
    """RTX 50 시리즈(compute capability 12.0) 감지. TensorFlow 미지원."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            name = result.stdout.strip().upper()
            return any(x in name for x in ["5060", "5070", "5080", "5090", "RTX 50"])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False
sys.path.insert(0, str(PROJECT_ROOT))

# GPU CUDA 오류(cuLaunchKernel 등) 방지를 위한 환경변수
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
# XLA JIT 비활성화 (일부 GPU에서 cuLaunchKernel InternalError 해결)
if 'TF_XLA_FLAGS' not in os.environ:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'

try:
    import yaml
except ImportError:
    yaml = None


def setup_gpu_memory(use_gpu: bool = True):
    """GPU 메모리 설정 (메모리 증가 허용)."""
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and use_gpu:
            try:
                # GPU 메모리 증가 허용 (필요한 만큼만 사용)
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU 메모리 증가 허용 설정 완료 ({len(gpus)}개 GPU)")
            except RuntimeError as e:
                print(f"⚠️ GPU 메모리 설정 실패: {e}")
                print("CPU로 전환합니다.")
                tf.config.set_visible_devices([], 'GPU')
        else:
            if not use_gpu:
                # GPU 비활성화
                tf.config.set_visible_devices([], 'GPU')
                print("ℹ️ GPU 비활성화됨. CPU로 실행됩니다.")
            else:
                print("ℹ️ GPU를 사용할 수 없습니다. CPU로 실행됩니다.")
    except Exception as e:
        print(f"⚠️ GPU 설정 중 오류: {e}")
        print("CPU로 전환합니다.")
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
        except:
            pass


def load_config(config_path: Path) -> dict:
    if yaml is None:
        raise ImportError("PyYAML이 필요합니다: pip install pyyaml")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(num_classes: int, input_shape=(224, 224, 3)):
    """MobileNetV2 기반 분류 모델 (경량, TFLite 변환 용이)."""
    import tensorflow as tf
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    base.trainable = True
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def train_stage1(
    data_dir: Path,
    output_dir: Path,
    img_size: int = 224,
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 1e-4,
    use_gpu: bool = True,
):
    """Stage1: animal vs non_animal 이진 분류."""
    import tensorflow as tf

    # GPU 정보 출력 (디버깅용)
    if use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow: {tf.__version__}, GPU 감지: {len(gpus)}개")
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  - {gpu.name}")

    # GPU 메모리 설정 (오류 발생 시 자동으로 CPU로 전환)
    try:
        setup_gpu_memory(use_gpu=use_gpu)
    except Exception as e:
        print(f"⚠️ GPU 설정 실패, CPU로 전환: {e}")
        tf.config.set_visible_devices([], 'GPU')
        use_gpu = False
    
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"train/ 또는 val/ 없음: {data_dir}")
    
    # 데이터셋 로딩 (CPU에서 실행하여 GPU 메모리 오류 방지)
    print("데이터셋 로딩 중...")
    with tf.device('/CPU:0'):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="int",
            shuffle=True,
            seed=42,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="int",
            shuffle=False,
        )
    print(f"✅ 데이터셋 로딩 완료 (train: {len(train_ds)}, val: {len(val_ds)})")
    num_classes = 2
    model = build_model(num_classes, (img_size, img_size, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        str(output_dir / "best.keras"),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ckpt, early],
    )
    model.save(str(output_dir / "final.keras"))
    import json
    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(["animal", "non_animal"], f, ensure_ascii=False)
    return model


def train_stage2(
    data_dir: Path,
    output_dir: Path,
    class_names: list,
    img_size: int = 224,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-4,
    use_gpu: bool = True,
):
    """Stage2: 14종 다중 분류 (YOLO와 동일한 클래스). class_names 순서대로 라벨 매핑."""
    import tensorflow as tf
    
    # GPU 메모리 설정 (오류 발생 시 자동으로 CPU로 전환)
    try:
        setup_gpu_memory(use_gpu=use_gpu)
    except Exception as e:
        print(f"⚠️ GPU 설정 실패, CPU로 전환: {e}")
        tf.config.set_visible_devices([], 'GPU')
        use_gpu = False
    
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"train/ 또는 val/ 없음: {data_dir}")
    
    # Keras는 폴더 이름 알파벳 순으로 라벨 부여. class_names 순서와 맞추려면
    # train/ 아래에 class_names 순서대로만 폴더가 있어야 함. 인코딩 이슈로
    # 일부 폴더가 다를 수 있으므로, image_dataset_from_directory 사용 후
    # class_names는 train_ds.class_names로 확인 가능.
    
    # 데이터셋 로딩 (CPU에서 실행하여 GPU 메모리 오류 방지)
    print("데이터셋 로딩 중...")
    with tf.device('/CPU:0'):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="int",
            shuffle=True,
            seed=42,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=(img_size, img_size),
            batch_size=batch_size,
            label_mode="int",
            shuffle=False,
        )
    print(f"✅ 데이터셋 로딩 완료 (train: {len(train_ds)}, val: {len(val_ds)})")
    num_classes = len(train_ds.class_names)
    if num_classes != len(class_names):
        print(f"경고: 데이터셋 클래스 수={num_classes}, config 클래스 수={len(class_names)}. 데이터셋 기준으로 학습합니다.")
    model = build_model(num_classes, (img_size, img_size, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        str(output_dir / "best.keras"),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ckpt, early],
    )
    model.save(str(output_dir / "final.keras"))
    # 라벨 매핑 저장 (추론 시 사용)
    import json
    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(list(train_ds.class_names), f, ensure_ascii=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="Stage1/Stage2 CNN 학습 (Keras)")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
        help="1=animal vs non_animal, 2=14종 분류 (YOLO와 동일한 클래스)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "cnn_stages.yaml",
        help="cnn_stages.yaml 경로",
    )
    parser.add_argument("--epochs", type=int, default=None, help="에포크 (기본: stage1=20, stage2=30)")
    parser.add_argument("--batch", type=int, default=32, help="배치 크기")
    parser.add_argument("--img-size", type=int, default=224, help="입력 이미지 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="모델 저장 경로 (기본: runs/cnn/stage1 또는 stage2)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="GPU를 사용하지 않고 CPU만 사용 (CUDA 오류 시 사용)",
    )
    args = parser.parse_args()

    # CPU 전용 모드: TensorFlow import 전에 GPU 비활성화 (CUDA 오류 방지)
    use_gpu = not args.cpu_only
    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif _is_rtx50_series_gpu():
        # RTX 50 시리즈(cc 12.0): TensorFlow가 CUDA 커널 미지원 → CPU 강제
        print("⚠️ RTX 50 시리즈 감지: TensorFlow가 compute capability 12.0을 아직 지원하지 않습니다.")
        print("   CUDA_ERROR_INVALID_PTX 방지를 위해 CPU 모드로 자동 전환합니다.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        use_gpu = False

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"설정 파일 없음: {config_path}")
        sys.exit(1)
    config = load_config(config_path)

    if args.stage == 1:
        stage_cfg = config["stage1"]
        data_dir = PROJECT_ROOT / stage_cfg["dataset_dir"]
        out_dir = args.out_dir or (PROJECT_ROOT / "runs" / "cnn" / "stage1")
        epochs = args.epochs or 20
        print("=" * 60)
        print("Stage1 CNN 학습: animal vs non_animal")
        print(f"데이터: {data_dir}")
        print(f"저장: {out_dir}")
        print("=" * 60)
        train_stage1(
            data_dir=data_dir,
            output_dir=out_dir,
            img_size=args.img_size,
            batch_size=args.batch,
            epochs=epochs,
            lr=args.lr,
            use_gpu=use_gpu,
        )
    else:
        stage_cfg = config["stage2"]
        data_dir = PROJECT_ROOT / stage_cfg["dataset_dir"]
        out_dir = args.out_dir or (PROJECT_ROOT / "runs" / "cnn" / "stage2")
        class_names = stage_cfg["class_names"]
        epochs = args.epochs or 30
        print("=" * 60)
        print("Stage2 CNN 학습: 14종 분류")
        print(f"클래스: {', '.join(class_names)}")
        print(f"데이터: {data_dir}")
        print(f"저장: {out_dir}")
        print("=" * 60)
        train_stage2(
            data_dir=data_dir,
            output_dir=out_dir,
            class_names=class_names,
            img_size=args.img_size,
            batch_size=args.batch,
            epochs=epochs,
            lr=args.lr,
            use_gpu=use_gpu,
        )
    print("=" * 60)
    print("학습 완료.")
    print("=" * 60)


if __name__ == "__main__":
    main()
