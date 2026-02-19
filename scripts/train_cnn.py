"""
4단계: Stage1(animal vs non_animal) / Stage2(8종) CNN 학습
Keras(TensorFlow) + MobileNetV2 기반, Jetson·TFLite 배포 고려
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    yaml = None


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
):
    """Stage1: animal vs non_animal 이진 분류."""
    import tensorflow as tf
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"train/ 또는 val/ 없음: {data_dir}")
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
):
    """Stage2: 8종 다중 분류. class_names 순서대로 라벨 매핑."""
    import tensorflow as tf
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"train/ 또는 val/ 없음: {data_dir}")
    # Keras는 폴더 이름 알파벳 순으로 라벨 부여. class_names 순서와 맞추려면
    # train/ 아래에 class_names 순서대로만 폴더가 있어야 함. 인코딩 이슈로
    # 일부 폴더가 다를 수 있으므로, image_dataset_from_directory 사용 후
    # class_names는 train_ds.class_names로 확인 가능.
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
        help="1=animal vs non_animal, 2=8종 분류",
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
    args = parser.parse_args()

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
        )
    else:
        stage_cfg = config["stage2"]
        data_dir = PROJECT_ROOT / stage_cfg["dataset_dir"]
        out_dir = args.out_dir or (PROJECT_ROOT / "runs" / "cnn" / "stage2")
        class_names = stage_cfg["class_names"]
        epochs = args.epochs or 30
        print("=" * 60)
        print("Stage2 CNN 학습: 8종 분류")
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
        )
    print("=" * 60)
    print("학습 완료.")
    print("=" * 60)


if __name__ == "__main__":
    main()
