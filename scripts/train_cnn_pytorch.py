"""
PyTorch 버전 CNN 학습 (RTX 50 시리즈 GPU 지원)
Stage1: animal vs non_animal | Stage2: 14종 분류
MobileNetV2 기반, train_cnn.py(Keras)와 동일한 데이터/구조 사용
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    yaml = None

# ImageNet 정규화 (학습 시 동일하게 사용)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_config(config_path: Path) -> dict:
    if yaml is None:
        raise ImportError("PyYAML이 필요합니다: pip install pyyaml")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(num_classes: int, img_size: int = 224):
    """MobileNetV2 기반 분류 모델 (Keras 버전과 유사 구조)."""
    base = mobilenet_v2(weights="IMAGENET1K_V1")
    # features만 사용 (conv layers) → (N, 1280, 7, 7). classifier는 pooling 포함하므로 제외
    features = base.features
    model = nn.Sequential(
        features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(1280, num_classes),
    )
    return model


def get_transforms(img_size: int, is_train: bool):
    """학습/검증용 transform."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


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
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    if use_gpu and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"train/ 또는 val/ 없음: {data_dir}")

    train_ds = ImageFolder(
        str(train_dir),
        transform=get_transforms(img_size, is_train=True),
    )
    val_ds = ImageFolder(
        str(val_dir),
        transform=get_transforms(img_size, is_train=False),
    )
    class_names = train_ds.classes
    print(f"✅ 데이터셋 로딩 완료 (train: {len(train_ds)}, val: {len(val_ds)})")
    print(f"클래스: {class_names}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_gpu,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_gpu,
    )

    num_classes = 2
    model = build_model(num_classes, img_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = outputs.max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()

        train_acc = train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
        val_acc = val_correct / val_total

        print(f"Epoch {epoch + 1}/{epochs}  train_loss: {train_loss / len(train_loader):.4f}  "
              f"train_acc: {train_acc:.4f}  val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping (patience={patience})")
                break

    torch.save(model.state_dict(), output_dir / "final.pt")
    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False)
    print(f"최고 val_acc: {best_val_acc:.4f}, 저장: {output_dir}")


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
    """Stage2: 14종 다중 분류."""
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    if use_gpu and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"train/ 또는 val/ 없음: {data_dir}")

    train_ds = ImageFolder(
        str(train_dir),
        transform=get_transforms(img_size, is_train=True),
    )
    val_ds = ImageFolder(
        str(val_dir),
        transform=get_transforms(img_size, is_train=False),
    )
    ds_class_names = train_ds.classes
    num_classes = len(ds_class_names)
    if num_classes != len(class_names):
        print(f"경고: 데이터셋 클래스 수={num_classes}, config={len(class_names)}. 데이터셋 기준 사용.")
    print(f"✅ 데이터셋 로딩 완료 (train: {len(train_ds)}, val: {len(val_ds)})")
    print(f"클래스: {ds_class_names}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_gpu,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_gpu,
    )

    model = build_model(num_classes, img_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    patience = 6
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = outputs.max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()

        train_acc = train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
        val_acc = val_correct / val_total

        print(f"Epoch {epoch + 1}/{epochs}  train_loss: {train_loss / len(train_loader):.4f}  "
              f"train_acc: {train_acc:.4f}  val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping (patience={patience})")
                break

    torch.save(model.state_dict(), output_dir / "final.pt")
    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(list(ds_class_names), f, ensure_ascii=False)
    print(f"최고 val_acc: {best_val_acc:.4f}, 저장: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch Stage1/Stage2 CNN 학습 (RTX 50 GPU 지원)")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
        help="1=animal vs non_animal, 2=14종 분류",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "cnn_stages.yaml",
        help="cnn_stages.yaml 경로",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cpu-only", action="store_true", help="CPU만 사용")
    args = parser.parse_args()

    use_gpu = not args.cpu_only
    if args.cpu_only:
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
        print("PyTorch Stage1 CNN: animal vs non_animal")
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
        print("PyTorch Stage2 CNN: 14종 분류")
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
    print("학습 완료. 모델: best.pt, final.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
