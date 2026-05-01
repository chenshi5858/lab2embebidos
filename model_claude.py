import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from thop import profile
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGE_SIZE = 96
NUM_CLASSES = 4


@dataclass
class SampleItem:
    path: Path
    label: int


class QuadrantDataset(Dataset):
    def __init__(self, samples: List[SampleItem], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        item = self.samples[index]
        with Image.open(item.path) as img:
            img = img.convert("L")
            if self.transform:
                img = self.transform(img)
        return img, int(item.label)


# ─────────────────────────────────────────────
#  MODELO 1 – FCTiny  (baseline, sin cambios)
# ─────────────────────────────────────────────
class FCTiny(nn.Module):
    """FC de dos capas, muy pequeño. ~1.2 M params → ~4.7 MB. Solo baseline."""
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
#  MODELO 2 – FCSmall  (baseline, sin cambios)
# ─────────────────────────────────────────────
class FCSmall(nn.Module):
    """FC de tres capas con dropout."""
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
#  MODELO 3 – CollapseY_CNN  ★ NUEVO ★
#  Idea clave: colapsar el eje Y (altura) con
#  AdaptiveAvgPool para conservar el eje X.
#  El clasificador opera sobre una "tira" 1×W
#  que contiene la distribución horizontal → la
#  tarea de cuadrante se vuelve trivial.
#  ~10 KB de params.
# ─────────────────────────────────────────────
class CollapseY_CNN(nn.Module):
    """
    Extrae features con 2 capas conv ligeras y luego colapsa
    el eje Y con AdaptiveAvgPool2d(1, W). El vector resultante
    de ancho W captura directamente la posición horizontal.
    Muy pequeño y muy apropiado para esta tarea.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # 1×96×96 → 8×48×48
            nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # 8×48×48 → 16×24×24
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 16×24×24 → 16×12×12
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # Colapsa altura → (B, 16, 1, 12) → (B, 16, 12)
        self.pool_y = nn.AdaptiveAvgPool2d((1, 12))
        self.classifier = nn.Sequential(
            nn.Flatten(),        # 16*12 = 192
            nn.Linear(192, 48),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool_y(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
#  MODELO 4 – DepthwiseCNN  ★ NUEVO ★
#  Convs depthwise separables: misma capacidad
#  receptiva que un CNN estándar pero ~8-10× menos
#  parámetros. BN + Dropout. ~15 KB.
# ─────────────────────────────────────────────
class DepthwiseSeparable(nn.Module):
    """Bloque depthwise separable: DW conv + PW conv + BN + ReLU."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class DepthwiseCNN(nn.Module):
    """
    Stack de bloques depthwise separables con stride=2 para reducción
    espacial. Conserva información posicional (sin global pooling hasta
    el final). Termina con AdaptiveAvgPool(1,6) para capturar eje X.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1, bias=False),  # →48×48
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.body = nn.Sequential(
            DepthwiseSeparable(8, 16, stride=2),    # →24×24
            DepthwiseSeparable(16, 24, stride=2),   # →12×12
            DepthwiseSeparable(24, 32, stride=2),   # →6×6
        )
        # Colapsa Y, conserva X con 6 columnas
        self.pool = nn.AdaptiveAvgPool2d((1, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),       # 32*6 = 192
            nn.Linear(192, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
#  MODELO 5 – SpatialCNN  ★ NUEVO ★
#  Diseño orientado a preservar la estructura
#  espacial en X. Usa kernels rectangulares 1×k
#  para capturar relaciones horizontales explíci-
#  tamente, y kernels k×1 para verticales.
#  Pool asimétrico: colapsa Y, mantiene X.
#  ~25 KB.
# ─────────────────────────────────────────────
class SpatialCNN(nn.Module):
    """
    CNN con kernels asimétricos que explotan la geometría de la tarea:
    - Rama horizontal: kernel 1×7 captura contexto en X
    - Rama vertical:   kernel 7×1 captura contexto en Y
    Se fusionan y se colapsa Y al final.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Extracción inicial compartida
        self.shared = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1, bias=False),   # 48×48
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False),  # 24×24
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # Rama H: kernel horizontal 1×7
        self.branch_h = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # Rama V: kernel vertical 7×1
        self.branch_v = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # Fusión + reducción
        self.fuse = nn.Sequential(
            nn.Conv2d(32, 24, 3, stride=2, padding=1, bias=False),  # 12×12
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )
        # Colapsa Y, conserva 8 columnas en X
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.classifier = nn.Sequential(
            nn.Flatten(),       # 24*8 = 192
            nn.Linear(192, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        x = self.shared(x)
        h = self.branch_h(x)
        v = self.branch_v(x)
        x = torch.cat([h, v], dim=1)
        x = self.fuse(x)
        x = self.pool(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
#  MODELO 6 – LightCNN  ★ NUEVO ★  (extra)
#  Versión ultra-liviana para ESP32-S3.
#  Conv estándar + BN, sin ramas paralelas.
#  ~8 KB. Prioriza velocidad sobre accuracy.
# ─────────────────────────────────────────────
class LightCNN(nn.Module):
    """
    Arquitectura ultra-compacta: 3 conv con stride=2 + BN,
    AdaptiveAvgPool(1,4) que colapsa Y y deja 4 columnas en X,
    clasificador lineal mínimo. Ideal para inferencia en MCU.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1, bias=False),    # 48×48
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 12, 3, stride=2, padding=1, bias=False),   # 24×24
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 16, 3, stride=2, padding=1, bias=False),  # 12×12
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, bias=False),  # 6×6
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 4))   # 4 columnas en X
        self.classifier = nn.Sequential(
            nn.Flatten(),   # 16*4 = 64
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
#  DATA UTILS
# ─────────────────────────────────────────────

def parse_label_line(line: str) -> Tuple[str, int]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return "", -1
    filename = parts[0]
    try:
        quadrant = int(parts[2])
    except ValueError:
        return "", -1
    return filename, quadrant


def gather_samples(base_dir: Path) -> List[SampleItem]:
    samples: List[SampleItem] = []
    dataset_folders = sorted(base_dir.glob("dataset_grupo_*_reescalado"))
    for dataset_folder in dataset_folders:
        for subfolder_name in ["celular", "esp"]:
            subfolder = dataset_folder / subfolder_name
            if not subfolder.is_dir():
                continue
            labels_file = subfolder / "etiquetas.txt"
            if not labels_file.exists():
                continue
            lines = labels_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                filename, quadrant = parse_label_line(line)
                if quadrant not in (0, 1, 2, 3):
                    continue
                image_path = subfolder / filename
                if not image_path.exists():
                    continue
                samples.append(SampleItem(path=image_path, label=quadrant))
    return samples


def split_samples(samples: List[SampleItem], train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    train_end = int(n_total * train_ratio)
    val_end = train_end + int(n_total * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
#  TRANSFORMS
#  Augmentation solo en train:
#   - Flip horizontal aleatorio (la imagen puede
#     estar en cualquier lado)
#   - Pequeña rotación (±10°)
#   - Jitter de brillo/contraste (imágenes de
#     cámara y ESP tienen condiciones distintas)
#  Normalize con media/std de grayscale estándar.
# ─────────────────────────────────────────────
NORMALIZE = transforms.Normalize(mean=[0.5], std=[0.5])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    NORMALIZE,
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    NORMALIZE,
])


def build_models():
    return [
        ("fc_tiny",       FCTiny()),
        ("fc_small",      FCSmall()),
        ("collapse_y",    CollapseY_CNN()),
        ("depthwise",     DepthwiseCNN()),
        ("spatial_cnn",   SpatialCNN()),
        ("light_cnn",     LightCNN()),
    ]


# ─────────────────────────────────────────────
#  TRAINING LOOP
#  Mejoras respecto al original:
#   - LR scheduler: ReduceLROnPlateau (baja LR
#     si val_loss no mejora en 3 epochs)
#   - Early stopping: detiene si no mejora en
#     `patience` epochs (evita sobreajuste)
#   - Guarda el mejor estado por val_acc
# ─────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion=None):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            if criterion is not None:
                loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total else 0.0
    loss_avg = loss_sum / total if total else 0.0
    return acc, loss_avg


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int = 7,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_acc = -1.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        train_loss = running_loss / total if total else 0.0
        val_acc, val_loss = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch}")
                break

        if epoch % 5 == 0:
            print(f"    Epoch {epoch:3d} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc, history


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_ops(model: nn.Module, input_shape=(1, 1, IMAGE_SIZE, IMAGE_SIZE)):
    model.eval()
    dummy = torch.zeros(input_shape)
    try:
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception:
        macs = 0.0
    return macs, macs * 2


def export_onnx(model: nn.Module, onnx_path: Path):
    model.eval()
    dummy = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model, dummy, onnx_path.as_posix(),
        input_names=["input"], output_names=["logits"],
        opset_version=18, do_constant_folding=True,
    )


def save_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_report_md(path: Path, rows: List[dict]):
    lines = [
        "# Model Report",
        "",
        "| model | val_acc | test_acc | params | size_kb | macs | flops | onnx |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {val_acc:.4f} | {test_acc:.4f} | {params} "
            "| {size_kb:.1f} | {macs:.2e} | {flops:.2e} | {onnx} |".format(**row)
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",    type=str, default=None)
    parser.add_argument("--output_dir",  type=str, default="outputs")
    parser.add_argument("--epochs",      type=int, default=40)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio",   type=float, default=0.2)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience",    type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).parent.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "onnx").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    samples = gather_samples(base_dir)
    if not samples:
        print("No samples found.")
        return

    print(f"Dataset: {len(samples)} samples total")
    train_samples, val_samples, test_samples = split_samples(
        samples, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"  Train: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}")

    train_ds = QuadrantDataset(train_samples, transform=TRAIN_TRANSFORM)
    val_ds   = QuadrantDataset(val_samples,   transform=EVAL_TRANSFORM)
    test_ds  = QuadrantDataset(test_samples,  transform=EVAL_TRANSFORM)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    print(f"Device: {device}")

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=pin_memory)

    train_loader = make_loader(train_ds, True)
    val_loader   = make_loader(val_ds,   False)
    test_loader  = make_loader(test_ds,  False)

    summary_rows = []
    history_rows = []

    for name, model in build_models():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        model.to(device)
        best_acc, history = train_model(
            model, train_loader, val_loader, device,
            args.epochs, args.lr, args.patience
        )
        test_acc, _ = evaluate(model, test_loader, device)
        print(f"  → best_val_acc={best_acc:.4f}  test_acc={test_acc:.4f}")

        for row in history:
            row["model"] = name
            history_rows.append(row)

        params  = count_parameters(model)
        size_kb = params * 4 / 1024           # float32 en KB
        size_mb = size_kb / 1024

        model_cpu = model.to("cpu")
        macs, flops = compute_ops(model_cpu)

        onnx_path = output_dir / "onnx" / f"{name}.onnx"
        export_onnx(model_cpu, onnx_path)

        ckpt_path = output_dir / "checkpoints" / f"{name}.pt"
        torch.save(model_cpu.state_dict(), ckpt_path)

        # Tamaño real del archivo .pt en KB
        pt_size_kb = ckpt_path.stat().st_size / 1024

        summary_rows.append({
            "model":       name,
            "val_acc":     best_acc,
            "test_acc":    test_acc,
            "params":      params,
            "size_kb":     pt_size_kb,
            "size_mb":     size_mb,
            "macs":        macs,
            "flops":       flops,
            "onnx":        onnx_path.as_posix(),
        })

        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_csv(output_dir / "summary.csv", summary_rows)
    save_csv(output_dir / "history.csv", history_rows)
    save_report_md(output_dir / "report.md", summary_rows)

    print(f"\n{'='*50}")
    print("RESUMEN FINAL:")
    print(f"{'Model':<15} {'val_acc':>8} {'test_acc':>9} {'params':>8} {'size_kb':>8}")
    for r in summary_rows:
        flag = " ✓ <200KB" if r["size_kb"] < 200 else ""
        print(f"{r['model']:<15} {r['val_acc']:>8.4f} {r['test_acc']:>9.4f} "
              f"{r['params']:>8} {r['size_kb']:>7.1f}KB{flag}")

    print(f"\nOutputs en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()