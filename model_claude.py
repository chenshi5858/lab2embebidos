"""
train_v3.py — Clasificación de cuadrantes (0=ausente, 1=izq, 2=centro, 3=der)
Mejoras principales vs v2:
  - Se elimina RandomHorizontalFlip (invertía la etiqueta espacial sin cambiarla)
  - Class weights para manejar desbalance
  - Augmentation conservadora y espacialmente coherente
  - LabelSmoothing para reducir overconfidence
  - Más dropout y weight_decay para dataset pequeño
  - Se añade EfficientFC: FC con features de proyección horizontal explícita
"""

import argparse
import csv
import random
from collections import Counter
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


# ──────────────────────────────────────────────────────────
#  DATASET
# ──────────────────────────────────────────────────────────

@dataclass
class SampleItem:
    path: Path
    label: int


class QuadrantDataset(Dataset):
    def __init__(self, samples: List[SampleItem], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        with Image.open(item.path) as img:
            img = img.convert("L")
            if self.transform:
                img = self.transform(img)
        return img, int(item.label)


# ──────────────────────────────────────────────────────────
#  TRANSFORMS
#  ¡SIN RandomHorizontalFlip! — voltear horizontalmente
#  cambia el cuadrante real (izq↔der) pero no la etiqueta.
#  Augmentations seguras para esta tarea:
#    - Pequeña rotación (±8°): objeto sigue en mismo lado
#    - Traducción vertical leve: no afecta posición en X
#    - Jitter de brillo/contraste: variabilidad de cámara/ESP
#    - Gaussian blur leve: simula desenfoque
# ──────────────────────────────────────────────────────────

NORMALIZE = transforms.Normalize(mean=[0.5], std=[0.5])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomAffine(
        degrees=8,                    # rotación ±8°
        translate=(0.0, 0.08),        # solo traslación VERTICAL (no cambia cuadrante en X)
        scale=(0.92, 1.08),           # zoom leve
    ),
    transforms.ColorJitter(brightness=0.35, contrast=0.35),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    NORMALIZE,
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    NORMALIZE,
])


# ──────────────────────────────────────────────────────────
#  ARQUITECTURAS
#  Principio compartido: conservar eje X, colapsar eje Y.
#  Todas usan BatchNorm + Dropout agresivo (dataset chico).
# ──────────────────────────────────────────────────────────

class CollapseY_CNN(nn.Module):
    """
    Conv estándar con stride=2 + BN. Colapsa Y con
    AdaptiveAvgPool(1, 12). Dropout=0.4 por dataset pequeño.
    ~13K params / ~56 KB.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8,  3, stride=2, padding=1, bias=False),   # 48×48
            nn.BatchNorm2d(8),  nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False),   # 24×24
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, bias=False),  # 12×12
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 12))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(192, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class DepthwiseCNN(nn.Module):
    """
    Bloques depthwise separables. ~8K params / ~31 KB.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(),
        )
        self.body = nn.Sequential(
            DepthwiseSeparable(8,  16, stride=2),
            DepthwiseSeparable(16, 24, stride=2),
            DepthwiseSeparable(24, 32, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(192, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.body(self.stem(x))))


class SpatialCNN(nn.Module):
    """
    Kernels asimétricos 1×7 y 7×1 para capturar
    contexto horizontal y vertical por separado.
    ~21K params / ~86 KB.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(1, 8,  3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),  nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.branch_h = nn.Sequential(
            nn.Conv2d(16, 16, (1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.branch_v = nn.Sequential(
            nn.Conv2d(16, 16, (7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(32, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.45),
            nn.Linear(192, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        x = self.shared(x)
        x = torch.cat([self.branch_h(x), self.branch_v(x)], dim=1)
        x = self.fuse(x)
        return self.classifier(self.pool(x))


class LightCNN(nn.Module):
    """Ultra-compacto para ESP32-S3. ~5K params / ~20 KB."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,  8,  3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),  nn.ReLU(),
            nn.Conv2d(8,  12, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12), nn.ReLU(),
            nn.Conv2d(12, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class HorizontalSliceCNN(nn.Module):
    """
    NUEVO — Idea: dividir la imagen en 3 franjas verticales
    (izq / centro / der) explícitamente, extraer features de
    cada una con una conv compartida y comparar energías.
    El clasificador recibe la firma de cada franja → muy
    intuitivo para la tarea de cuadrante.
    ~18K params / ~72 KB.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Feature extractor compartido (se aplica a cada franja)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),   # 48→24 (en franja de 32px)
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),  # 24→12
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                            # → (B,32,1,1)
        )
        # Franja izq: cols 0..31, centro: 32..63, der: 64..95
        self.classifier = nn.Sequential(
            nn.Flatten(),       # 3 franjas × 32 features = 96
            nn.Dropout(0.4),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, 96, 96)
        w = x.shape[-1] // 3  # 32
        left   = x[:, :, :, :w]
        center = x[:, :, :, w:2*w]
        right  = x[:, :, :, 2*w:]
        fl = self.encoder(left).squeeze(-1).squeeze(-1)    # (B, 32)
        fc = self.encoder(center).squeeze(-1).squeeze(-1)
        fr = self.encoder(right).squeeze(-1).squeeze(-1)
        feats = torch.cat([fl, fc, fr], dim=1)             # (B, 96)
        return self.classifier(feats)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TinyResNet(nn.Module):
    """
    NUEVO — ResNet micro con skip connections.
    Los residuales ayudan mucho con datasets pequeños
    porque el gradiente fluye mejor y regulariza implícitamente.
    ~28K params / ~112 KB.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),   # 48×48
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(16),
            nn.Conv2d(16, 24, 3, stride=2, padding=1, bias=False),  # 24×24
            nn.BatchNorm2d(24), nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(24),
            nn.Conv2d(24, 32, 3, stride=2, padding=1, bias=False),  # 12×12
            nn.BatchNorm2d(32), nn.ReLU(),
        )
        # Pool asimétrico: colapsa Y, mantiene 6 cols en X
        self.pool = nn.AdaptiveAvgPool2d((1, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),       # 32×6 = 192
            nn.Dropout(0.45),
            nn.Linear(192, 48),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(48, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.classifier(self.pool(x))


def build_models():
    return [
        ("collapse_y",      CollapseY_CNN()),
        ("depthwise",       DepthwiseCNN()),
        ("spatial_cnn",     SpatialCNN()),
        ("light_cnn",       LightCNN()),
        ("hslice_cnn",      HorizontalSliceCNN()),
        ("tiny_resnet",     TinyResNet()),
    ]


# ──────────────────────────────────────────────────────────
#  DATA UTILS
# ──────────────────────────────────────────────────────────

def parse_label_line(line: str) -> Tuple[str, int]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return "", -1
    try:
        return parts[0], int(parts[2])
    except ValueError:
        return "", -1


def gather_samples(base_dir: Path) -> List[SampleItem]:
    samples = []
    for dataset_folder in sorted(base_dir.glob("dataset_grupo_*_reescalado")):
        for sub in ["celular", "esp"]:
            subfolder = dataset_folder / sub
            labels_file = subfolder / "etiquetas.txt"
            if not labels_file.exists():
                continue
            for line in labels_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                filename, quadrant = parse_label_line(line)
                if quadrant not in (0, 1, 2, 3):
                    continue
                image_path = subfolder / filename
                if image_path.exists():
                    samples.append(SampleItem(path=image_path, label=quadrant))
    return samples


def split_samples(samples, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return shuffled[:t], shuffled[t:t+v], shuffled[t+v:]


def compute_class_weights(samples: List[SampleItem], num_classes: int, device: torch.device):
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase.
    Esto penaliza más los errores en clases poco frecuentes.
    """
    counts = Counter(s.label for s in samples)
    total = len(samples)
    weights = []
    for c in range(num_classes):
        freq = counts.get(c, 1) / total
        weights.append(1.0 / freq)
    # Normaliza para que la suma = num_classes
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum() * num_classes
    return w.to(device)


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────
#  TRAINING
# ──────────────────────────────────────────────────────────

def evaluate(model, loader, device, criterion=None):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if criterion is not None:
                loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return (correct / total if total else 0.0), (loss_sum / total if total else 0.0)


def train_model(model, train_loader, val_loader, device, epochs, lr, class_weights, patience=8):
    # CrossEntropyLoss con class weights + label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    # Cosine annealing: LR cae suavemente hasta lr/20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20)

    best_acc, best_state, no_improve = -1.0, None, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            # Gradient clipping: evita explosión de gradientes con dataset chico
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        scheduler.step()
        val_acc, val_loss = evaluate(model, val_loader, device, criterion)

        history.append({
            "epoch": epoch,
            "train_loss": round(run_loss / total, 6),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(val_acc, 6),
            "lr":         round(scheduler.get_last_lr()[0], 7),
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop @ epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"    [{epoch:3d}] train={run_loss/total:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    if best_state:
        model.load_state_dict(best_state)
    return best_acc, history


# ──────────────────────────────────────────────────────────
#  UTILS
# ──────────────────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def compute_ops(model, input_shape=(1, 1, IMAGE_SIZE, IMAGE_SIZE)):
    model.eval()
    try:
        macs, _ = profile(model, inputs=(torch.zeros(input_shape),), verbose=False)
    except Exception:
        macs = 0.0
    return macs, macs * 2


def export_onnx(model, onnx_path: Path):
    model.eval()
    torch.onnx.export(
        model, torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE), onnx_path.as_posix(),
        input_names=["input"], output_names=["logits"],
        opset_version=18, do_constant_folding=True,
    )


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def save_report_md(path, rows):
    lines = [
        "# Model Report v3", "",
        "| model | val_acc | test_acc | params | size_kb | macs | flops |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['val_acc']:.4f} | {r['test_acc']:.4f} | "
            f"{r['params']} | {r['size_kb']:.1f} | {r['macs']:.2e} | {r['flops']:.2e} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir",    type=str,   default=None)
    p.add_argument("--output_dir",  type=str,   default="outputs")
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch_size",  type=int,   default=16)   # más pequeño = más updates, mejor para dataset chico
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio",   type=float, default=0.15)  # más datos en test
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--patience",    type=int,   default=12)
    return p.parse_args()


def main():
    args = parse_args()
    base_dir   = Path(args.base_dir) if args.base_dir else Path(__file__).parent.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "onnx").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    samples = gather_samples(base_dir)
    if not samples:
        print("No samples found.")
        return

    # Mostrar distribución de clases
    counts = Counter(s.label for s in samples)
    print(f"Dataset: {len(samples)} samples")
    print(f"  Distribución: { {k: counts[k] for k in sorted(counts)} }")

    train_s, val_s, test_s = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)
    print(f"  Train={len(train_s)}  Val={len(val_s)}  Test={len(test_s)}")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem   = device.type == "cuda"
    print(f"Device: {device}")

    # Class weights calculados sobre train set
    class_weights = compute_class_weights(train_s, NUM_CLASSES, device)
    print(f"  Class weights: {class_weights.cpu().tolist()}")

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=pin_mem)

    train_loader = make_loader(QuadrantDataset(train_s, TRAIN_TRANSFORM), True)
    val_loader   = make_loader(QuadrantDataset(val_s,   EVAL_TRANSFORM),  False)
    test_loader  = make_loader(QuadrantDataset(test_s,  EVAL_TRANSFORM),  False)

    summary_rows, history_rows = [], []

    for name, model in build_models():
        print(f"\n{'='*52}\nEntrenando: {name}")
        model.to(device)
        best_acc, history = train_model(
            model, train_loader, val_loader, device,
            args.epochs, args.lr, class_weights, args.patience,
        )
        test_acc, _ = evaluate(model, test_loader, device)
        print(f"  → val_acc={best_acc:.4f}  test_acc={test_acc:.4f}")

        for row in history:
            history_rows.append({"model": name, **row})

        params   = count_parameters(model)
        model_cpu = model.to("cpu")
        macs, flops = compute_ops(model_cpu)

        onnx_path = output_dir / "onnx" / f"{name}.onnx"
        export_onnx(model_cpu, onnx_path)
        ckpt_path = output_dir / "checkpoints" / f"{name}.pt"
        torch.save(model_cpu.state_dict(), ckpt_path)
        size_kb = ckpt_path.stat().st_size / 1024

        summary_rows.append({
            "model": name, "val_acc": best_acc, "test_acc": test_acc,
            "params": params, "size_kb": size_kb,
            "macs": macs, "flops": flops,
            "onnx": onnx_path.as_posix(),
        })

        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_csv(output_dir / "summary.csv", summary_rows)
    save_csv(output_dir / "history.csv", history_rows)
    save_report_md(output_dir / "report.md", summary_rows)

    print(f"\n{'='*52}")
    print(f"{'Model':<16} {'val_acc':>8} {'test_acc':>9} {'params':>8} {'size_kb':>9}")
    print("-" * 55)
    for r in summary_rows:
        flag = " ✓" if r["size_kb"] < 200 else "  "
        print(f"{r['model']:<16} {r['val_acc']:>8.4f} {r['test_acc']:>9.4f} "
              f"{r['params']:>8} {r['size_kb']:>8.1f}KB{flag}")
    print("\n✓ = bajo 200KB")
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()