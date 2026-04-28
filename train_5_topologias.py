import argparse
import copy
import csv
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class SampleRecord:
    image_name: str
    target: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def labels_to_multihot(value_1: int, value_2: int, mode: str) -> List[int]:
    target = [0, 0, 0]

    if mode == "pair_quadrants":
        if value_1 == 0 and value_2 == 0:
            return target
        for q in (value_1, value_2):
            if q in (1, 2, 3):
                target[q - 1] = 1
        return target

    if mode == "flag_quadrant":
        if value_1 == 0:
            return target
        if value_2 in (1, 2, 3):
            target[value_2 - 1] = 1
        return target

    raise ValueError(f"Modo de etiqueta no soportado: {mode}")


class QuadrantDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_file: Path,
        label_mode: str,
        image_size: int = 96,
    ) -> None:
        self.images_dir = images_dir
        self.labels_file = labels_file
        self.label_mode = label_mode
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        self.records: List[SampleRecord] = self._read_labels()
        self.targets: List[torch.Tensor] = [r.target for r in self.records]

    def _read_labels(self) -> List[SampleRecord]:
        if not self.labels_file.exists():
            raise FileNotFoundError(f"No existe archivo de etiquetas: {self.labels_file}")

        records: List[SampleRecord] = []
        with self.labels_file.open("r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    raise ValueError(
                        f"Linea {line_num} invalida (se esperan 3 columnas): {raw_line.strip()}"
                    )

                image_name = parts[0]
                try:
                    value_1 = int(parts[1])
                    value_2 = int(parts[2])
                except ValueError:
                    # Permite una posible cabecera y continua con el resto.
                    if line_num == 1:
                        continue
                    raise ValueError(
                        f"Linea {line_num} invalida, etiquetas no numericas: {raw_line.strip()}"
                    )

                multihot = labels_to_multihot(value_1, value_2, self.label_mode)
                records.append(
                    SampleRecord(
                        image_name=image_name,
                        target=torch.tensor(multihot, dtype=torch.float32),
                    )
                )

        if not records:
            raise ValueError("No se encontraron registros validos en el archivo de etiquetas.")

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image_path = self.images_dir / record.image_name
        if not image_path.exists():
            raise FileNotFoundError(f"No se encontro imagen: {image_path}")

        image = Image.open(image_path).convert("L")
        tensor_image = self.transform(image)
        return tensor_image, record.target


class FCTiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 96, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FCSmall(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 96, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNTiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


class CNNStride(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


class CNNSmall(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)


def get_model_builders() -> Dict[str, Callable[[], nn.Module]]:
    return {
        "fc_tiny": FCTiny,
        "fc_small": FCSmall,
        "cnn_tiny": CNNTiny,
        "cnn_stride": CNNStride,
        "cnn_small": CNNSmall,
    }


def exact_match_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= threshold).float()
    matches = (preds == targets).all(dim=1).float()
    return matches.mean().item()


def labelwise_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= threshold).float()
    return (preds == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    threshold: float,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_exact = 0.0
    total_labelwise = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_exact += exact_match_accuracy(logits.detach(), targets, threshold) * batch_size
        total_labelwise += labelwise_accuracy(logits.detach(), targets, threshold) * batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_exact = total_exact / max(total_samples, 1)
    avg_labelwise = total_labelwise / max(total_samples, 1)
    return avg_loss, avg_exact, avg_labelwise


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_exact = 0.0
    total_labelwise = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_exact += exact_match_accuracy(logits, targets, threshold) * batch_size
        total_labelwise += labelwise_accuracy(logits, targets, threshold) * batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_exact = total_exact / max(total_samples, 1)
    avg_labelwise = total_labelwise / max(total_samples, 1)
    return avg_loss, avg_exact, avg_labelwise


def compute_pos_weight(train_subset: Subset) -> torch.Tensor:
    indices = train_subset.indices
    dataset = train_subset.dataset
    labels = torch.stack([dataset.targets[idx] for idx in indices])
    positives = labels.sum(dim=0)
    negatives = labels.size(0) - positives
    weights = negatives / positives.clamp(min=1.0)

    # Si una clase no tiene positivos o no tiene negativos, usa peso neutro 1.0.
    # Esto evita pesos 0 o extremadamente grandes por divisiones degeneradas.
    degenerate = (positives == 0) | (negatives == 0)
    weights[degenerate] = 1.0
    return weights


def subset_label_stats(subset: Subset) -> Tuple[torch.Tensor, int]:
    indices = subset.indices
    dataset = subset.dataset
    labels = torch.stack([dataset.targets[idx] for idx in indices])
    positives = labels.sum(dim=0)
    return positives, labels.size(0)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def estimate_macs(model: nn.Module, device: torch.device) -> int:
    model.eval()
    macs = [0]
    hooks = []

    def conv_hook(module: nn.Conv2d, _, output: torch.Tensor) -> None:
        out = output
        batch_size, out_channels, out_h, out_w = out.shape
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups
        layer_macs = (
            batch_size
            * out_channels
            * out_h
            * out_w
            * (in_channels // groups)
            * kernel_h
            * kernel_w
        )
        macs[0] += int(layer_macs)

    def linear_hook(module: nn.Linear, _, output: torch.Tensor) -> None:
        out = output
        batch_size = out.shape[0] if out.dim() > 1 else 1
        layer_macs = batch_size * module.in_features * module.out_features
        macs[0] += int(layer_macs)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    dummy = torch.randn(1, 1, 96, 96, device=device)
    _ = model(dummy)

    for hook in hooks:
        hook.remove()

    return macs[0]


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024.0 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def export_to_onnx(model: nn.Module, onnx_path: Path, device: torch.device) -> None:
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, 1, 96, 96, device=device)

    # Prefiere el exportador clasico para evitar conversiones de opset no estables.
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=18,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        # Compatibilidad con versiones viejas de torch que no aceptan dynamo.
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=18,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
        )


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    threshold: float,
) -> Tuple[nn.Module, List[dict], int, float, float]:
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_label_acc = 0.0
    best_epoch = 0
    wait = 0
    history: List[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_label_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, threshold
        )
        val_loss, val_acc, val_label_acc = evaluate(model, val_loader, criterion, device, threshold)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_exact_acc": train_acc,
                "train_label_acc": train_label_acc,
                "val_loss": val_loss,
                "val_exact_acc": val_acc,
                "val_label_acc": val_label_acc,
            }
        )

        print(
            f"[{model_name}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_exact={train_acc:.4f} train_label={train_label_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_exact={val_acc:.4f} val_label={val_label_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_label_acc = val_label_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[{model_name}] Early stopping en epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val_acc, best_val_label_acc


def generate_report(
    results: Sequence[dict],
    report_path: Path,
    dataset_size: int,
    split_sizes: Tuple[int, int],
    label_mode: str,
) -> None:
    train_size, val_size = split_sizes

    with PdfPages(report_path) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        title = "Reporte: 5 Topologias para Cuadrantes (96x96 Monocromatico)"
        details = (
            f"Total de imagenes: {dataset_size}\n"
            f"Split train/val: {train_size}/{val_size}\n"
            f"Modo de etiquetas: {label_mode}\n"
            f"Metricas: Exact Match Accuracy + Label-wise Accuracy (3 etiquetas binarias)\n"
            f"Operaciones reportadas: MACs y FLOPs aproximados por inferencia"
        )
        ax.text(0.03, 0.95, title, fontsize=17, weight="bold", va="top")
        ax.text(0.03, 0.82, details, fontsize=12, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for row in results:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis("off")

            header = f"Topologia: {row['model_name']}"
            metrics = (
                f"Best Epoch: {row['best_epoch']}\n"
                f"Val Exact Accuracy: {row['val_exact_acc']:.4f}\n"
                f"Val Label-wise Accuracy: {row['val_label_acc']:.4f}\n"
                f"Parametros entrenables: {row['params']:,}\n"
                f"Checkpoint (.pt): {row['pt_size_human']}\n"
                f"ONNX: {row['onnx_size_human']}\n"
                f"MACs por inferencia: {row['macs']:,}\n"
                f"FLOPs aprox. por inferencia: {row['flops']:,}\n"
                f"Archivo ONNX para Netron: {row['onnx_path']}"
            )
            architecture = textwrap.fill(row["model_repr"], width=110)

            ax.text(0.03, 0.96, header, fontsize=16, weight="bold", va="top")
            ax.text(0.03, 0.82, metrics, fontsize=11, va="top")
            ax.text(0.03, 0.36, "Arquitectura (PyTorch):", fontsize=12, weight="bold", va="top")
            ax.text(0.03, 0.32, architecture, fontsize=8, va="top", family="monospace")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.text(0.03, 0.95, "Resumen Comparativo", fontsize=16, weight="bold", va="top")

        headers = [
            "Modelo",
            "Val Exact",
            "Val Label",
            "Params",
            "PT Size",
            "ONNX Size",
            "MACs",
            "FLOPs",
        ]
        table_data = [
            [
                r["model_name"],
                f"{r['val_exact_acc']:.4f}",
                f"{r['val_label_acc']:.4f}",
                f"{r['params']:,}",
                r["pt_size_human"],
                r["onnx_size_human"],
                f"{r['macs']:,}",
                f"{r['flops']:,}",
            ]
            for r in results
        ]

        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.7)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def save_histories(results: Sequence[dict], history_path: Path) -> None:
    rows = []
    for result in results:
        for item in result["history"]:
            rows.append(
                {
                    "model_name": result["model_name"],
                    **item,
                }
            )

    if not rows:
        return

    fieldnames = [
        "model_name",
        "epoch",
        "train_loss",
        "train_exact_acc",
        "train_label_acc",
        "val_loss",
        "val_exact_acc",
        "val_label_acc",
    ]

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(results: Sequence[dict], summary_path: Path) -> None:
    fieldnames = [
        "model_name",
        "best_epoch",
        "val_exact_acc",
        "val_label_acc",
        "params",
        "macs",
        "flops",
        "pt_size_bytes",
        "onnx_size_bytes",
        "checkpoint_path",
        "onnx_path",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "model_name": row["model_name"],
                    "best_epoch": row["best_epoch"],
                    "val_exact_acc": row["val_exact_acc"],
                    "val_label_acc": row["val_label_acc"],
                    "params": row["params"],
                    "macs": row["macs"],
                    "flops": row["flops"],
                    "pt_size_bytes": row["pt_size_bytes"],
                    "onnx_size_bytes": row["onnx_size_bytes"],
                    "checkpoint_path": row["checkpoint_path"],
                    "onnx_path": row["onnx_path"],
                }
            )


def resolve_device(user_device: str) -> torch.device:
    if user_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if user_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidio cuda, pero no esta disponible.")
    return torch.device(user_device)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena 5 topologias (FC/CNN) para detectar cuadrantes activos en imagenes 96x96." 
    )
    parser.add_argument("--images_dir", type=Path, required=True, help="Carpeta de imagenes.")
    parser.add_argument("--labels_file", type=Path, required=True, help="Archivo de etiquetas.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Salida de resultados.")
    parser.add_argument("--label_mode", choices=["pair_quadrants", "flag_quadrant"], default="pair_quadrants")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral para metricas multietiqueta.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.train_ratio <= 0.0 or args.train_ratio >= 1.0:
        raise ValueError("train_ratio debe estar entre 0 y 1")
    if not (0.0 < args.threshold < 1.0):
        raise ValueError("threshold debe estar entre 0 y 1")

    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = args.output_dir
    checkpoints_dir = output_dir / "checkpoints"
    onnx_dir = output_dir / "onnx"
    report_dir = output_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset = QuadrantDataset(
        images_dir=args.images_dir,
        labels_file=args.labels_file,
        label_mode=args.label_mode,
        image_size=96,
    )

    total_size = len(dataset)
    train_size = int(total_size * args.train_ratio)
    val_size = total_size - train_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("El split train/val quedo invalido. Ajusta train_ratio o agrega datos.")

    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    pos_weight = compute_pos_weight(train_subset).to(device)
    train_pos, train_n = subset_label_stats(train_subset)
    val_pos, val_n = subset_label_stats(val_subset)

    print("=" * 80)
    print("Configuracion")
    print(f"Device: {device}")
    print(f"Muestras: total={total_size} train={train_size} val={val_size}")
    print(
        f"Positivos por cuadrante (train): {train_pos.tolist()} de {train_n} "
        f"=> {[round((x / max(train_n, 1)) * 100.0, 2) for x in train_pos.tolist()]}%"
    )
    print(
        f"Positivos por cuadrante (val): {val_pos.tolist()} de {val_n} "
        f"=> {[round((x / max(val_n, 1)) * 100.0, 2) for x in val_pos.tolist()]}%"
    )
    print(f"Pos weight BCE: {pos_weight.tolist()}")
    print("=" * 80)

    results: List[dict] = []
    model_builders = get_model_builders()

    for model_name, builder in model_builders.items():
        print("\n" + "-" * 80)
        print(f"Entrenando topologia: {model_name}")
        print("-" * 80)

        model = builder().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        model, history, best_epoch, best_val_acc, best_val_label_acc = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            threshold=args.threshold,
        )

        params = count_trainable_parameters(model)
        macs = estimate_macs(model, device=device)
        flops = 2 * macs

        checkpoint_path = checkpoints_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        pt_size_bytes = checkpoint_path.stat().st_size

        onnx_path = onnx_dir / f"{model_name}.onnx"
        export_to_onnx(model, onnx_path, device=device)
        onnx_size_bytes = onnx_path.stat().st_size

        result = {
            "model_name": model_name,
            "best_epoch": best_epoch,
            "val_exact_acc": best_val_acc,
            "val_label_acc": best_val_label_acc,
            "params": params,
            "macs": macs,
            "flops": flops,
            "pt_size_bytes": pt_size_bytes,
            "onnx_size_bytes": onnx_size_bytes,
            "pt_size_human": human_bytes(pt_size_bytes),
            "onnx_size_human": human_bytes(onnx_size_bytes),
            "checkpoint_path": str(checkpoint_path),
            "onnx_path": str(onnx_path),
            "model_repr": str(model),
            "history": history,
        }
        results.append(result)

        print(
            f"[{model_name}] best_val_exact={best_val_acc:.4f} best_val_label={best_val_label_acc:.4f} "
            f"params={params:,} macs={macs:,}"
        )

    summary_csv = output_dir / "summary.csv"
    history_csv = output_dir / "history.csv"
    report_pdf = report_dir / "output.pdf"

    save_summary(results, summary_csv)
    save_histories(results, history_csv)
    generate_report(
        results=results,
        report_path=report_pdf,
        dataset_size=total_size,
        split_sizes=(train_size, val_size),
        label_mode=args.label_mode,
    )

    print("\nProceso finalizado.")
    print(f"Resumen CSV: {summary_csv}")
    print(f"Historico entrenamiento CSV: {history_csv}")
    print(f"Reporte PDF: {report_pdf}")
    print(f"Modelos ONNX (Netron): {onnx_dir}")


if __name__ == "__main__":
    main()
