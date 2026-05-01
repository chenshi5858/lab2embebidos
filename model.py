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


class FCTiny(nn.Module):
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


class FCSmall(nn.Module):
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


class CNNTiny(nn.Module):
	def __init__(self, num_classes: int = NUM_CLASSES):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 4, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(4, 8, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(8 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 64),
			nn.ReLU(),
			nn.Linear(64, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		return self.classifier(x)


class CNNGAP(nn.Module):
	def __init__(self, num_classes: int = NUM_CLASSES):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 4, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(4, 8, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(8, 16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d((1, 1)),
		)
		self.classifier = nn.Linear(16, num_classes)

	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		return self.classifier(x)


class SkipPoolCNN(nn.Module):
	def __init__(self, num_classes: int = NUM_CLASSES):
		super().__init__()
		self.skip_pool = nn.MaxPool2d(kernel_size=8, stride=8)
		self.conv = nn.Sequential(
			nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
			nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
			nn.Conv2d(8, 12, kernel_size=3, padding=1, stride=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
			nn.Conv2d(12, 12, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(12, 16, kernel_size=3, padding=1),
			nn.ReLU(),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear((16 + 1) * 12 * 12, num_classes),
		)

	def forward(self, x):
		skip = self.skip_pool(x)
		y = self.conv(x)
		if skip.shape[-2:] != y.shape[-2:]:
			skip = torch.nn.functional.adaptive_max_pool2d(skip, y.shape[-2:])
		z = torch.cat([y, skip], dim=1)
		return self.classifier(z)


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
	train_samples = shuffled[:train_end]
	val_samples = shuffled[train_end:val_end]
	test_samples = shuffled[val_end:]
	return train_samples, val_samples, test_samples


def seed_everything(seed: int):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def build_models():
	return [
		("fc_tiny", FCTiny()),
		("fc_small", FCSmall()),
		("cnn_tiny", CNNTiny()),
		("cnn_gap", CNNGAP()),
		("cnn_skip_pool", SkipPoolCNN()),
	]


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
				loss = criterion(logits, labels)
				loss_sum += loss.item() * labels.size(0)
			preds = torch.argmax(logits, dim=1)
			correct += (preds == labels).sum().item()
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
):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)
	best_acc = -1.0
	best_state = None
	history = []
	for epoch in range(1, epochs + 1):
		model.train()
		running_loss = 0.0
		total = 0
		for images, labels in train_loader:
			images = images.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			logits = model(images)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * labels.size(0)
			total += labels.size(0)

		train_loss = running_loss / total if total else 0.0
		val_acc, val_loss = evaluate(model, val_loader, device, criterion)
		history.append(
			{
				"epoch": epoch,
				"train_loss": train_loss,
				"val_loss": val_loss,
				"val_acc": val_acc,
			}
		)
		if val_acc > best_acc:
			best_acc = val_acc
			best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

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
	flops = macs * 2
	return macs, flops


def export_onnx(model: nn.Module, onnx_path: Path):
	model.eval()
	dummy = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE)
	torch.onnx.export(
		model,
		dummy,
		onnx_path.as_posix(),
		input_names=["input"],
		output_names=["logits"],
		opset_version=18,
		do_constant_folding=True,
	)


def save_summary_csv(path: Path, rows: List[dict]):
	if not rows:
		return
	fieldnames = list(rows[0].keys())
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def save_history_csv(path: Path, rows: List[dict]):
	if not rows:
		return
	fieldnames = list(rows[0].keys())
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def save_report_md(path: Path, rows: List[dict]):
	lines = [
		"# Model Report",
		"",
		"| model | val_acc | test_acc | params | size_mb | macs | flops | onnx |",
		"| --- | --- | --- | --- | --- | --- | --- | --- |",
	]
	for row in rows:
		lines.append(
			"| {model} | {val_acc:.4f} | {test_acc:.4f} | {params} | {size_mb:.4f} | {macs:.2e} | {flops:.2e} | {onnx} |".format(
				**row
			)
		)
	path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
	parser = argparse.ArgumentParser(description="Train 5 small FC/CNN models for quadrant classification.")
	parser.add_argument("--base_dir", type=str, default=None, help="Root directory with dataset_grupo_*_reescalado")
	parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--train_ratio", type=float, default=0.7)
	parser.add_argument("--val_ratio", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--num_workers", type=int, default=0)
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
		print("No samples found. Check dataset_grupo_*_reescalado folders and etiquetas.txt files.")
		return

	train_samples, val_samples, test_samples = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)
	transform = transforms.Compose([
		transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
		transforms.ToTensor(),
	])

	train_ds = QuadrantDataset(train_samples, transform=transform)
	val_ds = QuadrantDataset(val_samples, transform=transform)
	test_ds = QuadrantDataset(test_samples, transform=transform)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	pin_memory = device.type == "cuda"

	train_loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=pin_memory,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=pin_memory,
	)
	test_loader = DataLoader(
		test_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=pin_memory,
	)

	summary_rows = []
	history_rows = []

	for name, model in build_models():
		print(f"Training {name}")
		model.to(device)
		best_acc, history = train_model(model, train_loader, val_loader, device, args.epochs, args.lr)
		test_acc, _ = evaluate(model, test_loader, device)

		for row in history:
			row["model"] = name
			history_rows.append(row)

		params = count_parameters(model)
		size_mb = params * 4 / (1024**2)

		model_cpu = model.to("cpu")
		macs, flops = compute_ops(model_cpu)

		onnx_path = output_dir / "onnx" / f"{name}.onnx"
		export_onnx(model_cpu, onnx_path)

		ckpt_path = output_dir / "checkpoints" / f"{name}.pt"
		torch.save(model_cpu.state_dict(), ckpt_path)

		summary_rows.append(
			{
				"model": name,
				"val_acc": best_acc,
				"test_acc": test_acc,
				"params": params,
				"size_mb": size_mb,
				"macs": macs,
				"flops": flops,
				"onnx": onnx_path.as_posix(),
			}
		)

		if device.type == "cuda":
			torch.cuda.empty_cache()

	save_summary_csv(output_dir / "summary.csv", summary_rows)
	save_history_csv(output_dir / "history.csv", history_rows)
	save_report_md(output_dir / "report.md", summary_rows)
	print("Done. Check outputs in", output_dir.resolve())


if __name__ == "__main__":
	main()
