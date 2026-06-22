"""Train and export SkipPoolCNN with TensorFlow/Keras.

This is the TensorFlow version of the PyTorch SkipPoolCNN from model.py.
Running this file trains the float model, fine-tunes it with quantization-aware
training, evaluates it and exports Keras/TFLite artifacts.

Default usage:
    python skippoolcnn_tf.py

Inference with a saved model:
    python skippoolcnn_tf.py --predict path/to/image.jpg --model_path outputs_tf/cnn_skip_pool_tf.keras
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple


def patch_random_for_tf_keras_python312():
    """tf-keras 2.16 calls random.randint with 1e9, which Python 3.12 rejects."""
    if sys.version_info < (3, 12):
        return
    if getattr(random.Random.randrange, "_tf_keras_py312_patch", False):
        return

    original_randrange = random.Random.randrange

    def compatible_randrange(self, start, stop=None, step=1):
        if isinstance(start, float) and start.is_integer():
            start = int(start)
        if isinstance(stop, float) and stop.is_integer():
            stop = int(stop)
        if isinstance(step, float) and step.is_integer():
            step = int(step)
        return original_randrange(self, start, stop, step)

    compatible_randrange._tf_keras_py312_patch = True
    random.Random.randrange = compatible_randrange
    random.randrange = random._inst.randrange
    random.randint = random._inst.randint


# TFMOT's quantization-aware training path is still safest with tf-keras.
# TensorFlow 2.16+ defaults to Keras 3 unless this is set before importing TF.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
patch_random_for_tf_keras_python312()

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers
except (ImportError, ModuleNotFoundError) as exc:
    raise SystemExit(
        "Faltan dependencias de entrenamiento. Instala con: pip install -r requirements.txt"
    ) from exc

try:
    import tensorflow_model_optimization as tfmot
except (ImportError, ModuleNotFoundError):
    tfmot = None

IMAGE_SIZE = 96
NUM_CLASSES = 4
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
MODEL_NAME = "cnn_skip_pool_tf"
CLASS_NAMES = ["Ausente", "Izquierda", "Centro", "Derecha"]


@dataclass
class SampleItem:
    path: Path
    label: int


def _pytorch_same_conv2d(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Conv2D with SAME output shapes and TFLite Micro-friendly padding."""
    prefix = name or "conv"
    return layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        activation=None,
        use_bias=True,
        name=prefix,
    )(x)


def build_skippoolcnn(
    num_classes: int = NUM_CLASSES,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    name: str = "cnn_skip_pool",
) -> tf.keras.Model:
    """Build the TensorFlow equivalent of model.py::SkipPoolCNN."""
    inputs = layers.Input(shape=input_shape, name="input")

    skip = layers.MaxPooling2D(
        pool_size=8,
        strides=8,
        padding="valid",
        name="skip_pool",
    )(inputs)

    x = _pytorch_same_conv2d(inputs, filters=4, strides=2, name="conv_0")
    x = layers.ReLU(name="relu_0")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=1, padding="same", name="pool_0")(x)

    x = _pytorch_same_conv2d(x, filters=8, strides=2, name="conv_1")
    x = layers.ReLU(name="relu_1")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=1, padding="same", name="pool_1")(x)

    x = _pytorch_same_conv2d(x, filters=12, strides=2, name="conv_2")
    x = layers.ReLU(name="relu_2")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=1, padding="same", name="pool_2")(x)

    x = _pytorch_same_conv2d(x, filters=12, strides=1, name="conv_3")
    x = layers.ReLU(name="relu_3")(x)

    x = _pytorch_same_conv2d(x, filters=16, strides=1, name="conv_4")
    x = layers.ReLU(name="relu_4")(x)

    x = layers.Concatenate(axis=-1, name="concat_skip")([x, skip])
    x = layers.Flatten(name="flatten")(x)
    outputs = layers.Dense(num_classes, name="classifier")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


class SkipPoolCNN(tf.keras.Model):
    """Subclass wrapper around the functional SkipPoolCNN graph."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_shape: tuple[int, int, int] = INPUT_SHAPE,
        name: str = "cnn_skip_pool",
    ):
        super().__init__(name=name)
        self.model = build_skippoolcnn(num_classes=num_classes, input_shape=input_shape, name=name)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.model(inputs, training=training)


def compile_for_training(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Compile for labels encoded as integers 0..3."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


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


def split_samples(
    samples: List[SampleItem],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[SampleItem], List[SampleItem], List[SampleItem]]:
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
    tf.keras.utils.set_random_seed(seed)


def resolve_default_base_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path.cwd(),
    ]
    for candidate in candidates:
        if list(candidate.glob("dataset_grupo_*_reescalado")):
            return candidate
    return Path(__file__).resolve().parent.parent


def load_image_for_dataset(path: tf.Tensor, label: tf.Tensor):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.int32)


def make_dataset(
    samples: List[SampleItem],
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> Optional[tf.data.Dataset]:
    if not samples:
        return None
    paths = [sample.path.as_posix() for sample in samples]
    labels = [sample.label for sample in samples]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(samples), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load_image_for_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_single_image(path: str | Path) -> tf.Tensor:
    image, _ = load_image_for_dataset(tf.constant(Path(path).as_posix()), tf.constant(0))
    return tf.expand_dims(image, axis=0)


def evaluate_model(model: tf.keras.Model, dataset: Optional[tf.data.Dataset]):
    if dataset is None:
        return 0.0, 0.0
    loss, acc = model.evaluate(dataset, verbose=0)
    return float(acc), float(loss)


def history_to_rows(history: tf.keras.callbacks.History, model_name: str) -> List[dict]:
    rows = []
    history_dict = history.history
    epochs = len(history.epoch)
    for i in range(epochs):
        row = {"model": model_name, "epoch": i + 1}
        for key, values in history_dict.items():
            row[key] = values[i]
        rows.append(row)
    return rows


def save_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_report_md(path: Path, row: dict, plot_paths: Optional[List[Path]] = None):
    lines = [
        "# TensorFlow SkipPoolCNN Report",
        "",
        "| model | int8_method | float_test_acc | qat_test_acc | int8_tflite_acc | params |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        (
            "| {model} | {int8_method} | {test_acc:.4f} | {qat_test_acc:.4f} | {tflite_int8_acc:.4f} | "
            "{params} |"
        ).format(**row),
        "",
        "## Model sizes",
        "",
        "Sizes use `1 MB = 1024^2 bytes`. The FP32 parameter estimate excludes file metadata.",
        "",
        "| artifact | quantization | size_mb |",
        "| --- | --- | ---: |",
        f"| Parameters only (estimated) | Float32 | {row['size_mb']:.4f} |",
        f"| Keras model | Float32, unquantized | {row['keras_float32_size_mb']:.4f} |",
        f"| TFLite model | Float32, unquantized | {row['tflite_float32_size_mb']:.4f} |",
        f"| TFLite model | Dynamic range | {row['tflite_dynamic_size_mb']:.4f} |",
        f"| TFLite model | INT8 | {row['tflite_int8_size_mb']:.4f} |",
    ]
    if plot_paths:
        lines.extend(["", "## Graficos de evaluacion", ""])
        for plot_path in plot_paths:
            relative_path = plot_path.relative_to(path.parent).as_posix()
            title = plot_path.stem.replace("_", " ").title()
            lines.extend([f"### {title}", "", f"![{title}]({relative_path})", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def predict_keras_dataset(
    model: tf.keras.Model,
    dataset: Optional[tf.data.Dataset],
) -> Tuple[List[int], List[int]]:
    """Return true labels and predictions for a Keras model."""
    if dataset is None:
        return [], []
    y_true: List[int] = []
    y_pred: List[int] = []
    for images, labels in dataset:
        logits = model(images, training=False)
        y_true.extend(int(value) for value in labels.numpy())
        y_pred.extend(int(value) for value in tf.argmax(logits, axis=1).numpy())
    return y_true, y_pred


def predict_tflite_dataset(
    tflite_path: str | Path,
    dataset: Optional[tf.data.Dataset],
) -> Tuple[List[int], List[int]]:
    """Return true labels and predictions for a TFLite model."""
    if dataset is None:
        return [], []
    interpreter = tf.lite.Interpreter(model_path=Path(tflite_path).as_posix())
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    y_true: List[int] = []
    y_pred: List[int] = []
    for images, labels in dataset:
        for image, label in zip(images, labels):
            image = tf.expand_dims(image, axis=0)
            interpreter.set_tensor(input_details["index"], quantize_input_for_tflite(image, input_details))
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])[0]
            y_true.append(int(label.numpy()))
            y_pred.append(int(output.argmax()))
    return y_true, y_pred


def plot_training_curves(history_rows: List[dict], output_path: Path):
    """Plot train/validation accuracy and loss for float and QAT phases."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {MODEL_NAME: "#1769aa", f"{MODEL_NAME}_qat": "#d97706"}
    labels = {MODEL_NAME: "Float", f"{MODEL_NAME}_qat": "QAT"}
    model_names = list(dict.fromkeys(row["model"] for row in history_rows))
    for model_name in model_names:
        rows = [row for row in history_rows if row["model"] == model_name]
        epochs = [int(row["epoch"]) for row in rows]
        color = colors.get(model_name, "#555555")
        label = labels.get(model_name, model_name)
        axes[0].plot(epochs, [float(row["accuracy"]) for row in rows], color=color, label=f"{label} train")
        if "val_accuracy" in rows[0]:
            axes[0].plot(
                epochs,
                [float(row["val_accuracy"]) for row in rows],
                color=color,
                linestyle="--",
                label=f"{label} validacion",
            )
        axes[1].plot(epochs, [float(row["loss"]) for row in rows], color=color, label=f"{label} train")
        if "val_loss" in rows[0]:
            axes[1].plot(
                epochs,
                [float(row["val_loss"]) for row in rows],
                color=color,
                linestyle="--",
                label=f"{label} validacion",
            )

    axes[0].set(title="Exactitud por epoca", xlabel="Epoca", ylabel="Accuracy", ylim=(0, 1))
    axes[1].set(title="Perdida por epoca", xlabel="Epoca", ylabel="Loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(summary_row: dict, output_path: Path):
    """Compare float, QAT and deployed INT8 test accuracy."""
    labels = ["Float", "QAT", "TFLite INT8"]
    values = [
        float(summary_row["test_acc"]),
        float(summary_row["qat_test_acc"]),
        float(summary_row["tflite_int8_acc"]),
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    bars = axis.bar(labels, values, color=["#1769aa", "#d97706", "#16836b"])
    axis.set(title="Exactitud en el conjunto de prueba", ylabel="Accuracy", ylim=(0, 1))
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    output_path: Path,
    title: str,
):
    """Plot counts and row-normalized percentages for each class."""
    matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES).numpy()
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)

    fig, axis = plt.subplots(figsize=(7, 5.8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set(
        title=title,
        xlabel="Clase predicha",
        ylabel="Clase real",
        xticks=range(NUM_CLASSES),
        yticks=range(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{matrix[row, column]}\n{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
            )
    fig.colorbar(image, ax=axis, label="Proporcion por clase real")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_evaluation_plots(
    output_dir: Path,
    history_rows: List[dict],
    summary_row: dict,
    test_ds: Optional[tf.data.Dataset],
    float_model: tf.keras.Model,
    qat_model: Optional[tf.keras.Model],
    tflite_int8_path: str,
) -> List[Path]:
    """Generate all plots used to inspect training and test performance."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: List[Path] = []

    training_path = plots_dir / "training_curves.png"
    plot_training_curves(history_rows, training_path)
    plot_paths.append(training_path)

    comparison_path = plots_dir / "accuracy_comparison.png"
    plot_accuracy_comparison(summary_row, comparison_path)
    plot_paths.append(comparison_path)

    if test_ds is not None:
        y_true, y_pred = predict_keras_dataset(float_model, test_ds)
        float_matrix_path = plots_dir / "confusion_matrix_float.png"
        plot_confusion_matrix(y_true, y_pred, float_matrix_path, "Matriz de confusion - Float")
        plot_paths.append(float_matrix_path)

        if qat_model is not None:
            y_true, y_pred = predict_keras_dataset(qat_model, test_ds)
            qat_matrix_path = plots_dir / "confusion_matrix_qat.png"
            plot_confusion_matrix(y_true, y_pred, qat_matrix_path, "Matriz de confusion - QAT")
            plot_paths.append(qat_matrix_path)

        if tflite_int8_path:
            y_true, y_pred = predict_tflite_dataset(tflite_int8_path, test_ds)
            int8_matrix_path = plots_dir / "confusion_matrix_int8.png"
            plot_confusion_matrix(y_true, y_pred, int8_matrix_path, "Matriz de confusion - TFLite INT8")
            plot_paths.append(int8_matrix_path)

    return plot_paths


def convert_to_tflite_float32(model: tf.keras.Model, output_path: str | Path) -> bytes:
    """Export a float32 TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)
    return tflite_model


def convert_to_tflite_dynamic_range(model: tf.keras.Model, output_path: str | Path) -> bytes:
    """Export a dynamic-range quantized TFLite model, no calibration data required."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)
    return tflite_model


def convert_to_tflite_int8(
    model: tf.keras.Model,
    representative_dataset: Callable[[], Iterable[list[tf.Tensor]]],
    output_path: str | Path,
) -> bytes:
    """
    Export a fully int8 quantized TFLite model.

    representative_dataset must yield batches like:
        yield [image_batch.astype("float32")]

    Each image batch must use channels-last shape:
        (batch, 96, 96, 1)
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)
    return tflite_model


def build_qat_model(
    float_model: tf.keras.Model,
    learning_rate: float,
) -> Optional[tf.keras.Model]:
    """Wrap a trained Keras model for quantization-aware fine-tuning."""
    if tfmot is None:
        print(
            "tensorflow-model-optimization no esta instalado; "
            "se usara cuantizacion INT8 post-training."
        )
        return None

    try:
        quantize_model = tfmot.quantization.keras.quantize_model
        qat_model = quantize_model(float_model)
    except Exception as exc:
        print("No se pudo crear el modelo QAT; se usara INT8 post-training.")
        print(f"Motivo: {exc}")
        return None

    compile_for_training(qat_model, learning_rate=learning_rate)
    return qat_model


def make_representative_dataset(
    samples: List[SampleItem],
    batch_size: int,
    max_batches: int,
) -> Callable[[], Iterable[list[tf.Tensor]]]:
    ds = make_dataset(samples, batch_size=batch_size, shuffle=False)

    def representative_dataset():
        if ds is None:
            return
        for images, _ in ds.take(max_batches):
            yield [tf.cast(images, tf.float32).numpy()]

    return representative_dataset


def quantize_input_for_tflite(image: tf.Tensor, input_details: dict):
    dtype = input_details["dtype"]
    if dtype == np_float32_dtype():
        return image.numpy()

    scale, zero_point = input_details["quantization"]
    if scale == 0:
        raise ValueError("Invalid input quantization scale: 0")

    values = image.numpy() / scale + zero_point
    if dtype == np_int8_dtype():
        values = values.clip(-128, 127)
    elif dtype == np_uint8_dtype():
        values = values.clip(0, 255)
    return values.round().astype(dtype)


def np_float32_dtype():
    return tf.as_dtype(tf.float32).as_numpy_dtype


def np_int8_dtype():
    return tf.as_dtype(tf.int8).as_numpy_dtype


def np_uint8_dtype():
    return tf.as_dtype(tf.uint8).as_numpy_dtype


def evaluate_tflite_model(tflite_path: str | Path, dataset: Optional[tf.data.Dataset]) -> float:
    """Evaluate a TFLite model accuracy over a batched tf.data dataset."""
    if dataset is None:
        return 0.0

    interpreter = tf.lite.Interpreter(model_path=Path(tflite_path).as_posix())
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    correct = 0
    total = 0
    for images, labels in dataset:
        for image, label in zip(images, labels):
            image = tf.expand_dims(image, axis=0)
            interpreter.set_tensor(input_details["index"], quantize_input_for_tflite(image, input_details))
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])[0]
            pred = int(output.argmax())
            correct += int(pred == int(label.numpy()))
            total += 1
    return correct / total if total else 0.0


def print_tflite_io_details(tflite_path: str | Path):
    interpreter = tf.lite.Interpreter(model_path=Path(tflite_path).as_posix())
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    print("INT8 TFLite IO:")
    print(
        "  input:",
        input_detail["shape"].tolist(),
        input_detail["dtype"],
        "quantization=",
        input_detail["quantization"],
    )
    print(
        "  output:",
        output_detail["shape"].tolist(),
        output_detail["dtype"],
        "quantization=",
        output_detail["quantization"],
    )


def predict_image(model: tf.keras.Model, image_path: str | Path):
    batch = load_single_image(image_path)
    logits = model(batch, training=False)
    probabilities = tf.nn.softmax(logits, axis=-1)[0]
    pred = int(tf.argmax(probabilities).numpy())
    print(f"Prediction: {pred}")
    print("Probabilities:", ", ".join(f"{p:.4f}" for p in probabilities.numpy()))


def parse_args():
    parser = argparse.ArgumentParser(description="Train TensorFlow SkipPoolCNN for quadrant classification.")
    parser.add_argument("--base_dir", type=str, default=None, help="Root directory with dataset_grupo_*_reescalado")
    parser.add_argument("--output_dir", type=str, default="outputs_tf", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_export_tflite", action="store_true", help="Do not export TFLite models")
    parser.add_argument("--no_export_int8", action="store_true", help="Do not export full int8 TFLite model")
    parser.add_argument("--no_qat", action="store_true", help="Skip quantization-aware fine-tuning")
    parser.add_argument("--qat_epochs", type=int, default=5, help="Epochs for quantization-aware fine-tuning")
    parser.add_argument("--qat_lr", type=float, default=1e-4, help="Learning rate for quantization-aware fine-tuning")
    parser.add_argument("--representative_batches", type=int, default=500, help="Calibration images for INT8 export")
    parser.add_argument("--model_path", type=str, default=None, help="Saved .keras model path for --predict")
    parser.add_argument("--predict", type=str, default=None, help="Image path for inference")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.predict and args.model_path:
        model = tf.keras.models.load_model(args.model_path, compile=False)
        predict_image(model, args.predict)
        return

    base_dir = Path(args.base_dir) if args.base_dir else resolve_default_base_dir()
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    tflite_dir = output_dir / "tflite"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tflite_dir.mkdir(parents=True, exist_ok=True)

    samples = gather_samples(base_dir)
    if not samples:
        print("No samples found. Check dataset_grupo_*_reescalado folders and etiquetas.txt files.")
        print("Base dir:", base_dir.resolve())
        return

    train_samples, val_samples, test_samples = split_samples(
        samples,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

    train_ds = make_dataset(train_samples, args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_dataset(val_samples, args.batch_size, shuffle=False, seed=args.seed)
    test_ds = make_dataset(test_samples, args.batch_size, shuffle=False, seed=args.seed)
    if train_ds is None:
        print("No training samples found after split. Adjust --train_ratio or check labels.")
        return

    model = build_skippoolcnn()
    compile_for_training(model, learning_rate=args.lr)
    model.summary()

    checkpoint_path = checkpoints_dir / f"{MODEL_NAME}.weights.h5"
    callbacks = []
    validation_data = val_ds
    monitor = "val_accuracy" if val_ds is not None else "accuracy"
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path.as_posix(),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    )

    history = model.fit(
        train_ds,
        validation_data=validation_data,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    if checkpoint_path.exists():
        model.load_weights(checkpoint_path.as_posix())

    val_acc, val_loss = evaluate_model(model, val_ds)
    test_acc, test_loss = evaluate_model(model, test_ds)

    keras_path = output_dir / f"{MODEL_NAME}.keras"
    model.save(keras_path.as_posix())

    qat_model = None
    qat_history_rows = []
    qat_keras_path = ""
    qat_weights_path = ""
    qat_val_acc = 0.0
    qat_test_acc = 0.0
    int8_export_model = model
    int8_method = "ptq_full_int8"

    if not args.no_qat and not args.no_export_tflite and not args.no_export_int8:
        qat_model = build_qat_model(model, learning_rate=args.qat_lr)
        if qat_model is not None:
            qat_weights = checkpoints_dir / f"{MODEL_NAME}_qat.weights.h5"
            qat_callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=qat_weights.as_posix(),
                    monitor=monitor,
                    mode="max",
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1,
                )
            ]
            print(f"Fine-tuning QAT for {args.qat_epochs} epochs")
            qat_history = qat_model.fit(
                train_ds,
                validation_data=validation_data,
                epochs=args.qat_epochs,
                callbacks=qat_callbacks,
            )
            if qat_weights.exists():
                qat_model.load_weights(qat_weights.as_posix())

            qat_val_acc, _ = evaluate_model(qat_model, val_ds)
            qat_test_acc, _ = evaluate_model(qat_model, test_ds)
            qat_history_rows = history_to_rows(qat_history, f"{MODEL_NAME}_qat")
            qat_model_path = output_dir / f"{MODEL_NAME}_qat.keras"
            try:
                qat_model.save(qat_model_path.as_posix())
                qat_keras_path = qat_model_path.as_posix()
            except Exception as exc:
                print(f"No se pudo guardar el modelo QAT Keras: {exc}")
            qat_weights_path = qat_weights.as_posix()
            int8_export_model = qat_model
            int8_method = "qat_full_int8"

    tflite_float32_path = ""
    tflite_dynamic_path = ""
    tflite_int8_path = ""
    tflite_int8_acc = 0.0
    if not args.no_export_tflite:
        float32_path = tflite_dir / f"{MODEL_NAME}_float32.tflite"
        dynamic_path = tflite_dir / f"{MODEL_NAME}_dynamic.tflite"
        convert_to_tflite_float32(model, float32_path)
        convert_to_tflite_dynamic_range(model, dynamic_path)
        tflite_float32_path = float32_path.as_posix()
        tflite_dynamic_path = dynamic_path.as_posix()

        if not args.no_export_int8:
            int8_path = tflite_dir / f"{MODEL_NAME}_int8.tflite"
            representative_dataset = make_representative_dataset(
                train_samples,
                batch_size=1,
                max_batches=args.representative_batches,
            )
            convert_to_tflite_int8(int8_export_model, representative_dataset, int8_path)
            tflite_int8_path = int8_path.as_posix()
            tflite_int8_acc = evaluate_tflite_model(int8_path, test_ds)
            print_tflite_io_details(int8_path)

    history_rows = history_to_rows(history, MODEL_NAME) + qat_history_rows
    save_csv(output_dir / "history.csv", history_rows)

    summary_row = {
        "model": MODEL_NAME,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "int8_method": int8_method if tflite_int8_path else "",
        "qat_val_acc": qat_val_acc,
        "qat_test_acc": qat_test_acc,
        "tflite_int8_acc": tflite_int8_acc,
        "params": model.count_params(),
        "size_mb": model.count_params() * 4 / (1024**2),
        "keras_float32_size_mb": keras_path.stat().st_size / (1024**2),
        "tflite_float32_size_mb": (
            Path(tflite_float32_path).stat().st_size / (1024**2) if tflite_float32_path else 0.0
        ),
        "tflite_dynamic_size_mb": (
            Path(tflite_dynamic_path).stat().st_size / (1024**2) if tflite_dynamic_path else 0.0
        ),
        "tflite_int8_size_mb": (
            Path(tflite_int8_path).stat().st_size / (1024**2) if tflite_int8_path else 0.0
        ),
        "keras": keras_path.as_posix(),
        "weights": checkpoint_path.as_posix(),
        "qat_keras": qat_keras_path,
        "qat_weights": qat_weights_path,
        "tflite_float32": tflite_float32_path,
        "tflite_dynamic": tflite_dynamic_path,
        "tflite_int8": tflite_int8_path,
    }
    save_csv(output_dir / "summary.csv", [summary_row])
    plot_paths = generate_evaluation_plots(
        output_dir=output_dir,
        history_rows=history_rows,
        summary_row=summary_row,
        test_ds=test_ds,
        float_model=model,
        qat_model=qat_model,
        tflite_int8_path=tflite_int8_path,
    )
    save_report_md(output_dir / "report.md", summary_row, plot_paths)

    print("Done. Check outputs in", output_dir.resolve())
    print("Evaluation plots:", (output_dir / "plots").resolve())
    print(f"Float validation accuracy: {val_acc:.4f}")
    print(f"Float test accuracy: {test_acc:.4f}")
    if qat_model is not None:
        print(f"QAT validation accuracy: {qat_val_acc:.4f}")
        print(f"QAT test accuracy: {qat_test_acc:.4f}")
    if tflite_int8_path:
        print(f"TFLite INT8 test accuracy: {tflite_int8_acc:.4f}")

    if args.predict:
        predict_image(model, args.predict)


if __name__ == "__main__":
    main()
