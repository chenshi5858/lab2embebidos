"""Train a tiny TensorFlow model for identifier presence detection.

The label files are expected at:
    dataset_grupo_*_reescalado/<source>/etiquetas.txt

Each label row is:
    image_name, presence_label, ignored_value

Only the first numeric column is used. Any value different from 0 is treated
as "identifier present".
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ModuleNotFoundError:  # Allows --scan_only without TensorFlow installed.
    np = None

try:
    import tensorflow as tf
except ModuleNotFoundError:  # Allows --scan_only without TensorFlow installed.
    tf = None


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int
    group: str
    source: str
    label_file: Path


@dataclass
class ScanStats:
    label_files: int = 0
    missing_images: int = 0
    bad_lines: int = 0
    per_source: dict[str, Counter] | None = None

    def __post_init__(self) -> None:
        if self.per_source is None:
            self.per_source = defaultdict(Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export a lightweight TensorFlow/TFLite model for identifier presence."
    )
    parser.add_argument("--base_dir", type=Path, default=Path("."), help="Project directory.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs") / "identifier_presence",
        help="Directory for model artifacts and reports.",
    )
    parser.add_argument(
        "--source",
        choices=("esp", "celular", "all"),
        default="esp",
        help="Image source to train with. The default matches the ESP camera domain.",
    )
    parser.add_argument("--img_size", type=int, default=96, help="Square grayscale input size.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument(
        "--positive_weight_multiplier",
        type=float,
        default=1.0,
        help="Extra multiplier for the positive class weight. Increase it to reduce false negatives.",
    )
    parser.add_argument(
        "--model_variant",
        choices=("spatial_tiny", "spatial_small"),
        default="spatial_tiny",
        help="Model architecture. spatial_tiny is the default TFLite Micro friendly model.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_mode",
        choices=("stratified", "group"),
        default="stratified",
        help="Use stratified image split or hold out whole dataset groups.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Presence probability threshold. If omitted, a validation threshold is selected.",
    )
    parser.add_argument(
        "--threshold_metric",
        choices=("balanced_accuracy", "f1", "f2", "accuracy", "mcc"),
        default="f2",
        help="Validation metric used to choose the threshold when --threshold is omitted.",
    )
    parser.add_argument(
        "--min_recall",
        type=float,
        default=None,
        help="If set, choose the highest-specificity validation threshold with at least this recall.",
    )
    parser.add_argument(
        "--representative_samples",
        type=int,
        default=200,
        help="Number of training images used for full integer quantization calibration.",
    )
    parser.add_argument("--scan_only", action="store_true", help="Only print dataset statistics.")
    parser.add_argument("--no_int8", action="store_true", help="Skip full integer TFLite export.")
    parser.add_argument("--no_c_array", action="store_true", help="Skip C/C++ array export for TFLite Micro.")
    return parser.parse_args()


def require_runtime() -> None:
    if tf is None or np is None:
        raise SystemExit(
            "TensorFlow is not installed in this Python environment.\n"
            "Create a Python 3.11/3.12 venv and run: python -m pip install -r requirements.txt"
        )


def scan_dataset(base_dir: Path, source_filter: str) -> tuple[list[Sample], ScanStats]:
    stats = ScanStats()
    samples: list[Sample] = []
    label_files = sorted(base_dir.glob("dataset_grupo_*_reescalado/*/etiquetas.txt"))
    for label_file in label_files:
        source = label_file.parent.name
        if source_filter != "all" and source != source_filter:
            continue

        stats.label_files += 1
        group = label_file.parents[1].name
        for line_no, raw_line in enumerate(
            label_file.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
        ):
            line = raw_line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                stats.bad_lines += 1
                continue

            try:
                label = 1 if int(parts[1]) != 0 else 0
            except ValueError:
                stats.bad_lines += 1
                continue

            image_path = label_file.parent / parts[0]
            if not image_path.exists():
                stats.missing_images += 1
                continue

            samples.append(
                Sample(
                    image_path=image_path,
                    label=label,
                    group=group,
                    source=source,
                    label_file=label_file,
                )
            )
            stats.per_source[f"{group}/{source}"][label] += 1

    return samples, stats


def summarize_samples(samples: Iterable[Sample]) -> Counter:
    return Counter(sample.label for sample in samples)


def split_stratified(
    samples: list[Sample], val_fraction: float, test_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    rng = random.Random(seed)
    by_label: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)

    train: list[Sample] = []
    val: list[Sample] = []
    test: list[Sample] = []

    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        n = len(label_samples)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))

        if test_fraction > 0 and n >= 6:
            n_test = max(1, n_test)
        if val_fraction > 0 and n - n_test >= 6:
            n_val = max(1, n_val)

        test.extend(label_samples[:n_test])
        val.extend(label_samples[n_test : n_test + n_val])
        train.extend(label_samples[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def split_by_group(
    samples: list[Sample], val_fraction: float, test_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    rng = random.Random(seed)
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group].append(sample)

    groups = sorted(grouped)
    rng.shuffle(groups)
    if len(groups) < 3:
        return split_stratified(samples, val_fraction, test_fraction, seed)

    n_groups = len(groups)
    n_test = max(1, int(round(n_groups * test_fraction)))
    n_val = max(1, int(round(n_groups * val_fraction)))
    n_train = n_groups - n_test - n_val
    if n_train < 1:
        n_test = 1
        n_val = 1

    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test : n_test + n_val])
    train_groups = set(groups[n_test + n_val :])

    train = [sample for group in train_groups for sample in grouped[group]]
    val = [sample for group in val_groups for sample in grouped[group]]
    test = [sample for group in test_groups for sample in grouped[group]]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def split_samples(args: argparse.Namespace, samples: list[Sample]) -> tuple[list[Sample], list[Sample], list[Sample]]:
    if args.split_mode == "group":
        train, val, test = split_by_group(samples, args.val_fraction, args.test_fraction, args.seed)
    else:
        train, val, test = split_stratified(samples, args.val_fraction, args.test_fraction, args.seed)

    for split_name, split in (("train", train), ("val", val), ("test", test)):
        counts = summarize_samples(split)
        if len(split) == 0 or counts[0] == 0 or counts[1] == 0:
            raise SystemExit(
                f"The {split_name} split is empty or has one class only: "
                f"absent={counts[0]}, present={counts[1]}. "
                "Try --split_mode stratified, --source esp, or different split fractions."
            )

    return train, val, test


def decode_resize_image(path: "tf.Tensor", img_size: int) -> "tf.Tensor":
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=1, expand_animations=False)
    image.set_shape([None, None, 1])
    image = tf.image.resize(image, [img_size, img_size], method="area")
    image = tf.cast(image, tf.float32)
    return image


def augment_image(image: "tf.Tensor", img_size: int) -> "tf.Tensor":
    # Small translations and lighting changes improve robustness without changing the label.
    pad = 8
    image = tf.image.resize_with_crop_or_pad(image, img_size + pad, img_size + pad)
    image = tf.image.random_crop(image, [img_size, img_size, 1])
    image = tf.image.random_brightness(image, max_delta=18.0)
    image = tf.image.random_contrast(image, lower=0.78, upper=1.22)
    return tf.clip_by_value(image, 0.0, 255.0)


def make_dataset(
    samples: list[Sample], img_size: int, batch_size: int, training: bool, seed: int
) -> "tf.data.Dataset":
    paths = [str(sample.image_path) for sample in samples]
    labels = [float(sample.label) for sample in samples]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(len(samples), seed=seed, reshuffle_each_iteration=True)

    def load_example(path: "tf.Tensor", label: "tf.Tensor") -> tuple["tf.Tensor", "tf.Tensor"]:
        image = decode_resize_image(path, img_size)
        if training:
            image = augment_image(image, img_size)
        return image, tf.reshape(tf.cast(label, tf.float32), [1])

    return (
        dataset.map(load_example, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_tiny_presence_model(
    img_size: int, learning_rate: float, model_variant: str = "spatial_tiny"
) -> "tf.keras.Model":
    if model_variant == "spatial_tiny":
        widths = (8, 16, 24, 32)
        dense_units = 32
        dropout = 0.20
    elif model_variant == "spatial_small":
        widths = (12, 24, 36, 48)
        dense_units = 48
        dropout = 0.25
    else:
        raise ValueError(f"Unsupported model_variant: {model_variant}")

    inputs = tf.keras.Input(shape=(img_size, img_size, 1), name="grayscale_96x96")

    # Keep preprocessing in the graph so the ESP firmware can feed raw 0..255 pixels.
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="scale_to_0_1")(inputs)
    for index, filters in enumerate(widths, 1):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", activation="relu", name=f"conv_{index}"
        )(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"pool_{index}")(x)

    # Flatten keeps spatial evidence from small identifiers that average pooling can erase.
    x = tf.keras.layers.Flatten(name="flatten_spatial_features")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="spatial_head")(x)
    x = tf.keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="presence")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"identifier_presence_{model_variant}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def compute_class_weight(samples: list[Sample], positive_weight_multiplier: float = 1.0) -> dict[int, float]:
    counts = summarize_samples(samples)
    total = counts[0] + counts[1]
    return {
        0: total / (2.0 * max(1, counts[0])),
        1: (total / (2.0 * max(1, counts[1]))) * positive_weight_multiplier,
    }


def predict_probabilities(
    model: "tf.keras.Model",
    samples: list[Sample],
    img_size: int,
    batch_size: int,
) -> "np.ndarray":
    dataset = make_dataset(samples, img_size, batch_size, training=False, seed=0)
    return model.predict(dataset, verbose=0).reshape(-1)


def metrics_from_probabilities(
    labels: "np.ndarray", probabilities: "np.ndarray", threshold: float
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(np.int32)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    accuracy = float((tp + tn) / max(1, len(labels)))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    specificity = float(tn / max(1, tn + fp))
    balanced_accuracy = float((recall + specificity) / 2.0)
    f1 = float(2 * precision * recall / max(1e-8, precision + recall))
    f2 = float(5 * precision * recall / max(1e-8, (4 * precision) + recall))
    mcc_denominator = np.sqrt(
        max(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = float(((tp * tn) - (fp * fn)) / mcc_denominator)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "f2": f2,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_predictions(
    model: "tf.keras.Model",
    samples: list[Sample],
    img_size: int,
    batch_size: int,
    threshold: float,
) -> dict[str, float | int]:
    probabilities = predict_probabilities(model, samples, img_size, batch_size)
    labels = np.array([sample.label for sample in samples], dtype=np.int32)
    return metrics_from_probabilities(labels, probabilities, threshold)


def threshold_score(metrics: dict[str, float | int], metric_name: str) -> tuple[float, float, float]:
    primary = float(metrics[metric_name])
    return primary, float(metrics["accuracy"]), float(metrics["f1"])


def find_best_threshold(
    model: "tf.keras.Model",
    samples: list[Sample],
    img_size: int,
    batch_size: int,
    threshold_metric: str,
    min_recall: float | None,
) -> tuple[float, dict[str, float | int]]:
    probabilities = predict_probabilities(model, samples, img_size, batch_size)
    labels = np.array([sample.label for sample in samples], dtype=np.int32)
    best_threshold = 0.5
    best_metrics = metrics_from_probabilities(labels, probabilities, best_threshold)
    if min_recall is not None:
        best_key = (-1.0, -1.0, -1.0)
    else:
        best_key = threshold_score(best_metrics, threshold_metric)

    for threshold in np.linspace(0.05, 0.95, 91):
        metrics = metrics_from_probabilities(labels, probabilities, float(threshold))
        if min_recall is not None:
            if float(metrics["recall"]) < min_recall:
                continue
            key = (
                float(metrics["specificity"]),
                float(metrics["precision"]),
                float(metrics["accuracy"]),
            )
        else:
            key = threshold_score(metrics, threshold_metric)
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key

    if min_recall is not None and best_key[0] < 0:
        # Fall back to the most sensitive threshold if the requested recall is unreachable.
        best_threshold = 0.05
        best_metrics = metrics_from_probabilities(labels, probabilities, best_threshold)

    return best_threshold, best_metrics


def write_manifest(path: Path, samples: list[Sample], base_dir: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label", "group", "source", "label_file"])
        for sample in samples:
            writer.writerow(
                [
                    sample.image_path.relative_to(base_dir),
                    sample.label,
                    sample.group,
                    sample.source,
                    sample.label_file.relative_to(base_dir),
                ]
            )


def write_dataset_summary(path: Path, stats: ScanStats, samples: list[Sample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "total", "absent_0", "present_1"])
        for source, counts in sorted(stats.per_source.items()):
            writer.writerow([source, counts[0] + counts[1], counts[0], counts[1]])
        totals = summarize_samples(samples)
        writer.writerow(["TOTAL", totals[0] + totals[1], totals[0], totals[1]])


def save_history(path: Path, history: "tf.keras.callbacks.History") -> None:
    keys = list(history.history)
    rows = zip(*(history.history[key] for key in keys))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for index, row in enumerate(rows, 1):
            writer.writerow([index, *row])


def make_frozen_tflite_converter(model: "tf.keras.Model", img_size: int) -> "tf.lite.TFLiteConverter":
    # Freezing avoids TF Lite converter crashes around Keras resource variables.
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    @tf.function(
        input_signature=[tf.TensorSpec([1, img_size, img_size, 1], tf.float32, name="grayscale_96x96")],
        autograph=False,
    )
    def serving(image: "tf.Tensor") -> "tf.Tensor":
        return model(image, training=False)

    concrete = serving.get_concrete_function()
    frozen = convert_variables_to_constants_v2(concrete)
    return tf.lite.TFLiteConverter.from_concrete_functions([frozen])


def convert_float_tflite(model: "tf.keras.Model", img_size: int) -> bytes:
    converter = make_frozen_tflite_converter(model, img_size)
    return converter.convert()


def convert_int8_tflite(
    model: "tf.keras.Model",
    train_samples: list[Sample],
    img_size: int,
    representative_samples: int,
) -> bytes:
    rep_dataset = make_dataset(
        train_samples[:representative_samples],
        img_size=img_size,
        batch_size=1,
        training=False,
        seed=0,
    )

    def representative_dataset() -> Iterable[list["tf.Tensor"]]:
        for image, _label in rep_dataset:
            yield [image]

    converter = make_frozen_tflite_converter(model, img_size)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def inspect_tflite(model_bytes: bytes) -> dict[str, object]:
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return {
        "input_name": str(input_detail["name"]),
        "input_shape": [int(value) for value in input_detail["shape"]],
        "input_dtype": str(input_detail["dtype"]),
        "input_quantization": tuple(float(value) for value in input_detail["quantization"]),
        "output_name": str(output_detail["name"]),
        "output_shape": [int(value) for value in output_detail["shape"]],
        "output_dtype": str(output_detail["dtype"]),
        "output_quantization": tuple(float(value) for value in output_detail["quantization"]),
    }


def write_c_array(model_bytes: bytes, output_dir: Path, var_name: str) -> tuple[Path, Path]:
    header_path = output_dir / "identifier_presence_model_data.h"
    source_path = output_dir / "identifier_presence_model_data.cc"

    header_path.write_text(
        "\n".join(
            [
                "#pragma once",
                "",
                f"extern const unsigned char {var_name}[];",
                f"extern const int {var_name}_len;",
                "",
            ]
        ),
        encoding="utf-8",
    )

    values = [f"0x{byte:02x}" for byte in model_bytes]
    lines = []
    for index in range(0, len(values), 12):
        lines.append("  " + ", ".join(values[index : index + 12]))

    source_path.write_text(
        "\n".join(
            [
                '#include "identifier_presence_model_data.h"',
                "",
                f"alignas(16) const unsigned char {var_name}[] = {{",
                ",\n".join(lines),
                "};",
                f"const int {var_name}_len = {len(model_bytes)};",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return source_path, header_path


def format_counts(samples: list[Sample]) -> str:
    counts = summarize_samples(samples)
    return f"total={len(samples)}, absent_0={counts[0]}, present_1={counts[1]}"


def model_summary_text(model: "tf.keras.Model") -> str:
    lines = ["Layer (type) | Output shape | Params", "--- | --- | ---"]
    for layer in model.layers:
        try:
            shape = tuple(None if dim is None else int(dim) for dim in layer.output.shape)
        except Exception:  # noqa: BLE001 - summary should never break report generation.
            shape = "unknown"
        lines.append(f"{layer.name} ({layer.__class__.__name__}) | {shape} | {layer.count_params()}")

    trainable = int(sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights))
    non_trainable = int(sum(tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights))
    lines.extend(
        [
            "",
            f"Total params: {model.count_params()}",
            f"Trainable params: {trainable}",
            f"Non-trainable params: {non_trainable}",
        ]
    )
    return "\n".join(lines)


def write_report(
    path: Path,
    args: argparse.Namespace,
    stats: ScanStats,
    train: list[Sample],
    val: list[Sample],
    test: list[Sample],
    model: "tf.keras.Model",
    threshold: float,
    threshold_mode: str,
    metrics: dict[str, dict[str, float | int]],
    artifacts: dict[str, str],
    tflite_info: dict[str, dict[str, object]],
) -> None:
    totals = summarize_samples(train + val + test)
    lines = [
        "# Identifier Presence TensorFlow Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Dataset",
        "",
        f"- Source filter: `{args.source}`",
        f"- Label files used: {stats.label_files}",
        f"- Samples used: {totals[0] + totals[1]}",
        f"- Absent class 0: {totals[0]}",
        f"- Present class 1: {totals[1]}",
        f"- Missing image references skipped: {stats.missing_images}",
        f"- Bad label lines skipped: {stats.bad_lines}",
        "",
        "## Splits",
        "",
        "| split | total | absent_0 | present_1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, samples in (("train", train), ("val", val), ("test", test)):
        counts = summarize_samples(samples)
        lines.append(f"| {name} | {len(samples)} | {counts[0]} | {counts[1]} |")

    lines.extend(
        [
            "",
            "## Model",
            "",
            f"- Variant: `{args.model_variant}`",
            f"- Input: `{args.img_size}x{args.img_size}x1` grayscale, raw pixel range 0..255.",
            f"- Parameters: {model.count_params()}",
            f"- Positive class weight multiplier: `{args.positive_weight_multiplier}`",
            "- Core ops: Conv2D, MaxPool2D, Flatten, Dense, Sigmoid.",
            "",
            "## Metrics",
            "",
            f"Threshold: `{threshold:.4f}` ({threshold_mode})",
            "",
            "| split | accuracy | precision | recall | specificity | balanced_accuracy | f1 | f2 | mcc | tn | fp | fn | tp |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, values in metrics.items():
        lines.append(
            "| {name} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | "
            "{specificity:.4f} | {balanced_accuracy:.4f} | {f1:.4f} | {f2:.4f} | {mcc:.4f} | "
            "{tn} | {fp} | {fn} | {tp} |".format(name=name, **values)
        )

    lines.extend(["", "## Artifacts", ""])
    for name, artifact_path in artifacts.items():
        lines.append(f"- {name}: `{artifact_path}`")

    if tflite_info:
        lines.extend(["", "## TFLite Quantization Info", ""])
        lines.append("```json")
        lines.append(json.dumps(tflite_info, indent=2))
        lines.append("```")

    lines.extend(
        [
            "",
            "## ESP Notes",
            "",
            "- Capture 96x96 grayscale frames on the ESP camera.",
            "- Feed pixels in row-major order. For the int8 model, quantize each pixel with:",
            "  `input_int8 = round(pixel / input_scale + input_zero_point)`.",
            "- The output is a sigmoid probability after dequantization. Presence is true when it is >= threshold.",
            "",
            "## Keras Summary",
            "",
            "```text",
            model_summary_text(model).strip(),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_scan(samples: list[Sample], stats: ScanStats) -> None:
    totals = summarize_samples(samples)
    print(f"label_files={stats.label_files}")
    print(
        f"total={totals[0] + totals[1]} absent_0={totals[0]} present_1={totals[1]} "
        f"missing_images={stats.missing_images} bad_lines={stats.bad_lines}"
    )
    for source, counts in sorted(stats.per_source.items()):
        print(f"{source}: total={counts[0] + counts[1]} absent_0={counts[0]} present_1={counts[1]}")


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = (base_dir / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    samples, stats = scan_dataset(base_dir, args.source)

    if not samples:
        raise SystemExit("No samples found. Check --base_dir and --source.")

    print_scan(samples, stats)
    if args.scan_only:
        return 0

    require_runtime()
    tf.keras.utils.set_random_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = split_samples(args, samples)
    print(f"train: {format_counts(train)}")
    print(f"val:   {format_counts(val)}")
    print(f"test:  {format_counts(test)}")

    write_dataset_summary(output_dir / "dataset_summary.csv", stats, samples)
    write_manifest(output_dir / "train_manifest.csv", train, base_dir)
    write_manifest(output_dir / "val_manifest.csv", val, base_dir)
    write_manifest(output_dir / "test_manifest.csv", test, base_dir)

    train_ds = make_dataset(train, args.img_size, args.batch_size, training=True, seed=args.seed)
    val_ds = make_dataset(val, args.img_size, args.batch_size, training=False, seed=args.seed)

    model = build_tiny_presence_model(args.img_size, args.learning_rate, args.model_variant)
    class_weight = compute_class_weight(train, args.positive_weight_multiplier)
    checkpoint_path = output_dir / "best_identifier_presence.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=14,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    save_history(output_dir / "history.csv", history)

    if checkpoint_path.exists():
        model = tf.keras.models.load_model(checkpoint_path)
    else:
        model.save(checkpoint_path)

    if args.threshold is None:
        threshold, val_metrics = find_best_threshold(
            model,
            val,
            args.img_size,
            args.batch_size,
            args.threshold_metric,
            args.min_recall,
        )
        if args.min_recall is None:
            threshold_mode = f"best validation {args.threshold_metric}"
        else:
            threshold_mode = f"validation recall >= {args.min_recall:g}, max specificity"
    else:
        threshold = float(args.threshold)
        val_metrics = evaluate_predictions(model, val, args.img_size, args.batch_size, threshold)
        threshold_mode = "fixed"

    metrics = {
        "val": val_metrics,
        "test": evaluate_predictions(model, test, args.img_size, args.batch_size, threshold),
    }

    artifacts = {
        "keras": str(checkpoint_path.relative_to(base_dir)),
        "history": str((output_dir / "history.csv").relative_to(base_dir)),
        "dataset_summary": str((output_dir / "dataset_summary.csv").relative_to(base_dir)),
    }
    tflite_info: dict[str, dict[str, object]] = {}

    float_tflite = convert_float_tflite(model, args.img_size)
    float_path = output_dir / "identifier_presence_float32.tflite"
    float_path.write_bytes(float_tflite)
    artifacts["tflite_float32"] = str(float_path.relative_to(base_dir))
    tflite_info["float32"] = inspect_tflite(float_tflite)

    if not args.no_int8:
        try:
            shuffled_train = train[:]
            random.Random(args.seed).shuffle(shuffled_train)
            int8_tflite = convert_int8_tflite(
                model,
                shuffled_train,
                img_size=args.img_size,
                representative_samples=args.representative_samples,
            )
            int8_path = output_dir / "identifier_presence_int8.tflite"
            int8_path.write_bytes(int8_tflite)
            artifacts["tflite_int8"] = str(int8_path.relative_to(base_dir))
            tflite_info["int8"] = inspect_tflite(int8_tflite)

            if not args.no_c_array:
                cc_path, h_path = write_c_array(int8_tflite, output_dir, "g_identifier_presence_model")
                artifacts["tflite_micro_cc"] = str(cc_path.relative_to(base_dir))
                artifacts["tflite_micro_h"] = str(h_path.relative_to(base_dir))
        except Exception as exc:  # noqa: BLE001 - keep float export even if int8 fails.
            artifacts["tflite_int8_error"] = repr(exc)
            print(f"WARNING: int8 export failed: {exc}", file=sys.stderr)

    write_report(
        output_dir / "report.md",
        args,
        stats,
        train,
        val,
        test,
        model,
        threshold,
        threshold_mode,
        metrics,
        artifacts,
        tflite_info,
    )
    artifacts["report"] = str((output_dir / "report.md").relative_to(base_dir))

    print("Done. Artifacts:")
    for name, artifact_path in artifacts.items():
        print(f"  {name}: {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
