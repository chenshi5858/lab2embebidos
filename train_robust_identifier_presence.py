"""Train a false-positive-resistant identifier detector for ESP32 cameras.

This pipeline intentionally differs from the earlier experiments:

* exact duplicates are removed before splitting;
* nearby frames stay together in capture blocks, avoiding optimistic leakage;
* false negatives and false positives have separate configurable costs;
* high-scoring negative training images are mined for a fine-tuning stage;
* the deployment threshold is selected on the INT8 validation predictions,
  targeting a minimum specificity instead of maximizing F1 alone.

The exported network only uses TensorFlow Lite Micro friendly operators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

import train_identifier_presence as base
from skippoolcnn_tf_presence_binary_version import predict_tflite_probabilities


MODEL_NAME = "robust_identifier_presence"
MODEL_VARIABLE = "g_robust_identifier_presence_model"


@dataclass(frozen=True)
class BlockedSample:
    sample: base.Sample
    block_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a robust INT8 presence model with hard-negative mining."
    )
    parser.add_argument("--base_dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs") / f"{MODEL_NAME}_all",
    )
    parser.add_argument("--source", choices=("esp", "celular", "all"), default="all")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--fine_tune_epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--fine_tune_learning_rate", type=float, default=1e-4)
    parser.add_argument("--block_size", type=int, default=40)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--false_positive_cost",
        type=float,
        default=2.0,
        help="Extra weight applied to absent images during training.",
    )
    parser.add_argument(
        "--false_negative_cost",
        type=float,
        default=1.0,
        help="Extra weight applied to present images during training.",
    )
    parser.add_argument("--hard_negative_fraction", type=float, default=0.50)
    parser.add_argument("--hard_negative_repeats", type=int, default=2)
    parser.add_argument(
        "--min_specificity",
        type=float,
        default=0.95,
        help="Minimum validation specificity requested when selecting the INT8 threshold.",
    )
    parser.add_argument(
        "--min_recall",
        type=float,
        default=0.50,
        help="Minimum validation recall requested when selecting the INT8 threshold.",
    )
    parser.add_argument("--representative_samples", type=int, default=400)
    parser.add_argument(
        "--hard_negative_dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional directory of known false-positive images. Every image is labelled absent. "
            "May be supplied more than once."
        ),
    )
    parser.add_argument(
        "--firmware_main",
        type=Path,
        default=None,
        help="Optional firmware main directory that receives the generated C array.",
    )
    parser.add_argument("--scan_only", action="store_true")
    return parser.parse_args()


def scan_extra_negatives(base_dir: Path, directories: list[Path]) -> list[base.Sample]:
    samples: list[base.Sample] = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for index, configured_path in enumerate(directories):
        directory = configured_path if configured_path.is_absolute() else base_dir / configured_path
        directory = directory.resolve()
        if not directory.is_dir():
            raise SystemExit(f"Hard-negative directory does not exist: {directory}")
        for image_path in sorted(directory.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in extensions:
                samples.append(
                    base.Sample(
                        image_path=image_path,
                        label=0,
                        group=f"field_hard_negatives_{index}",
                        source="esp",
                        label_file=directory,
                    )
                )
    return samples


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_exact_duplicates(
    samples: list[base.Sample],
) -> tuple[list[base.Sample], dict[str, int]]:
    by_digest: dict[str, list[base.Sample]] = defaultdict(list)
    for sample in samples:
        by_digest[file_digest(sample.image_path)].append(sample)

    unique: list[base.Sample] = []
    duplicates = 0
    conflicts = 0
    for digest_samples in by_digest.values():
        labels = {sample.label for sample in digest_samples}
        if len(labels) > 1:
            conflicts += len(digest_samples)
            continue
        unique.append(digest_samples[0])
        duplicates += len(digest_samples) - 1
    return unique, {
        "input_samples": len(samples),
        "unique_samples": len(unique),
        "duplicates_removed": duplicates,
        "conflicting_duplicates_removed": conflicts,
    }


def make_capture_blocks(
    samples: list[base.Sample], block_size: int
) -> dict[str, list[base.Sample]]:
    if block_size < 2:
        raise SystemExit("--block_size must be at least 2")
    sequences: dict[tuple[str, str], list[base.Sample]] = defaultdict(list)
    for sample in samples:
        sequences[(sample.group, sample.source)].append(sample)

    blocks: dict[str, list[base.Sample]] = {}
    for (group, source), sequence in sorted(sequences.items()):
        sequence.sort(key=lambda item: item.image_path.name.lower())
        for start in range(0, len(sequence), block_size):
            block_id = f"{group}/{source}/block_{start // block_size:03d}"
            blocks[block_id] = sequence[start : start + block_size]
    return blocks


def split_capture_blocks(
    blocks: dict[str, list[base.Sample]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[BlockedSample], list[BlockedSample], list[BlockedSample]]:
    """Split whole temporal blocks within every dataset group/source."""
    grouped_blocks: dict[str, list[str]] = defaultdict(list)
    for block_id in blocks:
        sequence_id = "/".join(block_id.split("/")[:2])
        grouped_blocks[sequence_id].append(block_id)

    split_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sequence_id, block_ids in sorted(grouped_blocks.items()):
        rng = random.Random(f"{seed}:{sequence_id}")
        rng.shuffle(block_ids)
        count = len(block_ids)
        if count < 3:
            split_ids["train"].extend(block_ids)
            continue
        n_test = max(1, int(round(count * test_fraction)))
        n_val = max(1, int(round(count * val_fraction)))
        while n_test + n_val >= count:
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        split_ids["test"].extend(block_ids[:n_test])
        split_ids["val"].extend(block_ids[n_test : n_test + n_val])
        split_ids["train"].extend(block_ids[n_test + n_val :])

    def materialize(name: str) -> list[BlockedSample]:
        result = [
            BlockedSample(sample, block_id)
            for block_id in split_ids[name]
            for sample in blocks[block_id]
        ]
        random.Random(f"{seed}:{name}").shuffle(result)
        counts = Counter(item.sample.label for item in result)
        if not result or counts[0] == 0 or counts[1] == 0:
            raise SystemExit(
                f"Capture-block {name} split has one class only: "
                f"absent={counts[0]}, present={counts[1]}. Adjust --block_size or fractions."
            )
        return result

    train, val, test = (materialize(name) for name in ("train", "val", "test"))
    return train, val, test


def unwrap(samples: list[BlockedSample]) -> list[base.Sample]:
    return [item.sample for item in samples]


def augment_image(image: tf.Tensor, img_size: int) -> tf.Tensor:
    pad = 8
    image = tf.image.resize_with_crop_or_pad(image, img_size + pad, img_size + pad)
    image = tf.image.random_crop(image, [img_size, img_size, 1])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=26.0)
    image = tf.image.random_contrast(image, lower=0.68, upper=1.35)
    gamma = tf.random.uniform([], 0.78, 1.28)
    image = 255.0 * tf.image.adjust_gamma(tf.clip_by_value(image / 255.0, 0.0, 1.0), gamma)
    image += tf.random.normal(tf.shape(image), stddev=tf.random.uniform([], 0.0, 7.0))

    # Mild blur emulates motion and focus variation without removing the pattern.
    image = tf.cond(
        tf.random.uniform([]) < 0.25,
        lambda: tf.nn.avg_pool2d(image[None, ...], 3, 1, "SAME")[0],
        lambda: image,
    )
    return tf.clip_by_value(image, 0.0, 255.0)


def make_dataset(
    samples: list[base.Sample],
    img_size: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = [str(sample.image_path) for sample in samples]
    labels = [float(sample.label) for sample in samples]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(samples), seed=seed, reshuffle_each_iteration=True)

    def load(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = base.decode_resize_image(path, img_size)
        if training:
            image = augment_image(image, img_size)
        return image, tf.reshape(tf.cast(label, tf.float32), [1])

    return (
        dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def conv_bn_relu(
    x: tf.Tensor, filters: int, stride: int, name: str
) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(2e-4),
        name=f"{name}_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    return tf.keras.layers.ReLU(name=f"{name}_relu")(x)


def build_model(img_size: int, learning_rate: float) -> tf.keras.Model:
    if img_size != 96:
        raise ValueError("The deployment model currently requires 96x96 input.")
    inputs = tf.keras.Input((96, 96, 1), name="grayscale_96x96")
    scaled = tf.keras.layers.Rescaling(1.0 / 255.0, name="scale_to_0_1")(inputs)
    raw_max = tf.keras.layers.MaxPooling2D(16, 16, name="raw_max_6x6")(scaled)
    raw_avg = tf.keras.layers.AveragePooling2D(16, 16, name="raw_avg_6x6")(scaled)

    x = scaled
    for index, (filters, stride) in enumerate(
        ((8, 2), (16, 2), (24, 2), (32, 2), (40, 1)), 1
    ):
        x = conv_bn_relu(x, filters, stride, f"stage{index}")
    x = tf.keras.layers.Concatenate(name="multiscale_fusion")([x, raw_max, raw_avg])
    x = tf.keras.layers.Flatten(name="spatial_flatten")(x)
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(3e-4),
        name="spatial_head",
    )(x)
    x = tf.keras.layers.Dropout(0.30, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="presence")(x)

    model = tf.keras.Model(inputs, outputs, name=MODEL_NAME)
    compile_model(model, learning_rate)
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.03),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )


def class_weights(
    samples: list[base.Sample], false_positive_cost: float, false_negative_cost: float
) -> dict[int, float]:
    weights = base.compute_class_weight(samples)
    weights[0] *= false_positive_cost
    weights[1] *= false_negative_cost
    return weights


def callbacks(checkpoint: Path, patience: int = 14) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=5e-6,
            verbose=1,
        ),
    ]


def mine_hard_negatives(
    model: tf.keras.Model,
    train: list[base.Sample],
    img_size: int,
    batch_size: int,
    fraction: float,
) -> tuple[list[base.Sample], list[tuple[float, base.Sample]]]:
    negatives = [sample for sample in train if sample.label == 0]
    probabilities = base.predict_probabilities(model, negatives, img_size, batch_size)
    ranked = sorted(zip(probabilities.tolist(), negatives), key=lambda item: item[0], reverse=True)
    count = max(1, int(round(len(ranked) * min(max(fraction, 0.0), 1.0))))
    selected = ranked[:count]
    return [sample for _, sample in selected], selected


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    min_specificity: float,
    min_recall: float,
) -> tuple[float, dict[str, float | int], str]:
    candidates: list[tuple[float, dict[str, float | int]]] = []
    for threshold in np.linspace(0.05, 0.99, 189):
        metrics = base.metrics_from_probabilities(labels, probabilities, float(threshold))
        candidates.append((float(threshold), metrics))

    feasible = [
        item
        for item in candidates
        if float(item[1]["specificity"]) >= min_specificity
        and float(item[1]["recall"]) >= min_recall
    ]
    if feasible:
        threshold, metrics = max(
            feasible,
            key=lambda item: (
                float(item[1]["recall"]),
                float(item[1]["precision"]),
                float(item[1]["specificity"]),
                -item[0],
            ),
        )
        mode = f"specificity>={min_specificity:.3f}, recall>={min_recall:.3f}"
    else:
        threshold, metrics = max(
            candidates,
            key=lambda item: (
                float(item[1]["specificity"]),
                float(item[1]["recall"]),
                float(item[1]["precision"]),
            ),
        )
        mode = "fallback: maximum validation specificity"
    return threshold, metrics, mode


def write_manifest(path: Path, samples: list[BlockedSample], base_dir: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "group", "source", "capture_block"])
        for item in samples:
            sample = item.sample
            writer.writerow(
                [
                    sample.image_path.relative_to(base_dir),
                    sample.label,
                    sample.group,
                    sample.source,
                    item.block_id,
                ]
            )


def write_c_array(
    model_bytes: bytes, output_dir: Path, threshold: float
) -> tuple[Path, Path]:
    header = output_dir / f"{MODEL_NAME}_model_data.h"
    source = output_dir / f"{MODEL_NAME}_model_data.cc"
    header.write_text(
        "\n".join(
            [
                "#pragma once",
                "",
                f"extern const unsigned char {MODEL_VARIABLE}[];",
                f"extern const int {MODEL_VARIABLE}_len;",
                f"constexpr float kRobustPresenceThreshold = {threshold:.8f}f;",
                "",
            ]
        ),
        encoding="utf-8",
    )
    values = [f"0x{byte:02x}" for byte in model_bytes]
    rows = ["  " + ", ".join(values[index : index + 12]) for index in range(0, len(values), 12)]
    source.write_text(
        "\n".join(
            [
                f'#include "{header.name}"',
                "",
                f"alignas(16) const unsigned char {MODEL_VARIABLE}[] = {{",
                ",\n".join(rows),
                "};",
                f"const int {MODEL_VARIABLE}_len = {len(model_bytes)};",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return source, header


def metrics_line(name: str, metrics: dict[str, float | int]) -> str:
    return (
        f"| {name} | {float(metrics['accuracy']):.4f} | "
        f"{float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | "
        f"{float(metrics['specificity']):.4f} | {float(metrics['balanced_accuracy']):.4f} | "
        f"{float(metrics['f1']):.4f} | {int(metrics['tn'])} | {int(metrics['fp'])} | "
        f"{int(metrics['fn'])} | {int(metrics['tp'])} |"
    )


def write_report(
    path: Path,
    args: argparse.Namespace,
    duplicate_stats: dict[str, int],
    model: tf.keras.Model,
    split_samples: dict[str, list[BlockedSample]],
    threshold: float,
    threshold_mode: str,
    metrics: dict[str, dict[str, float | int]],
    artifacts: dict[str, str | int | float],
) -> None:
    lines = [
        "# Robust Identifier Presence Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Leakage controls",
        "",
        f"- Exact duplicates removed: `{duplicate_stats['duplicates_removed']}`.",
        f"- Conflicting duplicate labels removed: `{duplicate_stats['conflicting_duplicates_removed']}`.",
        f"- Capture block size: `{args.block_size}` frames.",
        "- Entire capture blocks are assigned to only one split.",
        "",
        "## Training strategy",
        "",
        f"- False-positive cost: `{args.false_positive_cost}`.",
        f"- False-negative cost: `{args.false_negative_cost}`.",
        f"- Hard-negative fraction: `{args.hard_negative_fraction}`.",
        f"- Model parameters: `{model.count_params()}`.",
        "- Augmentation: translation, horizontal flip, brightness, contrast, gamma, noise and mild blur.",
        "",
        "## Split counts",
        "",
    ]
    for name, values in split_samples.items():
        counts = Counter(item.sample.label for item in values)
        blocks = len({item.block_id for item in values})
        lines.append(
            f"- {name}: `{len(values)}` images, absent `{counts[0]}`, present `{counts[1]}`, blocks `{blocks}`."
        )
    lines.extend(
        [
            "",
            "## INT8 decision rule",
            "",
            f"- Threshold: `{threshold:.4f}`.",
            f"- Selection: {threshold_mode}.",
            "- Threshold selection used INT8 validation outputs, not test outputs.",
            "",
            "## Metrics",
            "",
            "| split | accuracy | precision | recall | specificity | balanced_accuracy | f1 | tn | fp | fn | tp |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(metrics_line(name, values) for name, values in metrics.items())
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- {name}: `{value}`" for name, value in artifacts.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else base_dir / args.output_dir
    output_dir = output_dir.resolve()

    scanned, scan_stats = base.scan_dataset(base_dir, args.source)
    scanned.extend(scan_extra_negatives(base_dir, args.hard_negative_dir))
    if not scanned:
        raise SystemExit("No labelled samples found.")
    samples, duplicate_stats = remove_exact_duplicates(scanned)
    blocks = make_capture_blocks(samples, args.block_size)
    train_blocked, val_blocked, test_blocked = split_capture_blocks(
        blocks, args.val_fraction, args.test_fraction, args.seed
    )
    split_map = {"train": train_blocked, "val": val_blocked, "test": test_blocked}
    print(json.dumps(duplicate_stats, indent=2))
    for name, values in split_map.items():
        print(name, base.format_counts(unwrap(values)), "blocks=", len({item.block_id for item in values}))
    if args.scan_only:
        return 0

    tf.keras.utils.set_random_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in split_map.items():
        write_manifest(output_dir / f"{name}_manifest.csv", values, base_dir)

    train = unwrap(train_blocked)
    val = unwrap(val_blocked)
    test = unwrap(test_blocked)
    train_ds = make_dataset(train, args.img_size, args.batch_size, True, args.seed)
    val_ds = make_dataset(val, args.img_size, args.batch_size, False, args.seed)
    weights = class_weights(
        train, args.false_positive_cost, args.false_negative_cost
    )
    print("class weights:", weights)

    model = build_model(args.img_size, args.learning_rate)
    model.summary()
    initial_checkpoint = output_dir / f"best_{MODEL_NAME}_initial.keras"
    initial_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=weights,
        callbacks=callbacks(initial_checkpoint),
        verbose=2,
    )
    base.save_history(output_dir / "initial_history.csv", initial_history)
    model = tf.keras.models.load_model(initial_checkpoint)

    hard_negatives, ranked_negatives = mine_hard_negatives(
        model,
        train,
        args.img_size,
        args.batch_size,
        args.hard_negative_fraction,
    )
    with (output_dir / "hard_negatives.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["probability", "image"])
        for probability, sample in ranked_negatives:
            writer.writerow([f"{probability:.8f}", sample.image_path.relative_to(base_dir)])

    fine_tune_samples = train + hard_negatives * max(0, args.hard_negative_repeats)
    fine_tune_ds = make_dataset(
        fine_tune_samples, args.img_size, args.batch_size, True, args.seed + 1
    )
    compile_model(model, args.fine_tune_learning_rate)
    final_checkpoint = output_dir / f"best_{MODEL_NAME}.keras"
    fine_history = model.fit(
        fine_tune_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        class_weight=weights,
        callbacks=callbacks(final_checkpoint, patience=8),
        verbose=2,
    )
    base.save_history(output_dir / "hard_negative_history.csv", fine_history)
    model = tf.keras.models.load_model(final_checkpoint)

    float_bytes = base.convert_float_tflite(model, args.img_size)
    float_path = output_dir / f"{MODEL_NAME}_float32.tflite"
    float_path.write_bytes(float_bytes)

    representative = train[:]
    random.Random(args.seed).shuffle(representative)
    int8_bytes = base.convert_int8_tflite(
        model, representative, args.img_size, args.representative_samples
    )
    int8_path = output_dir / f"{MODEL_NAME}_int8.tflite"
    int8_path.write_bytes(int8_bytes)

    val_probabilities = predict_tflite_probabilities(int8_bytes, val, args.img_size)
    val_labels = np.asarray([sample.label for sample in val], dtype=np.int32)
    threshold, val_metrics, threshold_mode = select_threshold(
        val_labels,
        val_probabilities,
        args.min_specificity,
        args.min_recall,
    )
    test_probabilities = predict_tflite_probabilities(int8_bytes, test, args.img_size)
    test_labels = np.asarray([sample.label for sample in test], dtype=np.int32)
    metrics = {
        "validation_int8": val_metrics,
        "test_int8": base.metrics_from_probabilities(
            test_labels, test_probabilities, threshold
        ),
    }

    source_path, header_path = write_c_array(int8_bytes, output_dir, threshold)
    artifacts: dict[str, str | int | float] = {
        "keras": str(final_checkpoint.relative_to(base_dir)),
        "float32_tflite": str(float_path.relative_to(base_dir)),
        "int8_tflite": str(int8_path.relative_to(base_dir)),
        "model_data_cc": str(source_path.relative_to(base_dir)),
        "model_data_h": str(header_path.relative_to(base_dir)),
        "int8_size_bytes": len(int8_bytes),
        "threshold": threshold,
    }

    if args.firmware_main is not None:
        firmware_main = args.firmware_main.resolve()
        if not firmware_main.is_dir():
            raise SystemExit(f"Firmware main directory does not exist: {firmware_main}")
        shutil.copy2(source_path, firmware_main / source_path.name)
        shutil.copy2(header_path, firmware_main / header_path.name)
        artifacts["deployed_to"] = str(firmware_main)

    write_report(
        output_dir / "report.md",
        args,
        duplicate_stats,
        model,
        split_map,
        threshold,
        threshold_mode,
        metrics,
        artifacts,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("Done:", output_dir)
    print("Threshold:", threshold, threshold_mode)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
