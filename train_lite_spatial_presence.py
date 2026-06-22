"""Train a quantization-friendly LiteSpatialFusionCNN presence detector.

This compact architecture learns 12x12 convolutional features and fuses them
with max/average-pooled views of the raw image. A narrow spatial head preserves
small identifier evidence while keeping the full-INT8 model suitable for ESP.

Labels are read as: image_name, presence_label, ignored_quadrant
Only presence_label is used: 0 = absent, non-zero = present.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

import train_identifier_presence as data_utils
from skippoolcnn_tf_presence_binary_version import (
    path_for_report,
    plot_binary_confusion_matrix,
    plot_threshold_analysis,
    plot_training_history,
    predict_tflite_probabilities,
)


MODEL_NAME = "lite_spatial_presence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export a lightweight INT8 LiteSpatialFusionCNN detector."
    )
    parser.add_argument("--base_dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs") / MODEL_NAME,
    )
    parser.add_argument("--source", choices=("esp", "celular", "all"), default="esp")
    parser.add_argument("--variant", choices=("tiny", "small"), default="small")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_mode", choices=("stratified", "group"), default="stratified")
    parser.add_argument(
        "--threshold_metric",
        choices=("balanced_accuracy", "f1", "f2", "accuracy", "mcc"),
        default="f1",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min_recall", type=float, default=None)
    parser.add_argument("--positive_weight_multiplier", type=float, default=1.0)
    parser.add_argument("--representative_samples", type=int, default=250)
    parser.add_argument("--scan_only", action="store_true")
    parser.add_argument("--no_int8", action="store_true")
    parser.add_argument("--no_c_array", action="store_true")
    return parser.parse_args()


def conv_bn_relu(
    x: tf.Tensor,
    filters: int,
    stride: int,
    name: str,
) -> tf.Tensor:
    """Compact quantization-friendly convolution block."""
    x = tf.keras.layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_batch_norm")(x)
    return tf.keras.layers.ReLU(name=f"{name}_relu")(x)


def pooled_pyramid_features(x: tf.Tensor, spatial_size: int, name: str) -> tf.Tensor:
    """Keep average context and strongest local response using pool ops."""
    average = tf.keras.layers.AveragePooling2D(
        pool_size=spatial_size,
        name=f"{name}_average_pool",
    )(x)
    maximum = tf.keras.layers.MaxPooling2D(
        pool_size=spatial_size,
        name=f"{name}_max_pool",
    )(x)
    average = tf.keras.layers.Flatten(name=f"{name}_average_flatten")(average)
    maximum = tf.keras.layers.Flatten(name=f"{name}_max_flatten")(maximum)
    return tf.keras.layers.Concatenate(name=f"{name}_pooled_features")([average, maximum])


def build_lite_spatial_model(
    img_size: int,
    learning_rate: float,
    variant: str = "small",
) -> tf.keras.Model:
    if img_size != 96:
        raise ValueError("TinyPyramidDSCNN currently requires --img_size 96.")
    if variant == "tiny":
        widths = (4, 8, 12, 16)
        head_units = 8
    elif variant == "small":
        widths = (6, 12, 18, 24)
        head_units = 16
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    inputs = tf.keras.Input((img_size, img_size, 1), name="grayscale_96x96")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="scale_to_0_1")(inputs)
    raw_max = tf.keras.layers.MaxPooling2D(8, 8, name="raw_max_pool_12x12")(x)
    raw_average = tf.keras.layers.AveragePooling2D(8, 8, name="raw_average_pool_12x12")(x)
    for index, (filters, stride) in enumerate(
        ((widths[0], 2), (widths[1], 2), (widths[2], 2), (widths[3], 1)),
        1,
    ):
        x = conv_bn_relu(x, filters, stride, f"stage{index}")
    x = tf.keras.layers.Concatenate(name="spatial_fusion")([x, raw_max, raw_average])
    x = tf.keras.layers.Flatten(name="spatial_flatten")(x)
    x = tf.keras.layers.Dense(
        head_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="presence_head",
    )(x)
    x = tf.keras.layers.Dropout(0.15, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="presence")(x)

    model = tf.keras.Model(inputs, outputs, name=f"{MODEL_NAME}_{variant}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def write_c_array(
    model_bytes: bytes,
    output_dir: Path,
    threshold: float,
) -> tuple[Path, Path]:
    header_path = output_dir / f"{MODEL_NAME}_model_data.h"
    source_path = output_dir / f"{MODEL_NAME}_model_data.cc"
    variable = "g_lite_spatial_presence_model"
    header_path.write_text(
        "\n".join(
            [
                "#pragma once",
                "",
                f"extern const unsigned char {variable}[];",
                f"extern const int {variable}_len;",
                f"constexpr float kLiteSpatialPresenceThreshold = {threshold:.8f}f;",
                "",
            ]
        ),
        encoding="utf-8",
    )
    values = [f"0x{byte:02x}" for byte in model_bytes]
    lines = ["  " + ", ".join(values[index : index + 12]) for index in range(0, len(values), 12)]
    source_path.write_text(
        "\n".join(
            [
                f'#include "{header_path.name}"',
                "",
                f"alignas(16) const unsigned char {variable}[] = {{",
                ",\n".join(lines),
                "};",
                f"const int {variable}_len = {len(model_bytes)};",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return source_path, header_path


def save_summary(path: Path, rows: dict[str, dict[str, float | int]]) -> None:
    metric_names = list(next(iter(rows.values())).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", *metric_names])
        writer.writeheader()
        for split, metrics in rows.items():
            writer.writerow({"split": split, **metrics})


def write_report(
    path: Path,
    args: argparse.Namespace,
    model: tf.keras.Model,
    train: list[data_utils.Sample],
    val: list[data_utils.Sample],
    test: list[data_utils.Sample],
    threshold: float,
    threshold_mode: str,
    metrics: dict[str, dict[str, float | int]],
    artifacts: dict[str, str],
    tflite_info: dict[str, dict[str, object]],
) -> None:
    lines = [
        "# LiteSpatialFusionCNN Presence Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Model",
        "",
        f"- Variant: `{args.variant}`",
        f"- Parameters: {model.count_params()}",
        "- Input: `96x96x1`, raw grayscale 0..255.",
        "- Blocks: four Conv2D + BatchNorm + ReLU stages.",
        "- Fusion: learned 12x12 features plus max/average-pooled raw 12x12 views.",
        "",
        "## Dataset",
        "",
        "| split | total | absent | present |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, samples in (("train", train), ("val", val), ("test", test)):
        counts = data_utils.summarize_samples(samples)
        lines.append(f"| {name} | {len(samples)} | {counts[0]} | {counts[1]} |")
    lines.extend(
        [
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
            "{specificity:.4f} | {balanced_accuracy:.4f} | {f1:.4f} | {f2:.4f} | "
            "{mcc:.4f} | {tn} | {fp} | {fn} | {tp} |".format(name=name, **values)
        )
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- {name}: `{artifact}`" for name, artifact in artifacts.items())
    lines.extend(
        [
            "",
            "## Evaluation Plots",
            "",
            "![Training curves](plots/training_curves.png)",
            "",
            "![Threshold analysis](plots/threshold_analysis.png)",
            "",
            "![Keras confusion matrix](plots/confusion_matrix_test.png)",
            "",
            "![INT8 confusion matrix](plots/confusion_matrix_int8.png)",
            "",
            "## TFLite Quantization",
            "",
            "```json",
            json.dumps(tflite_info, indent=2),
            "```",
            "",
            "## Keras Summary",
            "",
            "```text",
            data_utils.model_summary_text(model),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = (base_dir / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    samples, stats = data_utils.scan_dataset(base_dir, args.source)
    if not samples:
        raise SystemExit("No samples found. Check --base_dir and --source.")
    data_utils.print_scan(samples, stats)
    if args.scan_only:
        return 0

    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = data_utils.split_samples(args, samples)
    print("train:", data_utils.format_counts(train))
    print("val:  ", data_utils.format_counts(val))
    print("test: ", data_utils.format_counts(test))

    data_utils.write_dataset_summary(output_dir / "dataset_summary.csv", stats, samples)
    data_utils.write_manifest(output_dir / "train_manifest.csv", train, base_dir)
    data_utils.write_manifest(output_dir / "val_manifest.csv", val, base_dir)
    data_utils.write_manifest(output_dir / "test_manifest.csv", test, base_dir)

    train_ds = data_utils.make_dataset(train, args.img_size, args.batch_size, True, args.seed)
    val_ds = data_utils.make_dataset(val, args.img_size, args.batch_size, False, args.seed)
    model = build_lite_spatial_model(args.img_size, args.learning_rate, args.variant)
    model.summary()
    checkpoint_path = output_dir / f"best_{MODEL_NAME}.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
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
        class_weight=data_utils.compute_class_weight(train, args.positive_weight_multiplier),
        callbacks=callbacks,
        verbose=2,
    )
    data_utils.save_history(output_dir / "history.csv", history)
    model = tf.keras.models.load_model(checkpoint_path)

    if args.threshold is None:
        threshold, val_metrics = data_utils.find_best_threshold(
            model,
            val,
            args.img_size,
            args.batch_size,
            args.threshold_metric,
            args.min_recall,
        )
        threshold_mode = (
            f"best validation {args.threshold_metric}"
            if args.min_recall is None
            else f"validation recall >= {args.min_recall:g}, max specificity"
        )
    else:
        threshold = float(args.threshold)
        val_metrics = data_utils.evaluate_predictions(
            model, val, args.img_size, args.batch_size, threshold
        )
        threshold_mode = "fixed"

    metrics = {
        "val": val_metrics,
        "test": data_utils.evaluate_predictions(
            model, test, args.img_size, args.batch_size, threshold
        ),
    }
    artifacts = {
        "keras": path_for_report(checkpoint_path, base_dir),
        "history": path_for_report(output_dir / "history.csv", base_dir),
    }
    tflite_info: dict[str, dict[str, object]] = {}

    float_bytes = data_utils.convert_float_tflite(model, args.img_size)
    float_path = output_dir / f"{MODEL_NAME}_float32.tflite"
    float_path.write_bytes(float_bytes)
    artifacts["tflite_float32"] = path_for_report(float_path, base_dir)
    tflite_info["float32"] = data_utils.inspect_tflite(float_bytes)

    if not args.no_int8:
        shuffled_train = train[:]
        random.Random(args.seed).shuffle(shuffled_train)
        int8_bytes = data_utils.convert_int8_tflite(
            model,
            shuffled_train,
            args.img_size,
            args.representative_samples,
        )
        int8_path = output_dir / f"{MODEL_NAME}_int8.tflite"
        int8_path.write_bytes(int8_bytes)
        artifacts["tflite_int8"] = path_for_report(int8_path, base_dir)
        tflite_info["int8"] = data_utils.inspect_tflite(int8_bytes)
        int8_probabilities = predict_tflite_probabilities(int8_bytes, test, args.img_size)
        test_labels = np.asarray([sample.label for sample in test], dtype=np.int32)
        metrics["test_int8"] = data_utils.metrics_from_probabilities(
            test_labels, int8_probabilities, threshold
        )
        if not args.no_c_array:
            cc_path, h_path = write_c_array(int8_bytes, output_dir, threshold)
            artifacts["tflite_micro_cc"] = path_for_report(cc_path, base_dir)
            artifacts["tflite_micro_h"] = path_for_report(h_path, base_dir)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_training_history(history, plots_dir / "training_curves.png")
    val_probabilities = data_utils.predict_probabilities(
        model, val, args.img_size, args.batch_size
    )
    val_labels = np.asarray([sample.label for sample in val], dtype=np.int32)
    plot_threshold_analysis(
        val_labels,
        val_probabilities,
        threshold,
        plots_dir / "threshold_analysis.png",
    )
    plot_binary_confusion_matrix(
        metrics["test"], plots_dir / "confusion_matrix_test.png", "LiteSpatialFusionCNN - Keras"
    )
    int8_metrics = metrics.get("test_int8", metrics["test"])
    plot_binary_confusion_matrix(
        int8_metrics, plots_dir / "confusion_matrix_int8.png", "LiteSpatialFusionCNN - INT8"
    )

    save_summary(output_dir / "metrics.csv", metrics)
    write_report(
        output_dir / "report.md",
        args,
        model,
        train,
        val,
        test,
        threshold,
        threshold_mode,
        metrics,
        artifacts,
        tflite_info,
    )
    print("Done:", output_dir)
    for name, artifact in artifacts.items():
        print(f"  {name}: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
