"""Compress the binary SkipPoolCNN with SVD low-rank factorization.

The full-rank model is trained (or loaded), its large ``spatial_head`` Dense
kernel is decomposed as W ~= A @ B, and the resulting low-rank model is then
fine-tuned. Only the compressed student is exported for the ESP-CAM.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

import skippoolcnn_tf_presence_binary_version as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply SVD low-rank factorization to the binary SkipPoolCNN."
    )
    parser.add_argument("--base_dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs") / "skippool_presence_low_rank_r8",
    )
    parser.add_argument("--source", choices=("esp", "celular", "all"), default="esp")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--baseline_epochs", type=int, default=80)
    parser.add_argument("--finetune_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--finetune_learning_rate", type=float, default=2e-4)
    parser.add_argument("--positive_weight_multiplier", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_mode", choices=("stratified", "group"), default="stratified"
    )
    parser.add_argument(
        "--baseline_model",
        type=Path,
        default=None,
        help="Optional trained multiscale .keras model. If omitted, train it first.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Presence decision threshold (default: 0.55). Use a custom value to override it.",
    )
    parser.add_argument(
        "--threshold_metric",
        choices=("balanced_accuracy", "f1", "f2", "accuracy", "mcc"),
        default="f1",
    )
    parser.add_argument("--min_recall", type=float, default=None)
    parser.add_argument("--representative_samples", type=int, default=200)
    parser.add_argument("--scan_only", action="store_true")
    parser.add_argument("--no_int8", action="store_true")
    parser.add_argument("--no_c_array", action="store_true")
    return parser.parse_args()


def compile_binary_model(model: tf.keras.Model, learning_rate: float) -> None:
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


def build_low_rank_model(
    img_size: int,
    rank: int,
    learning_rate: float,
) -> tf.keras.Model:
    """Build the multiscale SkipPoolCNN with a factored spatial Dense head."""
    if not 1 <= rank <= 24:
        raise ValueError("--rank must be between 1 and 24 for the 24-unit spatial head.")

    inputs = tf.keras.Input(shape=(img_size, img_size, 1), name="grayscale_96x96")
    scaled = tf.keras.layers.Rescaling(1.0 / 255.0, name="scale_to_0_1")(inputs)
    skip_max = tf.keras.layers.MaxPooling2D(
        pool_size=8, strides=8, padding="valid", name="skip_max_pool"
    )(scaled)
    skip_avg = tf.keras.layers.AveragePooling2D(
        pool_size=8, strides=8, padding="valid", name="skip_average_pool"
    )(scaled)

    x = scaled
    for index, (filters, stride) in enumerate(((6, 2), (12, 2), (18, 2), (24, 1))):
        x = tf.keras.layers.Conv2D(
            filters,
            3,
            strides=stride,
            padding="same",
            use_bias=False,
            name=f"conv_{index}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{index}")(x)
        x = tf.keras.layers.ReLU(name=f"relu_{index}")(x)

    x = tf.keras.layers.Concatenate(axis=-1, name="concat_skip")([x, skip_max, skip_avg])
    x = tf.keras.layers.Flatten(name="flatten_spatial_features")(x)
    x = tf.keras.layers.Dense(
        rank,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="spatial_head_factor_a",
    )(x)
    x = tf.keras.layers.Dense(
        24,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="spatial_head_factor_b",
    )(x)
    x = tf.keras.layers.Dropout(0.20, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="presence")(x)
    model = tf.keras.Model(inputs, outputs, name=f"skippool_presence_low_rank_r{rank}")
    compile_binary_model(model, learning_rate)
    return model


def initialize_from_svd(
    full_model: tf.keras.Model,
    low_rank_model: tf.keras.Model,
    rank: int,
) -> float:
    """Transfer shared weights and initialize A/B from the truncated SVD."""
    skipped = {"spatial_head_factor_a", "spatial_head_factor_b"}
    full_layers = {layer.name: layer for layer in full_model.layers}
    for layer in low_rank_model.layers:
        if layer.name in skipped or layer.name not in full_layers:
            continue
        source_weights = full_layers[layer.name].get_weights()
        if source_weights:
            layer.set_weights(source_weights)

    kernel, bias = full_model.get_layer("spatial_head").get_weights()
    u, singular_values, vt = np.linalg.svd(kernel, full_matrices=False)
    effective_rank = min(rank, len(singular_values))
    root_s = np.sqrt(singular_values[:effective_rank])
    factor_a = u[:, :effective_rank] * root_s[np.newaxis, :]
    factor_b = root_s[:, np.newaxis] * vt[:effective_rank, :]
    low_rank_model.get_layer("spatial_head_factor_a").set_weights(
        [factor_a.astype(np.float32)]
    )
    low_rank_model.get_layer("spatial_head_factor_b").set_weights(
        [factor_b.astype(np.float32), bias.astype(np.float32)]
    )
    reconstruction = factor_a @ factor_b
    return float(np.linalg.norm(kernel - reconstruction) / np.linalg.norm(kernel))


def make_callbacks(checkpoint: Path, patience: int) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_auc", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=max(3, patience // 3),
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_or_load_baseline(
    args: argparse.Namespace,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weight: dict[int, float],
    output_dir: Path,
) -> tuple[tf.keras.Model, Path, tf.keras.callbacks.History | None]:
    output_path = output_dir / "full_rank_skippool.keras"
    if args.baseline_model is not None:
        source_path = args.baseline_model.resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Baseline model not found: {source_path}")
        model = tf.keras.models.load_model(source_path)
        if "spatial_head" not in {layer.name for layer in model.layers}:
            raise ValueError("The baseline must be the multiscale model with spatial_head.")
        shutil.copy2(source_path, output_path)
        return model, output_path, None

    model = base.build_skippool_presence_model(
        args.img_size, args.learning_rate, model_variant="multiscale"
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.baseline_epochs,
        class_weight=class_weight,
        callbacks=make_callbacks(output_path, patience=14),
        verbose=2,
    )
    if output_path.exists():
        model = tf.keras.models.load_model(output_path)
    else:
        model.save(output_path)
    return model, output_path, history


def select_threshold(
    model: tf.keras.Model,
    samples: list[base.Sample],
    args: argparse.Namespace,
) -> tuple[float, dict[str, float | int], str]:
    if args.threshold is not None:
        threshold = float(args.threshold)
        metrics = base.evaluate_predictions(
            model, samples, args.img_size, args.batch_size, threshold
        )
        return threshold, metrics, "fixed"
    threshold, metrics = base.find_best_threshold(
        model,
        samples,
        args.img_size,
        args.batch_size,
        args.threshold_metric,
        args.min_recall,
    )
    if args.min_recall is None:
        mode = f"best validation {args.threshold_metric}"
    else:
        mode = f"validation recall >= {args.min_recall:g}, max specificity"
    return threshold, metrics, mode


def write_report(
    path: Path,
    args: argparse.Namespace,
    full_model: tf.keras.Model,
    low_rank_model: tf.keras.Model,
    reconstruction_error: float,
    initial_prediction_mae: float,
    thresholds: dict[str, float],
    threshold_mode: str,
    metrics: dict[str, dict[str, float | int]],
    sizes: dict[str, float],
    artifacts: dict[str, str],
    tflite_info: dict[str, object],
) -> None:
    full_params = full_model.count_params()
    low_rank_params = low_rank_model.count_params()
    reduction = 100.0 * (1.0 - low_rank_params / full_params)
    lines = [
        "# SkipPoolCNN Low-Rank Factorization Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        "",
        "- Compression: truncated SVD of the 24-unit `spatial_head` Dense kernel.",
        f"- Rank: `{args.rank}` of maximum `24`.",
        f"- Fine-tuning learning rate: `{args.finetune_learning_rate}`.",
        f"- Full-rank parameters: `{full_params}`.",
        f"- Low-rank parameters: `{low_rank_params}`.",
        f"- Parameter reduction: `{reduction:.2f}%`.",
        f"- SVD relative reconstruction error: `{reconstruction_error:.6f}`.",
        f"- Initial output MAE after factorization: `{initial_prediction_mae:.6f}`.",
        "",
        "The factorization replaces `W` with `A @ B`, where ",
        f"`A` has shape `(flattened_features, {args.rank})` and `B` has shape `({args.rank}, 24)`.",
        "",
        "## Thresholds",
        "",
        f"- Full rank: `{thresholds['full_rank']:.4f}`.",
        f"- Low rank: `{thresholds['low_rank']:.4f}` ({threshold_mode}).",
        "",
        "## Metrics",
        "",
        "| model | accuracy | precision | recall | specificity | balanced_accuracy | f1 | f2 | mcc | tn | fp | fn | tp |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in metrics.items():
        lines.append(
            "| {name} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | "
            "{specificity:.4f} | {balanced_accuracy:.4f} | {f1:.4f} | {f2:.4f} | "
            "{mcc:.4f} | {tn} | {fp} | {fn} | {tp} |".format(name=name, **values)
        )
    lines.extend(["", "## Model Sizes", "", "| artifact | size_mb |", "| --- | ---: |"])
    for name, size in sizes.items():
        lines.append(f"| {name} | {size:.4f} |")
    lines.extend(["", "## Artifacts", ""])
    for name, artifact in artifacts.items():
        lines.append(f"- {name}: `{artifact}`")
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![Low-rank fine-tuning](plots/low_rank_training_curves.png)",
            "",
            "![Threshold analysis](plots/threshold_analysis.png)",
            "",
            "![INT8 confusion matrix](plots/confusion_matrix_int8.png)",
            "",
            "## TFLite",
            "",
            "```json",
            json.dumps(tflite_info, indent=2),
            "```",
            "",
            "## Deployment",
            "",
            "Only the low-rank INT8 model is needed on the ESP-CAM. SVD and the full-rank model are training-time tools.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    base.require_runtime()
    base_dir = args.base_dir.resolve()
    output_dir = (
        (base_dir / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    samples, stats = base.scan_dataset(base_dir, args.source)
    if not samples:
        raise SystemExit("No samples found. Check --base_dir and --source.")
    base.print_scan(samples, stats)
    if args.scan_only:
        return 0

    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = base.split_samples(args, samples)
    print(f"train: {base.format_counts(train)}")
    print(f"val:   {base.format_counts(val)}")
    print(f"test:  {base.format_counts(test)}")
    base.write_dataset_summary(output_dir / "dataset_summary.csv", stats, samples)
    base.write_manifest(output_dir / "train_manifest.csv", train, base_dir)
    base.write_manifest(output_dir / "val_manifest.csv", val, base_dir)
    base.write_manifest(output_dir / "test_manifest.csv", test, base_dir)

    train_ds = base.make_dataset(
        train, args.img_size, args.batch_size, training=True, seed=args.seed
    )
    val_ds = base.make_dataset(
        val, args.img_size, args.batch_size, training=False, seed=args.seed
    )
    class_weight = base.compute_class_weight(train, args.positive_weight_multiplier)

    print("Training/loading full-rank SkipPoolCNN...")
    full_model, full_path, full_history = train_or_load_baseline(
        args, train_ds, val_ds, class_weight, output_dir
    )
    if full_history is not None:
        base.save_history(output_dir / "full_rank_history.csv", full_history)

    print(f"Applying rank-{args.rank} truncated SVD...")
    low_rank_model = build_low_rank_model(
        args.img_size, args.rank, args.finetune_learning_rate
    )
    reconstruction_error = initialize_from_svd(full_model, low_rank_model, args.rank)
    full_val_prob = base.predict_probabilities(
        full_model, val, args.img_size, args.batch_size
    )
    initial_low_rank_prob = base.predict_probabilities(
        low_rank_model, val, args.img_size, args.batch_size
    )
    initial_prediction_mae = float(np.mean(np.abs(full_val_prob - initial_low_rank_prob)))

    low_rank_path = output_dir / f"skippool_presence_low_rank_r{args.rank}.keras"
    print("Fine-tuning low-rank SkipPoolCNN...")
    low_rank_history = low_rank_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        class_weight=class_weight,
        callbacks=make_callbacks(low_rank_path, patience=12),
        verbose=2,
    )
    base.save_history(output_dir / "low_rank_history.csv", low_rank_history)
    if low_rank_path.exists():
        low_rank_model = tf.keras.models.load_model(low_rank_path)
    else:
        low_rank_model.save(low_rank_path)

    full_threshold, full_val_metrics, _ = select_threshold(full_model, val, args)
    low_threshold, low_val_metrics, threshold_mode = select_threshold(
        low_rank_model, val, args
    )
    metrics = {
        "full_rank_val": full_val_metrics,
        "full_rank_test": base.evaluate_predictions(
            full_model, test, args.img_size, args.batch_size, full_threshold
        ),
        "low_rank_val": low_val_metrics,
        "low_rank_test": base.evaluate_predictions(
            low_rank_model, test, args.img_size, args.batch_size, low_threshold
        ),
    }

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    training_plot = plots_dir / "low_rank_training_curves.png"
    threshold_plot = plots_dir / "threshold_analysis.png"
    confusion_plot = plots_dir / "confusion_matrix_int8.png"
    base.plot_training_history(low_rank_history, training_plot)
    low_val_prob = base.predict_probabilities(
        low_rank_model, val, args.img_size, args.batch_size
    )
    val_labels = np.asarray([sample.label for sample in val], dtype=np.int32)
    base.plot_threshold_analysis(val_labels, low_val_prob, low_threshold, threshold_plot)

    artifacts = {
        "full_rank_keras": base.path_for_report(full_path, base_dir),
        "low_rank_keras": base.path_for_report(low_rank_path, base_dir),
        "low_rank_history": base.path_for_report(output_dir / "low_rank_history.csv", base_dir),
        "dataset_summary": base.path_for_report(output_dir / "dataset_summary.csv", base_dir),
    }
    float_tflite = base.convert_float_tflite(low_rank_model, args.img_size)
    float_path = output_dir / f"skippool_presence_low_rank_r{args.rank}_float32.tflite"
    float_path.write_bytes(float_tflite)
    artifacts["tflite_float32"] = base.path_for_report(float_path, base_dir)
    tflite_info: dict[str, object] = {"float32": base.inspect_tflite(float_tflite)}

    int8_path: Path | None = None
    if not args.no_int8:
        shuffled_train = train[:]
        random.Random(args.seed).shuffle(shuffled_train)
        int8_tflite = base.convert_int8_tflite(
            low_rank_model,
            shuffled_train,
            args.img_size,
            args.representative_samples,
        )
        int8_path = output_dir / f"skippool_presence_low_rank_r{args.rank}_int8.tflite"
        int8_path.write_bytes(int8_tflite)
        artifacts["tflite_int8"] = base.path_for_report(int8_path, base_dir)
        tflite_info["int8"] = base.inspect_tflite(int8_tflite)
        int8_prob = base.predict_tflite_probabilities(int8_tflite, test, args.img_size)
        test_labels = np.asarray([sample.label for sample in test], dtype=np.int32)
        metrics["low_rank_test_int8"] = base.metrics_from_probabilities(
            test_labels, int8_prob, low_threshold
        )
        base.plot_binary_confusion_matrix(
            metrics["low_rank_test_int8"], confusion_plot, "Low-rank INT8 - Prueba"
        )
        if not args.no_c_array:
            cc_path, h_path = base.write_c_array(
                int8_tflite, output_dir, "g_skippool_presence_low_rank_model"
            )
            artifacts["tflite_micro_cc"] = base.path_for_report(cc_path, base_dir)
            artifacts["tflite_micro_h"] = base.path_for_report(h_path, base_dir)
    else:
        base.plot_binary_confusion_matrix(
            metrics["low_rank_test"], confusion_plot, "Low-rank float32 - Prueba"
        )

    artifacts["plot_training"] = base.path_for_report(training_plot, base_dir)
    artifacts["plot_threshold"] = base.path_for_report(threshold_plot, base_dir)
    artifacts["plot_confusion"] = base.path_for_report(confusion_plot, base_dir)
    sizes = {
        "full_rank_keras": full_path.stat().st_size / (1024**2),
        "low_rank_keras": low_rank_path.stat().st_size / (1024**2),
        "low_rank_tflite_float32": len(float_tflite) / (1024**2),
    }
    if int8_path is not None:
        sizes["low_rank_tflite_int8"] = int8_path.stat().st_size / (1024**2)

    report_path = output_dir / "report.md"
    write_report(
        report_path,
        args,
        full_model,
        low_rank_model,
        reconstruction_error,
        initial_prediction_mae,
        {"full_rank": full_threshold, "low_rank": low_threshold},
        threshold_mode,
        metrics,
        sizes,
        artifacts,
        tflite_info,
    )
    print(f"Done: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
