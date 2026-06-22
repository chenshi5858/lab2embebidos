"""Distill a large pretrained teacher into the binary multiscale SkipPoolCNN.

The teacher is used only during training. The exported model is the same
quantization-friendly SkipPool architecture defined in
skippoolcnn_tf_presence_binary_version.py.
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

import skippoolcnn_tf_presence_binary_version as skippool


MODEL_NAME = "skippool_presence_large_teacher_distilled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill pretrained MobileNetV2 into binary SkipPoolCNN."
    )
    parser.add_argument("--base_dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs") / MODEL_NAME,
    )
    parser.add_argument("--source", choices=("esp", "celular", "all"), default="esp")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--student_variant", choices=("multiscale", "compact"), default="multiscale")
    parser.add_argument("--student_epochs", type=int, default=80)
    parser.add_argument("--student_learning_rate", type=float, default=7e-4)
    parser.add_argument("--teacher_head_epochs", type=int, default=20)
    parser.add_argument("--teacher_finetune_epochs", type=int, default=15)
    parser.add_argument("--teacher_head_learning_rate", type=float, default=1e-3)
    parser.add_argument("--teacher_finetune_learning_rate", type=float, default=2e-5)
    parser.add_argument("--teacher_finetune_layers", type=int, default=40)
    parser.add_argument("--teacher_weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--alpha", type=float, default=0.75, help="Hard-label loss weight.")
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--split_mode", choices=("stratified", "group"), default="stratified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold_metric",
        choices=("balanced_accuracy", "f1", "f2", "accuracy", "mcc"),
        default="f1",
    )
    parser.add_argument("--min_recall", type=float, default=None)
    parser.add_argument("--positive_weight_multiplier", type=float, default=1.0)
    parser.add_argument("--representative_samples", type=int, default=250)
    parser.add_argument("--no_int8", action="store_true")
    parser.add_argument("--no_c_array", action="store_true")
    return parser.parse_args()


def compile_binary_model(model: tf.keras.Model, learning_rate: float) -> None:
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


def callbacks(patience: int = 8) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", patience=4, factor=0.5, min_lr=1e-6, verbose=1
        ),
    ]


def build_large_teacher(
    img_size: int,
    weights: str,
    learning_rate: float,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input((img_size, img_size, 1), name="grayscale_96x96")
    x = tf.keras.layers.RandomFlip("horizontal", name="teacher_flip")(inputs)
    x = tf.keras.layers.RandomTranslation(0.06, 0.06, fill_mode="reflect", name="teacher_translation")(x)
    x = tf.keras.layers.RandomRotation(0.05, fill_mode="reflect", name="teacher_rotation")(x)
    x = tf.keras.layers.RandomContrast(0.15, name="teacher_contrast")(x)
    x = tf.keras.layers.Concatenate(name="grayscale_to_rgb")([x, x, x])
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=None if weights == "none" else "imagenet",
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    average = tf.keras.layers.GlobalAveragePooling2D(name="teacher_global_average")(x)
    maximum = tf.keras.layers.GlobalMaxPooling2D(name="teacher_global_max")(x)
    x = tf.keras.layers.Concatenate(name="teacher_pool_fusion")([average, maximum])
    x = tf.keras.layers.Dense(
        192,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(2e-4),
        name="teacher_head",
    )(x)
    x = tf.keras.layers.Dropout(0.35, name="teacher_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="presence")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_identifier_presence_teacher")
    compile_binary_model(model, learning_rate)
    return model, backbone


def unfreeze_teacher_backbone(
    teacher: tf.keras.Model,
    backbone: tf.keras.Model,
    trainable_layers: int,
    learning_rate: float,
) -> None:
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - trainable_layers)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
    compile_binary_model(teacher, learning_rate)


def predict_probabilities(
    model: tf.keras.Model,
    samples: list[skippool.Sample],
    img_size: int,
    batch_size: int,
) -> np.ndarray:
    dataset = skippool.make_dataset(samples, img_size, batch_size, False, 0)
    return model.predict(dataset, verbose=0).reshape(-1)


def make_distillation_dataset(
    samples: list[skippool.Sample],
    teacher_probabilities: np.ndarray,
    img_size: int,
    batch_size: int,
    seed: int,
) -> tf.data.Dataset:
    paths = [str(sample.image_path) for sample in samples]
    labels = np.asarray([sample.label for sample in samples], dtype=np.float32)
    soft_targets = np.asarray(teacher_probabilities, dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, soft_targets))
    dataset = dataset.shuffle(len(samples), seed=seed, reshuffle_each_iteration=True)

    def load_example(path, label, soft_target):
        image = skippool.decode_resize_image(path, img_size)
        image = skippool.augment_image(image, img_size)
        return image, (
            tf.reshape(tf.cast(label, tf.float32), [1]),
            tf.reshape(tf.cast(soft_target, tf.float32), [1]),
        )

    return (
        dataset.map(load_example, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


class OfflineBinaryDistiller(tf.keras.Model):
    def __init__(
        self,
        student: tf.keras.Model,
        alpha: float,
        temperature: float,
        class_weights: dict[int, float],
    ):
        super().__init__(name="offline_binary_distiller")
        self.student = student
        self.alpha = alpha
        self.temperature = temperature
        self.negative_weight = float(class_weights[0])
        self.positive_weight = float(class_weights[1])
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.hard_loss_tracker = tf.keras.metrics.Mean(name="hard_loss")
        self.distillation_loss_tracker = tf.keras.metrics.Mean(name="distillation_loss")
        self.accuracy_metric = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        self.precision_metric = tf.keras.metrics.Precision(name="precision")
        self.recall_metric = tf.keras.metrics.Recall(name="recall")
        self.auc_metric = tf.keras.metrics.AUC(name="auc")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.hard_loss_tracker,
            self.distillation_loss_tracker,
            self.accuracy_metric,
            self.precision_metric,
            self.recall_metric,
            self.auc_metric,
        ]

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def _weights(self, labels):
        return tf.where(labels >= 0.5, self.positive_weight, self.negative_weight)

    @staticmethod
    def _weighted_mean(values, weights):
        values = tf.reshape(values, [-1])
        weights = tf.reshape(weights, [-1])
        return tf.reduce_sum(values * weights) / tf.maximum(tf.reduce_sum(weights), 1e-7)

    def _soften(self, probabilities):
        probabilities = tf.clip_by_value(probabilities, 1e-5, 1.0 - 1e-5)
        logits = tf.math.log(probabilities) - tf.math.log1p(-probabilities)
        return tf.math.sigmoid(logits / self.temperature)

    def train_step(self, data):
        images, (labels, teacher_probabilities) = data
        sample_weights = self._weights(labels)
        with tf.GradientTape() as tape:
            student_probabilities = self.student(images, training=True)
            hard_values = tf.keras.losses.binary_crossentropy(labels, student_probabilities)
            hard_loss = self._weighted_mean(hard_values, sample_weights)
            soft_teacher = self._soften(teacher_probabilities)
            soft_student = self._soften(student_probabilities)
            soft_values = tf.keras.losses.binary_crossentropy(soft_teacher, soft_student)
            distillation_loss = self._weighted_mean(soft_values, sample_weights)
            regularization = tf.add_n(self.student.losses or [tf.constant(0.0)])
            loss = (
                self.alpha * hard_loss
                + (1.0 - self.alpha) * (self.temperature**2) * distillation_loss
                + regularization
            )
        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        self._update_metrics(labels, student_probabilities, loss, hard_loss, distillation_loss)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        images, labels = data
        probabilities = self.student(images, training=False)
        sample_weights = self._weights(labels)
        hard_values = tf.keras.losses.binary_crossentropy(labels, probabilities)
        hard_loss = self._weighted_mean(hard_values, sample_weights)
        self._update_metrics(labels, probabilities, hard_loss, hard_loss, 0.0)
        return {metric.name: metric.result() for metric in self.metrics}

    def _update_metrics(self, labels, probabilities, loss, hard_loss, distillation_loss):
        self.loss_tracker.update_state(loss)
        self.hard_loss_tracker.update_state(hard_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.accuracy_metric.update_state(labels, probabilities)
        self.precision_metric.update_state(labels, probabilities)
        self.recall_metric.update_state(labels, probabilities)
        self.auc_metric.update_state(labels, probabilities)


def evaluate_with_validation_threshold(
    model: tf.keras.Model,
    val: list[skippool.Sample],
    test: list[skippool.Sample],
    args: argparse.Namespace,
) -> tuple[float, dict[str, float | int], dict[str, float | int]]:
    threshold, val_metrics = skippool.find_best_threshold(
        model,
        val,
        args.img_size,
        args.batch_size,
        args.threshold_metric,
        args.min_recall,
    )
    test_metrics = skippool.evaluate_predictions(
        model, test, args.img_size, args.batch_size, threshold
    )
    return threshold, val_metrics, test_metrics


def save_metrics(path: Path, metrics: dict[str, dict[str, float | int]]) -> None:
    fields = list(next(iter(metrics.values())))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *fields])
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"model": name, **values})


def write_c_array(model_bytes: bytes, output_dir: Path, threshold: float) -> tuple[Path, Path]:
    header = output_dir / f"{MODEL_NAME}_model_data.h"
    source = output_dir / f"{MODEL_NAME}_model_data.cc"
    variable = "g_skippool_large_teacher_distilled_model"
    header.write_text(
        "\n".join(
            [
                "#pragma once",
                "",
                f"extern const unsigned char {variable}[];",
                f"extern const int {variable}_len;",
                f"constexpr float kSkippoolLargeTeacherThreshold = {threshold:.8f}f;",
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
                f"alignas(16) const unsigned char {variable}[] = {{",
                ",\n".join(rows),
                "};",
                f"const int {variable}_len = {len(model_bytes)};",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return source, header


def write_report(
    path: Path,
    args: argparse.Namespace,
    teacher: tf.keras.Model,
    student: tf.keras.Model,
    thresholds: dict[str, float],
    metrics: dict[str, dict[str, float | int]],
    artifacts: dict[str, str],
    sizes_mb: dict[str, float],
    tflite_info: dict[str, object],
) -> None:
    lines = [
        "# Large Teacher -> SkipPool Knowledge Distillation Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        "",
        f"- Teacher: `MobileNetV2 ({args.teacher_weights})`",
        f"- Teacher parameters: `{teacher.count_params()}`",
        f"- Student: `SkipPoolCNN binary ({args.student_variant})`",
        f"- Student parameters: `{student.count_params()}`",
        f"- Compression by parameter count: `{(1.0 - student.count_params() / teacher.count_params()) * 100:.2f}%`",
        f"- Alpha: `{args.alpha}`",
        f"- Temperature: `{args.temperature}`",
        f"- Teacher head/fine-tune epochs: `{args.teacher_head_epochs}` / `{args.teacher_finetune_epochs}`",
        "",
        "## Thresholds",
        "",
    ]
    lines.extend(f"- {name}: `{value:.4f}`" for name, value in thresholds.items())
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| model | accuracy | precision | recall | specificity | balanced_accuracy | f1 | f2 | mcc | tn | fp | fn | tp |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, values in metrics.items():
        lines.append(
            "| {name} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {specificity:.4f} | "
            "{balanced_accuracy:.4f} | {f1:.4f} | {f2:.4f} | {mcc:.4f} | {tn} | {fp} | {fn} | {tp} |".format(
                name=name, **values
            )
        )
    lines.extend(["", "## Model Sizes", "", "| artifact | size_mb |", "| --- | ---: |"])
    lines.extend(f"| {name} | {size:.4f} |" for name, size in sizes_mb.items())
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- {name}: `{artifact}`" for name, artifact in artifacts.items())
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![Teacher training](plots/teacher_training_curves.png)",
            "",
            "![Student distillation](plots/student_training_curves.png)",
            "",
            "![Threshold](plots/threshold_analysis.png)",
            "",
            "![INT8 confusion matrix](plots/confusion_matrix_int8.png)",
            "",
            "## TFLite",
            "",
            "```json",
            json.dumps(tflite_info, indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1.")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be positive.")

    base_dir = args.base_dir.resolve()
    output_dir = (base_dir / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass

    samples, stats = skippool.scan_dataset(base_dir, args.source)
    if not samples:
        raise SystemExit("No samples found.")
    train, val, test = skippool.split_samples(args, samples)
    skippool.write_dataset_summary(output_dir / "dataset_summary.csv", stats, samples)
    skippool.write_manifest(output_dir / "train_manifest.csv", train, base_dir)
    skippool.write_manifest(output_dir / "val_manifest.csv", val, base_dir)
    skippool.write_manifest(output_dir / "test_manifest.csv", test, base_dir)

    class_weights = skippool.compute_class_weight(train, args.positive_weight_multiplier)
    train_ds = skippool.make_dataset(train, args.img_size, args.batch_size, True, args.seed)
    val_ds = skippool.make_dataset(val, args.img_size, args.batch_size, False, args.seed)

    print("Training large MobileNetV2 teacher...")
    teacher, backbone = build_large_teacher(
        args.img_size, args.teacher_weights, args.teacher_head_learning_rate
    )
    teacher_head_history = teacher.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.teacher_head_epochs,
        class_weight=class_weights,
        callbacks=callbacks(),
        verbose=2,
    )
    skippool.save_history(output_dir / "teacher_head_history.csv", teacher_head_history)

    teacher_finetune_history = None
    if args.teacher_finetune_epochs > 0:
        print("Fine-tuning the top of the teacher backbone...")
        unfreeze_teacher_backbone(
            teacher,
            backbone,
            args.teacher_finetune_layers,
            args.teacher_finetune_learning_rate,
        )
        teacher_finetune_history = teacher.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.teacher_finetune_epochs,
            class_weight=class_weights,
            callbacks=callbacks(patience=6),
            verbose=2,
        )
        skippool.save_history(
            output_dir / "teacher_finetune_history.csv", teacher_finetune_history
        )
    teacher_path = output_dir / "large_mobilenetv2_teacher.keras"
    teacher.save(teacher_path)

    print("Caching teacher probabilities...")
    teacher_train_probabilities = predict_probabilities(
        teacher, train, args.img_size, args.batch_size
    )

    print("Training non-distilled SkipPool baseline...")
    tf.keras.utils.set_random_seed(args.seed + 1)
    baseline = skippool.build_skippool_presence_model(
        args.img_size, args.student_learning_rate, args.student_variant
    )
    initial_weights = baseline.get_weights()
    baseline_history = baseline.fit(
        skippool.make_dataset(train, args.img_size, args.batch_size, True, args.seed),
        validation_data=val_ds,
        epochs=args.student_epochs,
        class_weight=class_weights,
        callbacks=callbacks(patience=14),
        verbose=2,
    )
    baseline_path = output_dir / "skippool_baseline.keras"
    baseline.save(baseline_path)
    skippool.save_history(output_dir / "baseline_history.csv", baseline_history)

    print("Training distilled SkipPool student...")
    student = skippool.build_skippool_presence_model(
        args.img_size, args.student_learning_rate, args.student_variant
    )
    student.set_weights(initial_weights)
    distiller = OfflineBinaryDistiller(
        student, args.alpha, args.temperature, class_weights
    )
    distiller.compile(optimizer=tf.keras.optimizers.Adam(args.student_learning_rate))
    distillation_history = distiller.fit(
        make_distillation_dataset(
            train,
            teacher_train_probabilities,
            args.img_size,
            args.batch_size,
            args.seed,
        ),
        validation_data=val_ds,
        epochs=args.student_epochs,
        callbacks=callbacks(patience=14),
        verbose=2,
    )
    student_path = output_dir / "distilled_skippool_student.keras"
    student.save(student_path)
    skippool.save_history(output_dir / "distillation_history.csv", distillation_history)

    teacher_threshold, _teacher_val, teacher_test = evaluate_with_validation_threshold(
        teacher, val, test, args
    )
    baseline_threshold, _baseline_val, baseline_test = evaluate_with_validation_threshold(
        baseline, val, test, args
    )
    student_threshold, student_val, student_test = evaluate_with_validation_threshold(
        student, val, test, args
    )
    metrics = {
        "teacher_test": teacher_test,
        "baseline_test": baseline_test,
        "student_val": student_val,
        "student_test": student_test,
    }
    thresholds = {
        "teacher": teacher_threshold,
        "baseline": baseline_threshold,
        "student": student_threshold,
    }

    artifacts = {
        "teacher_keras": skippool.path_for_report(teacher_path, base_dir),
        "baseline_keras": skippool.path_for_report(baseline_path, base_dir),
        "student_keras": skippool.path_for_report(student_path, base_dir),
        "teacher_head_history": skippool.path_for_report(
            output_dir / "teacher_head_history.csv", base_dir
        ),
        "baseline_history": skippool.path_for_report(
            output_dir / "baseline_history.csv", base_dir
        ),
        "distillation_history": skippool.path_for_report(
            output_dir / "distillation_history.csv", base_dir
        ),
    }
    if teacher_finetune_history is not None:
        artifacts["teacher_finetune_history"] = skippool.path_for_report(
            output_dir / "teacher_finetune_history.csv", base_dir
        )

    float_bytes = skippool.convert_float_tflite(student, args.img_size)
    float_path = output_dir / f"{MODEL_NAME}_float32.tflite"
    float_path.write_bytes(float_bytes)
    artifacts["tflite_float32"] = skippool.path_for_report(float_path, base_dir)
    tflite_info: dict[str, object] = {"float32": skippool.inspect_tflite(float_bytes)}

    int8_path = None
    if not args.no_int8:
        shuffled_train = train[:]
        random.Random(args.seed).shuffle(shuffled_train)
        int8_bytes = skippool.convert_int8_tflite(
            student, shuffled_train, args.img_size, args.representative_samples
        )
        int8_path = output_dir / f"{MODEL_NAME}_int8.tflite"
        int8_path.write_bytes(int8_bytes)
        artifacts["tflite_int8"] = skippool.path_for_report(int8_path, base_dir)
        tflite_info["int8"] = skippool.inspect_tflite(int8_bytes)
        test_labels = np.asarray([sample.label for sample in test], dtype=np.int32)
        int8_probabilities = skippool.predict_tflite_probabilities(
            int8_bytes, test, args.img_size
        )
        metrics["student_test_int8"] = skippool.metrics_from_probabilities(
            test_labels, int8_probabilities, student_threshold
        )
        if not args.no_c_array:
            cc_path, h_path = write_c_array(int8_bytes, output_dir, student_threshold)
            artifacts["tflite_micro_cc"] = skippool.path_for_report(cc_path, base_dir)
            artifacts["tflite_micro_h"] = skippool.path_for_report(h_path, base_dir)

    teacher_plot_history = teacher_finetune_history or teacher_head_history
    skippool.plot_training_history(
        teacher_plot_history, plots_dir / "teacher_training_curves.png"
    )
    skippool.plot_training_history(
        distillation_history, plots_dir / "student_training_curves.png"
    )
    val_labels = np.asarray([sample.label for sample in val], dtype=np.int32)
    student_val_probabilities = predict_probabilities(
        student, val, args.img_size, args.batch_size
    )
    skippool.plot_threshold_analysis(
        val_labels,
        student_val_probabilities,
        student_threshold,
        plots_dir / "threshold_analysis.png",
    )
    skippool.plot_binary_confusion_matrix(
        metrics.get("student_test_int8", student_test),
        plots_dir / "confusion_matrix_int8.png",
        "Large teacher distilled SkipPool - INT8",
    )

    save_metrics(output_dir / "metrics.csv", metrics)
    sizes_mb = {
        "teacher_keras": teacher_path.stat().st_size / (1024**2),
        "baseline_keras": baseline_path.stat().st_size / (1024**2),
        "student_keras": student_path.stat().st_size / (1024**2),
        "student_tflite_float32": float_path.stat().st_size / (1024**2),
    }
    if int8_path is not None:
        sizes_mb["student_tflite_int8"] = int8_path.stat().st_size / (1024**2)
    write_report(
        output_dir / "report.md",
        args,
        teacher,
        student,
        thresholds,
        metrics,
        artifacts,
        sizes_mb,
        tflite_info,
    )
    print("Done:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
