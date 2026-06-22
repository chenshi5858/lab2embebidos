"""Knowledge distillation for a compact binary SkipPool presence detector.

By default, the script trains a fresh multiscale SkipPool teacher on the same
split used by the 6,713-parameter compact student. One or more existing Keras
teachers can also be supplied. Training combines hard binary labels with
temperature-softened teacher probabilities, then exports and evaluates a
full-INT8 TFLite model.
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
import skippoolcnn_tf_presence_binary_version as skippool


MODEL_NAME = "skippool_presence_distilled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill an ensemble into compact SkipPoolCNN.")
    parser.add_argument("--base_dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs") / MODEL_NAME,
    )
    parser.add_argument("--source", choices=("esp", "celular", "all"), default="esp")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=7e-4)
    parser.add_argument("--alpha", type=float, default=0.90, help="Weight of hard-label loss.")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=20,
        help="Hard-label fine-tuning epochs after distillation; use 0 to disable.",
    )
    parser.add_argument("--fine_tune_learning_rate", type=float, default=1e-4)
    parser.add_argument("--teacher_epochs", type=int, default=70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--split_mode", choices=("stratified", "group"), default="stratified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold_metric",
        choices=("balanced_accuracy", "f1", "f2", "accuracy", "mcc"),
        default="f1",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min_recall", type=float, default=None)
    parser.add_argument("--positive_weight_multiplier", type=float, default=1.0)
    parser.add_argument("--representative_samples", type=int, default=250)
    parser.add_argument(
        "--teacher_model",
        type=Path,
        action="append",
        default=None,
        help="Repeat for each .keras teacher; otherwise trains a fresh multiscale teacher.",
    )
    parser.add_argument("--no_int8", action="store_true")
    parser.add_argument("--no_c_array", action="store_true")
    return parser.parse_args()


class EnsembleTeacher(tf.keras.Model):
    """Average teacher logits, avoiding overconfident probability averaging."""

    def __init__(self, teachers: list[tf.keras.Model]):
        super().__init__(name="ensemble_teacher")
        self.teachers = teachers
        for teacher in self.teachers:
            teacher.trainable = False

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        logits = []
        for teacher in self.teachers:
            probability = tf.clip_by_value(teacher(inputs, training=False), 1e-5, 1.0 - 1e-5)
            logits.append(tf.math.log(probability) - tf.math.log1p(-probability))
        return tf.math.sigmoid(tf.add_n(logits) / float(len(logits)))


class BinaryDistiller(tf.keras.Model):
    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        alpha: float,
        temperature: float,
        class_weights: dict[int, float],
    ):
        super().__init__(name="binary_knowledge_distiller")
        self.student = student
        self.teacher = teacher
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
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [
            self.loss_tracker,
            self.hard_loss_tracker,
            self.distillation_loss_tracker,
            self.accuracy_metric,
            self.precision_metric,
            self.recall_metric,
            self.auc_metric,
        ]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.student(inputs, training=training)

    def _sample_weights(self, labels: tf.Tensor) -> tf.Tensor:
        labels = tf.reshape(tf.cast(labels, tf.float32), [-1])
        return tf.where(labels >= 0.5, self.positive_weight, self.negative_weight)

    @staticmethod
    def _weighted_mean(values: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        values = tf.reshape(values, [-1])
        weights = tf.reshape(weights, [-1])
        return tf.reduce_sum(values * weights) / tf.maximum(tf.reduce_sum(weights), 1e-7)

    def _soften(self, probabilities: tf.Tensor) -> tf.Tensor:
        probabilities = tf.clip_by_value(probabilities, 1e-5, 1.0 - 1e-5)
        logits = tf.math.log(probabilities) - tf.math.log1p(-probabilities)
        return tf.math.sigmoid(logits / self.temperature)

    def train_step(self, data):
        images, labels = data
        weights = self._sample_weights(labels)
        teacher_probabilities = tf.stop_gradient(self.teacher(images, training=False))
        with tf.GradientTape() as tape:
            student_probabilities = self.student(images, training=True)
            hard_values = tf.keras.losses.binary_crossentropy(labels, student_probabilities)
            hard_loss = self._weighted_mean(hard_values, weights)
            soft_teacher = self._soften(teacher_probabilities)
            soft_student = self._soften(student_probabilities)
            distillation_values = tf.keras.losses.binary_crossentropy(soft_teacher, soft_student)
            distillation_loss = self._weighted_mean(distillation_values, weights)
            loss = (
                self.alpha * hard_loss
                + (1.0 - self.alpha) * (self.temperature**2) * distillation_loss
                + tf.add_n(self.student.losses or [tf.constant(0.0)])
            )
        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        self._update_metrics(labels, student_probabilities, loss, hard_loss, distillation_loss)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        images, labels = data
        probabilities = self.student(images, training=False)
        weights = self._sample_weights(labels)
        hard_values = tf.keras.losses.binary_crossentropy(labels, probabilities)
        hard_loss = self._weighted_mean(hard_values, weights)
        self._update_metrics(labels, probabilities, hard_loss, hard_loss, 0.0)
        return {metric.name: metric.result() for metric in self.metrics}

    def _update_metrics(
        self,
        labels: tf.Tensor,
        probabilities: tf.Tensor,
        loss: tf.Tensor,
        hard_loss: tf.Tensor,
        distillation_loss: tf.Tensor,
    ) -> None:
        self.loss_tracker.update_state(loss)
        self.hard_loss_tracker.update_state(hard_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.accuracy_metric.update_state(labels, probabilities)
        self.precision_metric.update_state(labels, probabilities)
        self.recall_metric.update_state(labels, probabilities)
        self.auc_metric.update_state(labels, probabilities)


def resolve_teacher_paths(args: argparse.Namespace, base_dir: Path) -> list[Path]:
    paths = args.teacher_model or []
    resolved = [path if path.is_absolute() else base_dir / path for path in paths]
    missing = [path for path in resolved if not path.exists()]
    if missing:
        raise SystemExit("Missing teacher model(s):\n" + "\n".join(str(path) for path in missing))
    return resolved


def training_callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=14, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", patience=5, factor=0.5, min_lr=1e-5, verbose=1
        ),
    ]


def predict_model_probabilities(
    model: tf.keras.Model,
    samples: list[data_utils.Sample],
    img_size: int,
    batch_size: int,
) -> np.ndarray:
    dataset = data_utils.make_dataset(samples, img_size, batch_size, False, 0)
    return model.predict(dataset, verbose=0).reshape(-1)


def write_c_array(model_bytes: bytes, output_dir: Path, threshold: float) -> tuple[Path, Path]:
    header = output_dir / f"{MODEL_NAME}_model_data.h"
    source = output_dir / f"{MODEL_NAME}_model_data.cc"
    variable = "g_skippool_presence_distilled_model"
    header.write_text(
        "\n".join(
            [
                "#pragma once",
                "",
                f"extern const unsigned char {variable}[];",
                f"extern const int {variable}_len;",
                f"constexpr float kSkippoolPresenceDistilledThreshold = {threshold:.8f}f;",
                "",
            ]
        ),
        encoding="utf-8",
    )
    values = [f"0x{value:02x}" for value in model_bytes]
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


def save_metrics(path: Path, metrics: dict[str, dict[str, float | int]]) -> None:
    fields = list(next(iter(metrics.values())).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *fields])
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"model": name, **values})


def write_report(
    path: Path,
    args: argparse.Namespace,
    student: tf.keras.Model,
    teacher_paths: list[Path],
    threshold: float,
    metrics: dict[str, dict[str, float | int]],
    artifacts: dict[str, str],
    tflite_info: dict[str, object],
) -> None:
    lines = [
        "# Compact SkipPool Knowledge Distillation Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Distillation",
        "",
        f"- Student parameters: `{student.count_params()}`",
        f"- Alpha (hard-label weight): `{args.alpha}`",
        f"- Temperature: `{args.temperature}`",
        f"- Hard-label fine-tuning epochs: `{args.fine_tune_epochs}`",
        f"- Fine-tuning learning rate: `{args.fine_tune_learning_rate}`",
        f"- Fine-tuning selected: `{getattr(args, 'fine_tune_selected', False)}`",
        f"- Threshold: `{threshold:.4f}` (validation {args.threshold_metric})",
        "- Teachers:",
    ]
    lines.extend(f"  - `{teacher}`" for teacher in teacher_paths)
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
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- {name}: `{artifact}`" for name, artifact in artifacts.items())
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![Training](plots/training_curves.png)",
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
    teacher_paths = resolve_teacher_paths(args, base_dir)
    samples, stats = data_utils.scan_dataset(base_dir, args.source)
    if not samples:
        raise SystemExit("No samples found.")

    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    output_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = data_utils.split_samples(args, samples)
    data_utils.write_dataset_summary(output_dir / "dataset_summary.csv", stats, samples)
    data_utils.write_manifest(output_dir / "train_manifest.csv", train, base_dir)
    data_utils.write_manifest(output_dir / "val_manifest.csv", val, base_dir)
    data_utils.write_manifest(output_dir / "test_manifest.csv", test, base_dir)

    class_weights = data_utils.compute_class_weight(train, args.positive_weight_multiplier)
    train_ds = data_utils.make_dataset(train, args.img_size, args.batch_size, True, args.seed)
    val_ds = data_utils.make_dataset(val, args.img_size, args.batch_size, False, args.seed)

    if teacher_paths:
        teachers = [tf.keras.models.load_model(path, compile=False) for path in teacher_paths]
    else:
        print("Training a fresh multiscale teacher on the current dataset...")
        tf.keras.utils.set_random_seed(args.seed)
        teacher_model = skippool.build_skippool_presence_model(
            args.img_size, args.learning_rate, model_variant="multiscale"
        )
        teacher_history = teacher_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.teacher_epochs,
            class_weight=class_weights,
            callbacks=training_callbacks(),
            verbose=2,
        )
        teacher_path = output_dir / "fresh_multiscale_teacher.keras"
        teacher_model.save(teacher_path)
        data_utils.save_history(output_dir / "teacher_history.csv", teacher_history)
        teacher_paths = [teacher_path]
        teachers = [teacher_model]
    teacher = EnsembleTeacher(teachers)

    print("Training a non-distilled compact baseline...")
    tf.keras.utils.set_random_seed(args.seed + 1)
    baseline = skippool.build_skippool_presence_model(
        args.img_size, args.learning_rate, model_variant="compact"
    )
    initial_student_weights = baseline.get_weights()
    baseline_train_ds = data_utils.make_dataset(
        train, args.img_size, args.batch_size, True, args.seed
    )
    baseline_history = baseline.fit(
        baseline_train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=training_callbacks(),
        verbose=2,
    )
    baseline_path = output_dir / "compact_baseline.keras"
    baseline.save(baseline_path)
    data_utils.save_history(output_dir / "baseline_history.csv", baseline_history)

    print("Training the distilled compact student...")
    student = skippool.build_skippool_presence_model(
        args.img_size, args.learning_rate, model_variant="compact"
    )
    student.set_weights(initial_student_weights)
    distiller = BinaryDistiller(
        student,
        teacher,
        args.alpha,
        args.temperature,
        class_weights,
    )
    distiller.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))
    distilled_train_ds = data_utils.make_dataset(
        train, args.img_size, args.batch_size, True, args.seed
    )
    history = distiller.fit(
        distilled_train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=training_callbacks(),
        verbose=2,
    )
    data_utils.save_history(output_dir / "distillation_history.csv", history)

    args.fine_tune_selected = False
    if args.fine_tune_epochs > 0:
        print("Fine-tuning the distilled student with hard labels...")
        pre_fine_tune_weights = student.get_weights()
        _, pre_fine_tune_metrics = data_utils.find_best_threshold(
            student,
            val,
            args.img_size,
            args.batch_size,
            args.threshold_metric,
            args.min_recall,
        )
        student.compile(
            optimizer=tf.keras.optimizers.Adam(args.fine_tune_learning_rate),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
        fine_tune_train_ds = data_utils.make_dataset(
            train, args.img_size, args.batch_size, True, args.seed
        )
        fine_tune_history = student.fit(
            fine_tune_train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=training_callbacks(),
            verbose=2,
        )
        _, post_fine_tune_metrics = data_utils.find_best_threshold(
            student,
            val,
            args.img_size,
            args.batch_size,
            args.threshold_metric,
            args.min_recall,
        )
        if post_fine_tune_metrics[args.threshold_metric] < pre_fine_tune_metrics[args.threshold_metric]:
            print("Fine-tuning did not improve validation; restoring distilled weights.")
            student.set_weights(pre_fine_tune_weights)
        else:
            args.fine_tune_selected = True
            history = fine_tune_history
        data_utils.save_history(output_dir / "fine_tune_history.csv", fine_tune_history)

    data_utils.save_history(output_dir / "history.csv", history)
    student_path = output_dir / f"best_{MODEL_NAME}.keras"
    student.save(student_path)

    if args.threshold is None:
        threshold, val_metrics = data_utils.find_best_threshold(
            student,
            val,
            args.img_size,
            args.batch_size,
            args.threshold_metric,
            args.min_recall,
        )
    else:
        threshold = float(args.threshold)
        val_metrics = data_utils.evaluate_predictions(
            student, val, args.img_size, args.batch_size, threshold
        )
    test_labels = np.asarray([sample.label for sample in test], dtype=np.int32)
    test_metrics = data_utils.evaluate_predictions(
        student, test, args.img_size, args.batch_size, threshold
    )
    baseline_threshold, _ = data_utils.find_best_threshold(
        baseline,
        val,
        args.img_size,
        args.batch_size,
        args.threshold_metric,
        args.min_recall,
    )
    baseline_metrics = data_utils.evaluate_predictions(
        baseline, test, args.img_size, args.batch_size, baseline_threshold
    )
    teacher_val_probabilities = predict_model_probabilities(
        teacher, val, args.img_size, args.batch_size
    )
    teacher_val_labels = np.asarray([sample.label for sample in val], dtype=np.int32)
    teacher_threshold, _ = max(
        (
            (
                float(candidate),
                data_utils.metrics_from_probabilities(
                    teacher_val_labels, teacher_val_probabilities, float(candidate)
                ),
            )
            for candidate in np.linspace(0.05, 0.95, 91)
        ),
        key=lambda item: (item[1][args.threshold_metric], item[1]["accuracy"]),
    )
    teacher_test_probabilities = predict_model_probabilities(
        teacher, test, args.img_size, args.batch_size
    )
    teacher_metrics = data_utils.metrics_from_probabilities(
        test_labels, teacher_test_probabilities, teacher_threshold
    )
    metrics = {
        "teacher_test": teacher_metrics,
        "baseline_test": baseline_metrics,
        "student_val": val_metrics,
        "student_test": test_metrics,
    }

    artifacts = {
        "teacher_keras": skippool.path_for_report(teacher_paths[0], base_dir),
        "baseline_keras": skippool.path_for_report(baseline_path, base_dir),
        "student_keras": skippool.path_for_report(student_path, base_dir),
        "history": skippool.path_for_report(output_dir / "history.csv", base_dir),
        "distillation_history": skippool.path_for_report(
            output_dir / "distillation_history.csv", base_dir
        ),
    }
    if args.fine_tune_epochs > 0:
        artifacts["fine_tune_history"] = skippool.path_for_report(
            output_dir / "fine_tune_history.csv", base_dir
        )
    float_bytes = data_utils.convert_float_tflite(student, args.img_size)
    float_path = output_dir / f"{MODEL_NAME}_float32.tflite"
    float_path.write_bytes(float_bytes)
    artifacts["tflite_float32"] = skippool.path_for_report(float_path, base_dir)
    tflite_info: dict[str, object] = {"float32": data_utils.inspect_tflite(float_bytes)}

    if not args.no_int8:
        shuffled_train = train[:]
        random.Random(args.seed).shuffle(shuffled_train)
        int8_bytes = data_utils.convert_int8_tflite(
            student,
            shuffled_train,
            args.img_size,
            args.representative_samples,
        )
        int8_path = output_dir / f"{MODEL_NAME}_int8.tflite"
        int8_path.write_bytes(int8_bytes)
        artifacts["tflite_int8"] = skippool.path_for_report(int8_path, base_dir)
        tflite_info["int8"] = data_utils.inspect_tflite(int8_bytes)
        int8_probabilities = skippool.predict_tflite_probabilities(
            int8_bytes, test, args.img_size
        )
        metrics["student_test_int8"] = data_utils.metrics_from_probabilities(
            test_labels, int8_probabilities, threshold
        )
        if not args.no_c_array:
            cc_path, h_path = write_c_array(int8_bytes, output_dir, threshold)
            artifacts["tflite_micro_cc"] = skippool.path_for_report(cc_path, base_dir)
            artifacts["tflite_micro_h"] = skippool.path_for_report(h_path, base_dir)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    skippool.plot_training_history(history, plots_dir / "training_curves.png")
    student_val_probabilities = data_utils.predict_probabilities(
        student, val, args.img_size, args.batch_size
    )
    skippool.plot_threshold_analysis(
        teacher_val_labels,
        student_val_probabilities,
        threshold,
        plots_dir / "threshold_analysis.png",
    )
    skippool.plot_binary_confusion_matrix(
        metrics.get("student_test_int8", test_metrics),
        plots_dir / "confusion_matrix_int8.png",
        "Distilled compact SkipPool - INT8",
    )
    save_metrics(output_dir / "metrics.csv", metrics)
    write_report(
        output_dir / "report.md",
        args,
        student,
        teacher_paths,
        threshold,
        metrics,
        artifacts,
        tflite_info,
    )
    print("Done:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
