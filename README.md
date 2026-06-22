# Identifier Presence Model

This project trains a small TensorFlow binary classifier for the resized datasets
`dataset_grupo_*_reescalado`. The model predicts whether the identifier is
present in a grayscale image. In each `etiquetas.txt` row, only the first
numeric column is used:

```text
esp_0068.png, 1, 2
```

Here `1` means identifier present. The last numeric column is ignored.

## Model strategy

- Input: `96x96x1` grayscale.
- Target deployment: ESP32-S3-CAM / TensorFlow Lite Micro.
- Default source: `esp`, because those images match the camera domain better
  than phone images. Use `--source all` if you want to mix all sources.
- Architecture: tiny spatial CNN with TFLite Micro friendly ops:
  `Conv2D`, `MaxPool2D`, `Flatten`, `Dense`.
- Training uses small brightness/contrast/translation augmentation and class
  weights for imbalance.
- The default threshold is selected with validation F2, which prioritizes
  detecting the identifier over avoiding every false positive.
- Exported artifacts include `.keras`, float32 `.tflite`, full-int8 `.tflite`,
  and a C/C++ array for firmware.

## Environment

The current `venv` in this folder uses Python 3.14, and TensorFlow wheels are
not available for that version. Create a Python 3.11 or 3.12 environment.

On Windows:

```powershell
py -3.12 -m venv .venv_tf
.\.venv_tf\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On WSL:

```bash
python3 -m venv ~/.venvs/lab2embebidos_tf
source ~/.venvs/lab2embebidos_tf/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train and export

From this folder:

```powershell
python train_identifier_presence.py --source esp
```

Main outputs are written to:

```text
outputs/identifier_presence/
```

Useful variants:

```powershell
python train_identifier_presence.py --scan_only
python train_identifier_presence.py --source all
python train_identifier_presence.py --split_mode group --source esp
python train_identifier_presence.py --source esp --threshold 0.5
python train_identifier_presence.py --source esp --min_recall 0.90
python train_identifier_presence.py --source esp --positive_weight_multiplier 1.4
python train_identifier_presence.py --source esp --model_variant spatial_small
```

## Binary SkipPoolCNN variant

To train the SkipPoolCNN architecture as a binary presence detector instead of
predicting quadrants, run:

```powershell
python skippoolcnn_tf_presence_binary_version.py --source esp
```

This variant uses the first numeric label as `0 = absent` and `1 = present`, ignores the
quadrant column, selects a validation threshold, and exports Float32/INT8 TFLite
models, firmware C arrays, reports, and evaluation plots.

The current default is the improved `multiscale` variant. It adds max/average
skip branches, batch normalization and a small spatial head, and selects the
threshold using validation F1. Its outputs are written to:

```text
outputs/skippool_presence_binary_improved/
```

To reproduce the original 6,713-parameter architecture, use:

```powershell
python skippoolcnn_tf_presence_binary_version.py --source esp --model_variant compact --output_dir outputs/skippool_presence_binary_compact
```

## LiteSpatialFusionCNN

The lighter alternative created for ESP deployment keeps a 12x12 spatial map
and fuses it with max/average-pooled raw pixels. Train and export it with:

```powershell
python train_lite_spatial_presence.py --source esp --variant small
```

Outputs, including the full-INT8 model and C/C++ model array, are written to
`outputs/lite_spatial_presence/`.

## ESP32-S3-CAM inference notes

Use the int8 model for TensorFlow Lite Micro:

```text
outputs/identifier_presence/identifier_presence_int8.tflite
outputs/identifier_presence/identifier_presence_model_data.cc
outputs/identifier_presence/identifier_presence_model_data.h
```

Capture a `96x96` grayscale frame and write pixels in row-major order. For the
int8 model, quantize each pixel using the values reported in
`outputs/identifier_presence/report.md`:

```c
input_int8 = round(pixel / input_scale + input_zero_point);
```

After inference, dequantize the output and compare it with the threshold written
in the report. By default the script selects the threshold that maximizes
validation F2, which gives more importance to recall. Pass `--threshold 0.5` if
you want a fixed threshold. Values greater or equal to the threshold mean the
identifier is present.

For field testing, if the ESP still misses identifiers, start with threshold
`0.25`. It is more sensitive than the automatic threshold and is usually the
better first value when false negatives are more harmful than false positives.
