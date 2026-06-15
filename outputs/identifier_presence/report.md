# Identifier Presence TensorFlow Report

Generated: 2026-06-14T21:39:23

## Dataset

- Source filter: `esp`
- Label files used: 4
- Samples used: 1416
- Absent class 0: 568
- Present class 1: 848
- Missing image references skipped: 4
- Bad label lines skipped: 0

## Splits

| split | total | absent_0 | present_1 |
| --- | ---: | ---: | ---: |
| train | 992 | 398 | 594 |
| val | 212 | 85 | 127 |
| test | 212 | 85 | 127 |

## Model

- Variant: `spatial_tiny`
- Input: `96x96x1` grayscale, raw pixel range 0..255.
- Parameters: 48601
- Positive class weight multiplier: `1.4`
- Core ops: Conv2D, MaxPool2D, Flatten, Dense, Sigmoid.

## Metrics

Threshold: `0.3000` (validation recall >= 0.9, max specificity)

| split | accuracy | precision | recall | specificity | balanced_accuracy | f1 | f2 | mcc | tn | fp | fn | tp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 0.7406 | 0.7278 | 0.9055 | 0.4941 | 0.6998 | 0.8070 | 0.8634 | 0.4495 | 42 | 43 | 12 | 115 |
| test | 0.7453 | 0.7417 | 0.8819 | 0.5412 | 0.7115 | 0.8058 | 0.8498 | 0.4580 | 46 | 39 | 15 | 112 |

## ESP Threshold Recommendation

For field testing on the ESP, start with threshold `0.25` if the robot is still
missing identifiers. On the quantized TFLite test run, `0.25` gave recall
`0.9134` with `11` false negatives. Threshold `0.30` is more conservative and
gave recall `0.8819` with `15` false negatives.

## Artifacts

- keras: `outputs/identifier_presence/best_identifier_presence.keras`
- history: `outputs/identifier_presence/history.csv`
- dataset_summary: `outputs/identifier_presence/dataset_summary.csv`
- tflite_float32: `outputs/identifier_presence/identifier_presence_float32.tflite`
- tflite_int8: `outputs/identifier_presence/identifier_presence_int8.tflite`
- tflite_micro_cc: `outputs/identifier_presence/identifier_presence_model_data.cc`
- tflite_micro_h: `outputs/identifier_presence/identifier_presence_model_data.h`

## TFLite Quantization Info

```json
{
  "float32": {
    "input_name": "grayscale_96x96",
    "input_shape": [
      1,
      96,
      96,
      1
    ],
    "input_dtype": "<class 'numpy.float32'>",
    "input_quantization": [
      0.0,
      0.0
    ],
    "output_name": "Identity",
    "output_shape": [
      1,
      1
    ],
    "output_dtype": "<class 'numpy.float32'>",
    "output_quantization": [
      0.0,
      0.0
    ]
  },
  "int8": {
    "input_name": "grayscale_96x96",
    "input_shape": [
      1,
      96,
      96,
      1
    ],
    "input_dtype": "<class 'numpy.int8'>",
    "input_quantization": [
      1.0,
      -128.0
    ],
    "output_name": "Identity",
    "output_shape": [
      1,
      1
    ],
    "output_dtype": "<class 'numpy.int8'>",
    "output_quantization": [
      0.00390625,
      -128.0
    ]
  }
}
```

## ESP Notes

- Capture 96x96 grayscale frames on the ESP camera.
- Feed pixels in row-major order. For the int8 model, quantize each pixel with:
  `input_int8 = round(pixel / input_scale + input_zero_point)`.
- The output is a sigmoid probability after dequantization. Presence is true when it is >= threshold.

## Keras Summary

```text
Layer (type) | Output shape | Params
--- | --- | ---
grayscale_96x96 (InputLayer) | (None, 96, 96, 1) | 0
scale_to_0_1 (Rescaling) | (None, 96, 96, 1) | 0
conv_1 (Conv2D) | (None, 96, 96, 8) | 80
pool_1 (MaxPooling2D) | (None, 48, 48, 8) | 0
conv_2 (Conv2D) | (None, 48, 48, 16) | 1168
pool_2 (MaxPooling2D) | (None, 24, 24, 16) | 0
conv_3 (Conv2D) | (None, 24, 24, 24) | 3480
pool_3 (MaxPooling2D) | (None, 12, 12, 24) | 0
conv_4 (Conv2D) | (None, 12, 12, 32) | 6944
pool_4 (MaxPooling2D) | (None, 6, 6, 32) | 0
flatten_spatial_features (Flatten) | (None, 1152) | 0
spatial_head (Dense) | (None, 32) | 36896
head_dropout (Dropout) | (None, 32) | 0
presence (Dense) | (None, 1) | 33

Total params: 48601
Trainable params: 48601
Non-trainable params: 0
```
