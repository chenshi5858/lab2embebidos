# SkipPoolCNN Low-Rank Factorization Report

Generated: 2026-06-22T15:29:15

## Configuration

- Compression: truncated SVD of the 24-unit `spatial_head` Dense kernel.
- Rank: `8` of maximum `24`.
- Fine-tuning learning rate: `0.0002`.
- Full-rank parameters: `96679`.
- Low-rank parameters: `36967`.
- Parameter reduction: `61.76%`.
- SVD relative reconstruction error: `0.663195`.
- Initial output MAE after factorization: `0.037540`.

The factorization replaces `W` with `A @ B`, where 
`A` has shape `(flattened_features, 8)` and `B` has shape `(8, 24)`.

## Thresholds

- Full rank: `0.4900`.
- Low rank: `0.3700` (best validation f1).

## Metrics

| model | accuracy | precision | recall | specificity | balanced_accuracy | f1 | f2 | mcc | tn | fp | fn | tp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_rank_val | 0.7422 | 0.7153 | 0.8047 | 0.6797 | 0.7422 | 0.7574 | 0.7851 | 0.4882 | 87 | 41 | 25 | 103 |
| full_rank_test | 0.7500 | 0.7133 | 0.8359 | 0.6641 | 0.7500 | 0.7698 | 0.8082 | 0.5076 | 85 | 43 | 21 | 107 |
| low_rank_val | 0.7148 | 0.6590 | 0.8906 | 0.5391 | 0.7148 | 0.7575 | 0.8321 | 0.4590 | 69 | 59 | 14 | 114 |
| low_rank_test | 0.6875 | 0.6304 | 0.9062 | 0.4688 | 0.6875 | 0.7436 | 0.8333 | 0.4170 | 60 | 68 | 12 | 116 |
| low_rank_test_int8 | 0.6914 | 0.6310 | 0.9219 | 0.4609 | 0.6914 | 0.7492 | 0.8441 | 0.4314 | 59 | 69 | 10 | 118 |

## Model Sizes

| artifact | size_mb |
| --- | ---: |
| full_rank_keras | 1.1918 |
| low_rank_keras | 0.5127 |
| low_rank_tflite_float32 | 0.1466 |
| low_rank_tflite_int8 | 0.0444 |

## Artifacts

- full_rank_keras: `outputs\skippool_presence_low_rank_r8\full_rank_skippool.keras`
- low_rank_keras: `outputs\skippool_presence_low_rank_r8\skippool_presence_low_rank_r8.keras`
- low_rank_history: `outputs\skippool_presence_low_rank_r8\low_rank_history.csv`
- dataset_summary: `outputs\skippool_presence_low_rank_r8\dataset_summary.csv`
- tflite_float32: `outputs\skippool_presence_low_rank_r8\skippool_presence_low_rank_r8_float32.tflite`
- tflite_int8: `outputs\skippool_presence_low_rank_r8\skippool_presence_low_rank_r8_int8.tflite`
- tflite_micro_cc: `outputs\skippool_presence_low_rank_r8\skippool_presence_binary_model_data.cc`
- tflite_micro_h: `outputs\skippool_presence_low_rank_r8\skippool_presence_binary_model_data.h`
- plot_training: `outputs\skippool_presence_low_rank_r8\plots\low_rank_training_curves.png`
- plot_threshold: `outputs\skippool_presence_low_rank_r8\plots\threshold_analysis.png`
- plot_confusion: `outputs\skippool_presence_low_rank_r8\plots\confusion_matrix_int8.png`

## Plots

![Low-rank fine-tuning](plots/low_rank_training_curves.png)

![Threshold analysis](plots/threshold_analysis.png)

![INT8 confusion matrix](plots/confusion_matrix_int8.png)

## TFLite

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

## Deployment

Only the low-rank INT8 model is needed on the ESP-CAM. SVD and the full-rank model are training-time tools.
