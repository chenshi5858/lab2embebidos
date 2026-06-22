# SkipPoolCNN Low-Rank Factorization Report

Generated: 2026-06-22T00:12:00

## Configuration

- Compression: truncated SVD of the 24-unit `spatial_head` Dense kernel.
- Rank: `8` of maximum `24`.
- Fine-tuning learning rate: `0.0002`.
- Full-rank parameters: `96679`.
- Low-rank parameters: `36967`.
- Parameter reduction: `61.76%`.
- SVD relative reconstruction error: `0.627609`.
- Initial output MAE after factorization: `0.034698`.

The factorization replaces `W` with `A @ B`, where 
`A` has shape `(flattened_features, 8)` and `B` has shape `(8, 24)`.

## Thresholds

- Full rank: `0.4300`.
- Low rank: `0.4200` (best validation f1).

## Metrics

| model | accuracy | precision | recall | specificity | balanced_accuracy | f1 | f2 | mcc | tn | fp | fn | tp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_rank_val | 0.7804 | 0.8240 | 0.8047 | 0.7442 | 0.7744 | 0.8142 | 0.8085 | 0.5460 | 64 | 22 | 25 | 103 |
| full_rank_test | 0.7617 | 0.8080 | 0.7891 | 0.7209 | 0.7550 | 0.7984 | 0.7928 | 0.5073 | 62 | 24 | 27 | 101 |
| low_rank_val | 0.7664 | 0.8047 | 0.8047 | 0.7093 | 0.7570 | 0.8047 | 0.8047 | 0.5140 | 61 | 25 | 25 | 103 |
| low_rank_test | 0.7523 | 0.7820 | 0.8125 | 0.6628 | 0.7376 | 0.7969 | 0.8062 | 0.4804 | 57 | 29 | 24 | 104 |
| low_rank_test_int8 | 0.7570 | 0.7879 | 0.8125 | 0.6744 | 0.7435 | 0.8000 | 0.8075 | 0.4910 | 58 | 28 | 24 | 104 |

## Model Sizes

| artifact | size_mb |
| --- | ---: |
| full_rank_keras | 1.1834 |
| low_rank_keras | 0.5040 |
| low_rank_tflite_float32 | 0.1452 |
| low_rank_tflite_int8 | 0.0425 |

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
