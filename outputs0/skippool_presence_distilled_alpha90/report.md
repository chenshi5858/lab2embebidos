# Compact SkipPool Knowledge Distillation Report

Generated: 2026-06-20T17:21:34

## Distillation

- Student parameters: `6713`
- Alpha (hard-label weight): `0.9`
- Temperature: `2.0`
- Threshold: `0.2400` (validation f1)
- Teachers:
  - `C:\Users\PC Delf\Desktop\lab2embebidos\outputs\skippool_presence_distilled\fresh_multiscale_teacher.keras`

## Metrics

| model | accuracy | precision | recall | specificity | balanced_accuracy | f1 | f2 | mcc | tn | fp | fn | tp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_test | 0.7617 | 0.8080 | 0.7891 | 0.7209 | 0.7550 | 0.7984 | 0.7928 | 0.5073 | 62 | 24 | 27 | 101 |
| baseline_test | 0.7056 | 0.6879 | 0.9297 | 0.3721 | 0.6509 | 0.7907 | 0.8686 | 0.3760 | 32 | 54 | 9 | 119 |
| student_val | 0.7103 | 0.6919 | 0.9297 | 0.3837 | 0.6567 | 0.7933 | 0.8699 | 0.3869 | 33 | 53 | 9 | 119 |
| student_test | 0.6963 | 0.6821 | 0.9219 | 0.3605 | 0.6412 | 0.7841 | 0.8613 | 0.3517 | 31 | 55 | 10 | 118 |
| student_test_int8 | 0.7009 | 0.6839 | 0.9297 | 0.3605 | 0.6451 | 0.7881 | 0.8673 | 0.3649 | 31 | 55 | 9 | 119 |

## Artifacts

- teacher_keras: `outputs\skippool_presence_distilled\fresh_multiscale_teacher.keras`
- baseline_keras: `outputs\skippool_presence_distilled_alpha90\compact_baseline.keras`
- student_keras: `outputs\skippool_presence_distilled_alpha90\best_skippool_presence_distilled.keras`
- history: `outputs\skippool_presence_distilled_alpha90\history.csv`
- tflite_float32: `outputs\skippool_presence_distilled_alpha90\skippool_presence_distilled_float32.tflite`
- tflite_int8: `outputs\skippool_presence_distilled_alpha90\skippool_presence_distilled_int8.tflite`
- tflite_micro_cc: `outputs\skippool_presence_distilled_alpha90\skippool_presence_distilled_model_data.cc`
- tflite_micro_h: `outputs\skippool_presence_distilled_alpha90\skippool_presence_distilled_model_data.h`

## Plots

![Training](plots/training_curves.png)

![Threshold](plots/threshold_analysis.png)

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
