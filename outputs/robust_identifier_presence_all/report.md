# Robust Identifier Presence Report

Generated: 2026-06-22T13:47:36

## Leakage controls

- Exact duplicates removed: `37`.
- Conflicting duplicate labels removed: `0`.
- Capture block size: `40` frames.
- Entire capture blocks are assigned to only one split.

## Training strategy

- False-positive cost: `2.0`.
- False-negative cost: `1.0`.
- Hard-negative fraction: `0.5`.
- Model parameters: `120489`.
- Augmentation: translation, horizontal flip, brightness, contrast, gamma, noise and mild blur.

## Split counts

- train: `1551` images, absent `459`, present `1092`, blocks `40`.
- val: `380` images, absent `154`, present `226`, blocks `10`.
- test: `385` images, absent `150`, present `235`, blocks `10`.

## INT8 decision rule

- Threshold: `0.4850`.
- Selection: specificity>=0.950, recall>=0.500.
- Threshold selection used INT8 validation outputs, not test outputs.

## Metrics

| split | accuracy | precision | recall | specificity | balanced_accuracy | f1 | tn | fp | fn | tp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation_int8 | 0.8737 | 0.9635 | 0.8186 | 0.9545 | 0.8866 | 0.8852 | 147 | 7 | 41 | 185 |
| test_int8 | 0.7065 | 0.9552 | 0.5447 | 0.9600 | 0.7523 | 0.6938 | 144 | 6 | 107 | 128 |

## Artifacts

- keras: `outputs/robust_identifier_presence_all/best_robust_identifier_presence.keras`
- float32_tflite: `outputs/robust_identifier_presence_all/robust_identifier_presence_float32.tflite`
- int8_tflite: `outputs/robust_identifier_presence_all/robust_identifier_presence_int8.tflite`
- model_data_cc: `outputs/robust_identifier_presence_all/robust_identifier_presence_model_data.cc`
- model_data_h: `outputs/robust_identifier_presence_all/robust_identifier_presence_model_data.h`
- int8_size_bytes: `129584`
- threshold: `0.485`
