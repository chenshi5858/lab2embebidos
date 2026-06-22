# Robust Identifier Presence Report

Generated: 2026-06-22T13:42:36

## Leakage controls

- Exact duplicates removed: `4`.
- Conflicting duplicate labels removed: `0`.
- Capture block size: `40` frames.
- Entire capture blocks are assigned to only one split.

## Training strategy

- False-positive cost: `2.5`.
- False-negative cost: `1.0`.
- Hard-negative fraction: `0.5`.
- Model parameters: `120489`.
- Augmentation: translation, horizontal flip, brightness, contrast, gamma, noise and mild blur.

## Split counts

- train: `967` images, absent `366`, present `601`, blocks `25`.
- val: `224` images, absent `118`, present `106`, blocks `6`.
- test: `225` images, absent `82`, present `143`, blocks `6`.

## INT8 decision rule

- Threshold: `0.6100`.
- Selection: specificity>=0.950, recall>=0.500.
- Threshold selection used INT8 validation outputs, not test outputs.

## Metrics

| split | accuracy | precision | recall | specificity | balanced_accuracy | f1 | tn | fp | fn | tp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation_int8 | 0.7545 | 0.9180 | 0.5283 | 0.9576 | 0.7430 | 0.6707 | 113 | 5 | 50 | 56 |
| test_int8 | 0.4622 | 0.8438 | 0.1888 | 0.9390 | 0.5639 | 0.3086 | 77 | 5 | 116 | 27 |

## Artifacts

- keras: `outputs/robust_identifier_presence/best_robust_identifier_presence.keras`
- float32_tflite: `outputs/robust_identifier_presence/robust_identifier_presence_float32.tflite`
- int8_tflite: `outputs/robust_identifier_presence/robust_identifier_presence_int8.tflite`
- model_data_cc: `outputs/robust_identifier_presence/robust_identifier_presence_model_data.cc`
- model_data_h: `outputs/robust_identifier_presence/robust_identifier_presence_model_data.h`
- int8_size_bytes: `129584`
- threshold: `0.6100000000000001`
