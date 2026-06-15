# Model Report

| model | val_acc | test_acc | params | size_mb | macs | flops | onnx |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fc_tiny | 0.5830 | 0.5042 | 1180292 | 4.5025 | 1.18e+06 | 2.36e+06 | outputs/onnx/fc_tiny.onnx |
| fc_small | 0.5851 | 0.5212 | 2376260 | 9.0647 | 2.38e+06 | 4.75e+06 | outputs/onnx/fc_small.onnx |
| cnn_tiny | 0.7170 | 0.6780 | 295572 | 1.1275 | 1.29e+06 | 2.58e+06 | outputs/onnx/cnn_tiny.onnx |
| cnn_gap | 0.5553 | 0.5339 | 1572 | 0.0060 | 1.67e+06 | 3.34e+06 | outputs/onnx/cnn_gap.onnx |
| cnn_skip_pool | 0.7340 | 0.6695 | 14060 | 0.0536 | 8.18e+05 | 1.64e+06 | outputs/onnx/cnn_skip_pool.onnx |