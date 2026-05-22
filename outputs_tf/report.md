# TensorFlow SkipPoolCNN Report

| model | int8_method | float_test_acc | qat_test_acc | int8_tflite_acc | params | size_mb | tflite_int8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_skip_pool_tf | qat_full_int8 | 0.7091 | 0.7227 | 0.7273 | 14060 | 0.0536 | outputs_tf/tflite/cnn_skip_pool_tf_int8.tflite |