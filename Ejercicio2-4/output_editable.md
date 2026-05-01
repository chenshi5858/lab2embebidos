# Ejercicio 2.4 - Analisis

Red elegida: `cnn_skip_pool` | Test acc: 66.82% | Latencia: 27.3 ms | Tamano: 54.9 KB

## Como se construyeron los graficos
```text
Cada punto = una topologia entrenada.
Eje Y = accuracy de test.
Accuracy vs tamano: eje X = parametros * 4 / 1024^2; linea roja = 0.5 MB SRAM.
Accuracy vs FPS: eje X = 1 / (MACs / 30e6).
Accuracy vs potencia: eje X = potencia_mW estimada con carga a 1 FPS.
```

## Accuracy vs tamano
![Accuracy vs tamano](figures/accuracy_vs_size.png)

## Accuracy vs FPS
![Accuracy vs FPS](figures/accuracy_vs_fps.png)

## Accuracy vs potencia
![Accuracy vs potencia](figures/accuracy_vs_power.png)

## Roofline por capa
![Roofline por capa](figures/roofline_layers.png)

## Calculo del Roofline
```text
FLOPs_conv = elementos_output * (canales_in/groups * kernel_h * kernel_w) * 2
FLOPs_linear = elementos_output * in_features * 2
intensidad_operacional = FLOPs / bytes_movidos
tiempo_capa = max(FLOPs / pico_compute, bytes_movidos / bandwidth_memoria)
rendimiento_capa = FLOPs / tiempo_capa
pico_compute = 30 MMAC/s * 2 = 0.060 GFLOP/s
bandwidth = 0.12 GB/s
```

## Mapa de etiquetas del Roofline
| Etiqueta | Modelo | Capa / tipo | Salida |
| --- | --- | --- | --- |
| FT-FC1 | fc_tiny | net.1 (FC) | 1x128 |
| FT-ReLU1 | fc_tiny | net.2 (ReLU) | 1x128 |
| FT-FC2 | fc_tiny | net.3 (FC) | 1x4 |
| FS-FC1 | fc_small | net.1 (FC) | 1x256 |
| FS-ReLU1 | fc_small | net.2 (ReLU) | 1x256 |
| FS-FC2 | fc_small | net.4 (FC) | 1x64 |
| FS-ReLU2 | fc_small | net.5 (ReLU) | 1x64 |
| FS-FC3 | fc_small | net.6 (FC) | 1x4 |
| CT-Conv1 | cnn_tiny | features.0 (Conv) | 1x4x96x96 |
| CT-ReLU1 | cnn_tiny | features.1 (ReLU) | 1x4x96x96 |
| CT-MaxPool1 | cnn_tiny | features.2 (MaxPool) | 1x4x48x48 |
| CT-Conv2 | cnn_tiny | features.3 (Conv) | 1x8x48x48 |
| CT-ReLU2 | cnn_tiny | features.4 (ReLU) | 1x8x48x48 |
| CT-MaxPool2 | cnn_tiny | features.5 (MaxPool) | 1x8x24x24 |
| CT-FC1 | cnn_tiny | classifier.1 (FC) | 1x64 |
| CT-ReLU3 | cnn_tiny | classifier.2 (ReLU) | 1x64 |
| CT-FC2 | cnn_tiny | classifier.3 (FC) | 1x4 |
| CG-Conv1 | cnn_gap | features.0 (Conv) | 1x4x96x96 |
| CG-ReLU1 | cnn_gap | features.1 (ReLU) | 1x4x96x96 |
| CG-MaxPool1 | cnn_gap | features.2 (MaxPool) | 1x4x48x48 |
| CG-Conv2 | cnn_gap | features.3 (Conv) | 1x8x48x48 |
| CG-ReLU2 | cnn_gap | features.4 (ReLU) | 1x8x48x48 |
| CG-MaxPool2 | cnn_gap | features.5 (MaxPool) | 1x8x24x24 |
| CG-Conv3 | cnn_gap | features.6 (Conv) | 1x16x24x24 |
| CG-ReLU3 | cnn_gap | features.7 (ReLU) | 1x16x24x24 |
| CG-AvgPool1 | cnn_gap | features.8 (AvgPool) | 1x16x1x1 |
| CG-FC1 | cnn_gap | classifier (FC) | 1x4 |
| SP-MaxPool1 | cnn_skip_pool | skip_pool (MaxPool) | 1x1x12x12 |
| SP-Conv1 | cnn_skip_pool | conv.0 (Conv) | 1x4x48x48 |
| SP-ReLU1 | cnn_skip_pool | conv.1 (ReLU) | 1x4x48x48 |
| SP-MaxPool2 | cnn_skip_pool | conv.2 (MaxPool) | 1x4x48x48 |
| SP-Conv2 | cnn_skip_pool | conv.3 (Conv) | 1x8x24x24 |
| SP-ReLU2 | cnn_skip_pool | conv.4 (ReLU) | 1x8x24x24 |
| SP-MaxPool3 | cnn_skip_pool | conv.5 (MaxPool) | 1x8x24x24 |
| SP-Conv3 | cnn_skip_pool | conv.6 (Conv) | 1x12x12x12 |
| SP-ReLU3 | cnn_skip_pool | conv.7 (ReLU) | 1x12x12x12 |
| SP-MaxPool4 | cnn_skip_pool | conv.8 (MaxPool) | 1x12x12x12 |
| SP-Conv4 | cnn_skip_pool | conv.9 (Conv) | 1x12x12x12 |
| SP-ReLU4 | cnn_skip_pool | conv.10 (ReLU) | 1x12x12x12 |
| SP-Conv5 | cnn_skip_pool | conv.11 (Conv) | 1x16x12x12 |
| SP-ReLU5 | cnn_skip_pool | conv.12 (ReLU) | 1x16x12x12 |
| SP-FC1 | cnn_skip_pool | classifier.1 (FC) | 1x4 |

## Analisis y decision
| Modelo | Veredicto | Analisis |
| --- | --- | --- |
| fc_tiny / fc_small | No recomendadas | Accuracy alrededor de 60%, pero pesos de 4.50 MB y 9.06 MB en float32; exceden la SRAM interna. |
| cnn_tiny | Buen accuracy, mala memoria | Mejor test acc (67.27%), pero 1.13 MB de parametros. Requiere PSRAM o int8 para ser realista. |
| cnn_gap | Muy pequena, baja precision | Solo 6.1 KB de pesos, pero test acc de 50.00%. |
| cnn_skip_pool | Elegida | 66.82% test acc, 73.35% val acc, 54.9 KB de pesos, 0.82 MMACs, 27.3 ms y 36.7 FPS estimados. |

Decision final: usar `cnn_skip_pool`. Mantiene accuracy competitivo frente a cnn_tiny, pero con una fraccion del tamano y menor costo computacional. Para la implementacion final en ESP32-S3 se recomienda cuantizacion int8 con ESP-NN/TFLite Micro.