# Ejercicio 2.2 - Modelos

Cinco topologias pequenas entrenadas desde `model.py`: dos fully connected y tres CNN. Las imagenes se redimensionan en memoria a 96x96, que es el tamano esperado por las arquitecturas.

## Resumen de metricas
| Modelo | Val acc | Test acc | Params | Tamano | MACs | FLOPs | ONNX |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fc_tiny | 61.50% | 60.00% | 1,180,292 | 4.5025 MB | 1.18e+06 | 2.36e+06 | fc_tiny.onnx |
| fc_small | 60.59% | 59.09% | 2,376,260 | 9.0647 MB | 2.38e+06 | 4.75e+06 | fc_small.onnx |
| cnn_tiny | 69.70% | 67.27% | 295,572 | 1.1275 MB | 1.29e+06 | 2.58e+06 | cnn_tiny.onnx |
| cnn_gap | 51.94% | 50.00% | 1,572 | 0.0060 MB | 1.67e+06 | 3.34e+06 | cnn_gap.onnx |
| cnn_skip_pool | 73.35% | 66.82% | 14,060 | 0.0536 MB | 8.18e+05 | 1.64e+06 | cnn_skip_pool.onnx |

## Calculo de metricas reportadas
```text
accuracy = aciertos / total_muestras
tamano_parametros_MB = parametros * 4 bytes / 1024^2  (float32)
FLOPs ~= 2 * MACs
MACs se obtuvieron con thop.profile(model, input=(1,1,96,96))
ONNX se exporto con torch.onnx.export y se abrio en Netron para las capturas.
```

## Lectura rapida
cnn_tiny alcanza el mayor accuracy de test, pero ocupa mas de 1 MB en parametros float32. cnn_skip_pool queda practicamente empatado en accuracy, usa solo 0.0536 MB y necesita menos operaciones, por lo que se conserva como candidato principal para ESP32-S3.

## Visualizaciones Netron
### fc_tiny
Test acc: 60.00% | Tamano: 4.5025 MB | MACs: 1.18e+06
![Netron fc_tiny](netron_crops/fc_tiny_netron_crop.png)

### fc_small
Test acc: 59.09% | Tamano: 9.0647 MB | MACs: 2.38e+06
![Netron fc_small](netron_crops/fc_small_netron_crop.png)

### cnn_tiny
Test acc: 67.27% | Tamano: 1.1275 MB | MACs: 1.29e+06
![Netron cnn_tiny](netron_crops/cnn_tiny_netron_crop.png)

### cnn_gap
Test acc: 50.00% | Tamano: 0.0060 MB | MACs: 1.67e+06
![Netron cnn_gap](netron_crops/cnn_gap_netron_crop.png)

### cnn_skip_pool
Test acc: 66.82% | Tamano: 0.0536 MB | MACs: 8.18e+05
![Netron cnn_skip_pool](netron_crops/cnn_skip_pool_netron_crop.png)
