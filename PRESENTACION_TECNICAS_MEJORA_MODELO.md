# Tecnicas probadas para mejorar el detector de identificadores

## 1. Resumen ejecutivo

El trabajo tuvo dos etapas que no deben compararse directamente:

1. **Clasificacion por cuadrante (4 clases):** se compararon cinco arquitecturas, se eligio SkipPoolCNN y se probaron TensorFlow Lite, cuantizacion dinamica, cuantizacion INT8 y QAT.
2. **Deteccion binaria de presencia:** se cambio la salida a ausente/presente, se ajustaron pesos de clase y umbrales, se probo una arquitectura multiescala, un modelo compacto y knowledge distillation.

Conclusiones principales:

- SkipPoolCNN fue elegido por su relacion entre accuracy, parametros y MACs, no por tener la mayor accuracy absoluta.
- QAT + INT8 mantuvo e incluso mejoro ligeramente la accuracy del experimento de cuadrantes: `0.7091 -> 0.7273`.
- El mejor detector binario equilibrado fue el SkipPool multiescala: F1 `0.7984`, balanced accuracy `0.7550` y MCC `0.5073`.
- El student compacto redujo los parametros de `96,679` a `6,713` (93.1%) y el TFLite INT8 de `101.68 KB` a `14.03 KB` (86.2%).
- Knowledge distillation no supero al baseline compacto en este split. Su mejor INT8 obtuvo F1 `0.7881`, muy cerca del baseline float `0.7907`.

## 2. Preparacion de datos

- Imagenes redimensionadas a `96x96`.
- LANCZOS para reducir imagenes y BICUBIC para ampliarlas.
- Conversion a escala de grises con un canal: `96x96x1`.
- Normalizacion `0..255 -> 0..1` incluida dentro del grafo TensorFlow.
- Dataset binario final: `1,420` imagenes ESP, `570` sin identificador y `850` con identificador.
- Split estratificado: `992` train, `214` validacion y `214` test.
- Semilla fija `42` para reproducibilidad.

No se encontro data augmentation geometrico o fotometrico en los scripts conservados. La mejora se baso en preprocesamiento, arquitectura, balance de clases, seleccion de umbral y compresion.

## 3. Busqueda manual de arquitectura: clasificacion por cuadrante

Esta etapa es una comparacion manual de arquitecturas, no NAS automatico.

| Modelo | Val accuracy | Test accuracy | Parametros | Tamano FP32 | MACs | Lectura |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FC Tiny | 0.5830 | 0.5042 | 1,180,292 | 4.5025 MB | 1.18 M | Demasiados parametros para su accuracy |
| FC Small | 0.5851 | 0.5212 | 2,376,260 | 9.0647 MB | 2.38 M | Peor relacion costo/rendimiento |
| CNN Tiny | **0.7170** | **0.6780** | 295,572 | 1.1275 MB | 1.29 M | Mejor accuracy de test |
| CNN GAP | 0.5553 | 0.5339 | **1,572** | **0.0060 MB** | 1.67 M | Muy pequeno, pero pierde informacion espacial |
| CNN SkipPool | **0.7340** | 0.6695 | 14,060 | 0.0536 MB | **0.818 M** | Mejor compromiso para ESP32 |

**Decision:** SkipPool conserva una rama de max-pooling directo desde la imagen y la concatena con las features convolucionales. Tuvo 21 veces menos parametros que CNN Tiny y un test accuracy solo `0.0085` menor.

## 4. TensorFlow Lite y cuantizacion del SkipPool de cuadrantes

| Version | Test accuracy | Tamano | Cambio de accuracy |
| --- | ---: | ---: | ---: |
| Keras/TFLite Float32 | 0.7091 | 59.55 KB TFLite | Referencia |
| TFLite Dynamic Range | No se guardo metrica | 22.32 KB | No evaluado en el reporte |
| Modelo despues de QAT | 0.7227 | 342.84 KB Keras | +1.36 puntos porcentuales |
| TFLite full INT8 desde QAT | **0.7273** | **20.77 KB** | +1.82 puntos porcentuales |

La reduccion Float32 -> INT8 fue `65.1%` en almacenamiento TFLite.

### Tecnicas de cuantizacion

- **Dynamic range PTQ:** cuantiza principalmente pesos; no necesita dataset representativo.
- **Full integer PTQ:** cuantiza pesos y activaciones usando imagenes representativas para calibrar rangos.
- **QAT:** durante 5 epochs de fine-tuning se simulan los errores de cuantizacion. Se uso learning rate `1e-4` y luego se exporto a full INT8.
- La mejora pequena de accuracy con QAT puede actuar como regularizacion; no significa que INT8 siempre sea mas preciso.

## 5. Cambio de tarea: presencia binaria

El problema original predecia uno de cuatro cuadrantes. Se reemplazo por una salida sigmoid:

- `0`: identificador ausente.
- `1`: identificador presente.

Este cambio hizo que el objetivo coincidiera con el comportamiento requerido en la ESP-CAM y permitio medir precision, recall, especificidad, F1, F2, balanced accuracy y MCC.

## 6. Primer detector binario: Spatial Tiny

- Arquitectura: cuatro bloques Conv2D + MaxPool, Flatten y Dense.
- Parametros: `48,601`.
- Peso adicional de clase positiva: `1.4`, para reducir falsos negativos.
- Early stopping y ReduceLROnPlateau.
- Umbral elegido en validacion: `0.30`, exigiendo recall >= `0.90` y maximizando especificidad.

| Version | Accuracy | Precision | Recall | Specificity | Balanced acc. | F1 | F2 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spatial Tiny float, test | 0.7453 | 0.7417 | 0.8819 | 0.5412 | 0.7115 | **0.8058** | 0.8498 | 0.4580 |

Tamanos: Float32 `194.59 KB`; INT8 `55.11 KB`, una reduccion de `71.7%`.

En la prueba INT8 se documento ademas:

- Umbral `0.25`: recall `0.9134`, 11 falsos negativos.
- Umbral `0.30`: recall `0.8819`, 15 falsos negativos.

El reporte conservado no contiene la fila completa de metricas INT8; no se deben inventar las metricas faltantes.

## 7. SkipPool binario multiescala

Mejoras arquitectonicas:

- Rama convolucional con BatchNorm.
- Dos skips de la imagen: MaxPool conserva respuestas locales fuertes y AveragePool conserva contexto.
- Flatten para no borrar la posicion de un identificador pequeno.
- Dense de 24 unidades, dropout `0.20` y regularizacion L2 `1e-4`.
- Pesos de clase inversamente proporcionales a la frecuencia.
- Umbral `0.43`, seleccionado por mejor F1 de validacion.

| Version | Accuracy | Precision | Recall | Specificity | Balanced acc. | F1 | F2 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Multiescala float, test | **0.7617** | **0.8080** | 0.7891 | **0.7209** | **0.7550** | **0.7984** | 0.7928 | **0.5073** |
| Multiescala INT8, test | 0.7570 | 0.8016 | 0.7891 | 0.7093 | 0.7492 | 0.7953 | 0.7915 | 0.4966 |

La cuantizacion costo solo `0.0047` de accuracy y `0.0031` de F1. El TFLite paso de `381.83 KB` a `101.68 KB` (`73.4%` menos).

Comparado con Spatial Tiny, el multiescala mejoro accuracy, especificidad, balanced accuracy y MCC, pero redujo recall. Los umbrales y el numero de muestras no fueron exactamente iguales, por lo que esta comparacion debe presentarse como indicativa.

## 8. Compresion arquitectonica: student compacto

- Teacher multiescala: `96,679` parametros.
- Student SkipPool compacto: `6,713` parametros.
- Reduccion de parametros: `93.1%`.
- Student Float32 TFLite: `31.40 KB`.
- Student INT8 TFLite: `14.03 KB`.
- Operaciones estimadas durante exportacion: `0.900 M MACs`.

Baseline compacto sin distillation:

| Accuracy | Precision | Recall | Specificity | Balanced acc. | F1 | F2 | MCC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7056 | 0.6879 | **0.9297** | 0.3721 | 0.6509 | **0.7907** | 0.8686 | 0.3760 |

El modelo compacto conserva casi todo el F1 del teacher (`0.7907` vs `0.7984`), aunque obtiene muchos mas falsos positivos. Es el mejor compromiso de tamano antes de KD.

## 9. Knowledge distillation

El teacher multiescala genera probabilidades suaves. El student aprende con:

`L = alpha * BCE(etiqueta, student) + (1-alpha) * T^2 * BCE(teacher_T, student_T) + regularizacion`

- Temperatura `T = 2.0`.
- Misma inicializacion, split y arquitectura para baseline y student.
- Se probaron `alpha = 0.75` y `alpha = 0.90`.
- `alpha = 0.90` significa 90% de peso para etiquetas reales y 10% para la informacion del teacher.

| Modelo | Acc. | Prec. | Recall | Spec. | Balanced acc. | F1 | F2 | MCC | Tamano INT8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher float | 0.7617 | 0.8080 | 0.7891 | 0.7209 | 0.7550 | 0.7984 | 0.7928 | 0.5073 | 101.68 KB |
| Baseline compacto float | 0.7056 | 0.6879 | 0.9297 | 0.3721 | 0.6509 | **0.7907** | 0.8686 | 0.3760 | No exportado |
| KD alpha 0.75 float | 0.6916 | 0.6722 | **0.9453** | 0.3140 | 0.6296 | 0.7857 | **0.8743** | 0.3477 | 31.40 KB float |
| KD alpha 0.75 INT8 | 0.6869 | 0.6704 | 0.9375 | 0.3140 | 0.6257 | 0.7818 | 0.8683 | 0.3333 | 14.03 KB |
| KD alpha 0.90 float | 0.6963 | 0.6821 | 0.9219 | 0.3605 | 0.6412 | 0.7841 | 0.8613 | 0.3517 | 31.40 KB float |
| KD alpha 0.90 INT8 | **0.7009** | **0.6839** | 0.9297 | **0.3605** | **0.6451** | **0.7881** | 0.8673 | **0.3649** | **14.03 KB** |

**Resultado honesto:** KD no supero el F1 del baseline compacto. `alpha=0.90` fue la mejor variante INT8 y quedo a `0.0026` de F1 del baseline float. El teacher solo era `0.0077` mejor que el baseline en F1, por lo que habia poco conocimiento adicional que transferir.

## 10. Fine-tuning posterior a KD

- Se probaron 20 epochs solo con etiquetas reales.
- Learning rate `1e-4`.
- Se comparo la metrica de validacion antes y despues.
- El fine-tuning no mejoro validacion y fue descartado automaticamente, restaurando los pesos destilados.

Esto es un resultado negativo util: la proteccion de validacion evito degradar el modelo final.

## 11. Ajuste de umbral

El umbral se eligio sobre validacion, nunca sobre test:

| Modelo | Umbral | Objetivo |
| --- | ---: | --- |
| Spatial Tiny | 0.30 | Recall >= 0.90 y maxima especificidad |
| Multiescala | 0.43 | Mejor F1 |
| KD alpha 0.75 | 0.21 | Mejor F1 |
| KD alpha 0.90 | 0.24 | Mejor F1 |

Bajar el umbral aumenta recall y reduce falsos negativos, pero tambien reduce precision y especificidad. Para el robot, el umbral es una decision de sistema, no solo una propiedad del modelo.

## 12. Tecnicas implementadas sin metrica final conservada

- **LiteSpatialFusionCNN:** nueva arquitectura cuantizable con fusion MaxPool/AveragePool. El codigo conserva variantes tiny (`23,829` parametros) y small (`66,711`), pero no existe actualmente un reporte de entrenamiento, por lo que no debe aparecer con accuracy inventada.
- **NAS:** se discutio como posibilidad, pero no existe un script ni un reporte de busqueda NAS en el repositorio actual. La comparacion de cinco arquitecturas fue busqueda manual.
- **Data augmentation:** no aparece implementado en los pipelines conservados.

## 13. Como explicar las metricas

- **Accuracy:** proporcion total de aciertos; puede ocultar problemas cuando las clases estan desbalanceadas.
- **Precision:** de las detecciones positivas, cuantas eran correctas.
- **Recall/sensibilidad:** de los identificadores presentes, cuantos fueron detectados. Un recall alto reduce falsos negativos.
- **Specificity:** de las imagenes sin identificador, cuantas fueron rechazadas correctamente.
- **F1:** equilibrio entre precision y recall.
- **F2:** da mas importancia al recall; relevante si perder un identificador es mas grave que una falsa alarma.
- **Balanced accuracy:** promedio de recall y especificidad; mas informativa ante desbalance.
- **MCC:** calidad global de la matriz de confusion entre `-1` y `1`; penaliza soluciones sesgadas hacia una clase.

## 14. Orden sugerido para las diapositivas

1. Problema y restriccion de ESP32-CAM.
2. Dataset y preprocesamiento `96x96` grayscale.
3. Benchmark de cinco arquitecturas.
4. Por que se eligio SkipPoolCNN.
5. TensorFlow Lite, PTQ y QAT.
6. Cambio de cuadrantes a presencia binaria.
7. Spatial Tiny y ajuste para recall.
8. SkipPool multiescala y equilibrio de metricas.
9. Compresion al student de 6,713 parametros.
10. Knowledge distillation y formula de perdida.
11. Resultados de KD y fine-tuning negativo.
12. Modelo recomendado y trabajo futuro.

## 15. Mensaje final recomendado

> No todas las tecnicas aumentaron la accuracy. La mejora real fue encontrar un compromiso medible entre deteccion y costo: el modelo compacto INT8 ocupa 14.03 KB y mantiene F1 0.7881, muy cerca del teacher de 101.68 KB. QAT fue efectivo en la tarea original; KD, en cambio, mostro que un teacher apenas mejor que el student no entrega suficiente ventaja.

## Fuentes locales

- `outputs_tf/report.md`: SkipPool de cuadrantes, QAT e INT8.
- `outputs/report.md` en Git HEAD: benchmark PyTorch de cinco arquitecturas.
- `outputs/identifier_presence/report.md` en Git HEAD: Spatial Tiny binario.
- `outputs/skippool_presence_binary_improved/report.md`: multiescala binario.
- `outputs/skippool_presence_distilled/report.md`: KD alpha 0.75.
- `outputs/skippool_presence_distilled_alpha90/report.md`: KD alpha 0.90.
- `outputs/skippool_presence_distilled_finetuned/report.md`: fine-tuning posterior.
