# Dataset Laboratorio 2 Sistemas Embebidos - Grupo 8

El dataset busca imitar lo que se cree que será la competencia en cuanto a setup:

- Iluminación natural de sala y horario de "Laboratorio de prototipos"
- Sensor de la cámara entregada en el curso del ESP-CAM
- De fondo hay personas mirando
- En algunas (pocas) fotos, hay incluso más de un identificador, para lo que se etiquetó el más cercano

Las fotos y etiquetas se encuentran en el directorio [`esp/`](./esp/).

## Resolución

- B&N
- 128 x 128
- Sin postprocesado (Sin ecualización de histograma, aumento de contraste, reescalado, etc.)

## Resumen del dataset

**Número total de fotos:** 598

| ¿Hay ID? | Sí | No |
|---|---:|---:|
| Cantidad | 269 | 329 |

| ¿Sector? | Izquierda | Centro | Derecha |
|---|---:|---:|---:|
| Cantidad | 80 | 138 | 51 |

## Extra

Por malos resultados en el entrenamiento, nos pusimos la capa y les regalamos este dataset también con [bounding boxes](./esp/bounding_boxes.txt). El archivo tiene el formato de:

```txt
{filename},{is_id_on_pic?},{bb_x(or -1)},{bb_y(or -1)},{bb_w(or -1)},{bb_h(or -1)}
```