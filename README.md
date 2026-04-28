# Lab 2 - 5 Topologias (FC/CNN) para Cuadrantes

Este proyecto entrena 5 topologias pequenas para detectar en que cuadrante(s) aparece el objeto de interes en imagenes monocromaticas de 96x96.

- Se entrenan **2 modelos fully connected** y **3 modelos CNN**.
- Se exporta cada modelo a **ONNX** para visualizarlo en **Netron**.
- Se calcula para cada topologia:
  - Accuracy (exact match para 3 etiquetas binarias)
  - Tamano del modelo
  - Numero de operaciones aproximado (MACs y FLOPs)
- Se genera `output.pdf` con el resumen.

## Estructura esperada de datos

Ejemplo:

```text
carpeta_dataset/
  images/
    20260415_140218.jpg
    ...
  labels.txt
```

Formato de `labels.txt` (una linea por imagen):

```text
20260415_140218.jpg, 1, 1
20260415_140221.jpg, 1, 3
20260421_161444.jpg, 0, 0
```

### Modos de etiqueta soportados

1. `pair_quadrants` (por defecto):
   - Los dos valores representan cuadrantes activos.
   - `0, 0` significa ningun cuadrante activo.
   - Ejemplos:
     - `1, 1` -> solo cuadrante 1
     - `1, 2` -> cuadrantes 1 y 2
     - `2, 3` -> cuadrantes 2 y 3

2. `flag_quadrant`:
   - Primer valor: bandera de presencia (`0` o `1`)
   - Segundo valor: cuadrante (`1..3`)
   - `0, 0` -> sin objeto

Tip practico:
- Si al entrenar ves que el cuadrante 1 sale con 100% de positivos en train/val, probablemente tu dataset corresponde a `flag_quadrant`.

## Instalacion

```bash
pip install -r requirements.txt
```

## Preprocesamiento de imagenes a 96x96

Si tienes imagenes de 512x512 o 128x128, primero reescalalas a 96x96:

```bash
python resize_images_to_96.py \
  --input_dir carpeta_dataset/images_originales \
  --output_dir carpeta_dataset/images_96 \
  --recursive \
  --grayscale \
  --overwrite
```

Luego usa `carpeta_dataset/images_96` como `--images_dir` en el entrenamiento.

## Ejecucion

```bash
python train_5_topologias.py \
  --images_dir carpeta_dataset/images \
  --labels_file carpeta_dataset/labels.txt \
  --output_dir outputs \
  --label_mode pair_quadrants \
  --train_ratio 0.8 \
  --threshold 0.5 \
  --epochs 25 \
  --batch_size 32 \
  --lr 1e-3
```

Nota: el script divide solo en entrenamiento y validacion. Por ejemplo, `--train_ratio 0.8` implica 80% train y 20% val.

## Salidas generadas

En la carpeta `outputs/`:

- `summary.csv`: resumen por topologia
- `history.csv`: curvas de entrenamiento por epoca
- `checkpoints/*.pt`: pesos de cada modelo
- `onnx/*.onnx`: modelos para abrir en Netron
- `report/output.pdf`: reporte final solicitado

## Abrir visualizacion en Netron

```bash
netron outputs/onnx/fc_tiny.onnx
```

Haz lo mismo para cada `.onnx` generado.
