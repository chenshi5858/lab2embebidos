# Detector de identificador para ESP32-CAM

Proyecto ESP-IDF preparado para correr `outputs_tf/tflite/cnn_skip_pool_tf_int8.tflite`
con TensorFlow Lite Micro. El modelo espera imagenes monocromaticas de `96x96` y
entrega 4 clases:

- `0`: identificador ausente
- `1`: identificador a la izquierda
- `2`: identificador al centro
- `3`: identificador a la derecha

La deteccion binaria usada para el LED es `clase != 0`.

## Uso sin camara

El proyecto parte en modo imagen estatica. Esto permite compilar, flashear y
verificar que el modelo carga antes de conectar la ESP-CAM.

```bash
cd esp_identifier_detection
idf.py set-target esp32
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

La imagen estatica por defecto es negra, por lo que lo normal es que clasifique
`ausente`. Para probar una foto propia, genera un nuevo arreglo C:

```bash
pip install Pillow
python tools/image_to_static_data.py ../ruta/a/imagen.jpg main/static_image_data.cc
idf.py build flash monitor
```

El script convierte a grayscale y reescala a `96x96`, igual que el pipeline de
entrenamiento.

## Uso con ESP-CAM

Cuando conectes la AI Thinker:

```bash
idf.py menuconfig
```

En `Identifier detector`, habilita `Use ESP-CAM as image source`.

La configuracion de pines ya viene para AI Thinker. El LED por defecto es GPIO 4
(flash LED de varias ESP32-CAM), configurable desde menuconfig.

Tambien puedes cambiarlo directamente en `sdkconfig.defaults` si quieres dejarlo
fijo para el grupo.

## Dependencias

El archivo `main/idf_component.yml` pide:

- `esp-tflite-micro`
- `esp32-camera`

ESP-IDF las descarga con el Component Manager durante `idf.py build`.
