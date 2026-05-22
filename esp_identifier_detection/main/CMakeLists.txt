idf_component_register(
    SRCS
        "app_main.cc"
        "camera_provider.cc"
        "identifier_detector.cc"
        "identifier_model_data.cc"
        "static_image_data.cc"
    INCLUDE_DIRS "."
    REQUIRES
        esp_driver_gpio
        esp_driver_ledc
        esp-tflite-micro
        esp32-camera
        esp_timer
)
