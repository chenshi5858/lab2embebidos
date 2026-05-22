#include <stdint.h>

#include "camera_provider.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "identifier_detector.h"
#include "sdkconfig.h"
#include "static_image_data.h"

namespace {

constexpr char kTag[] = "identifier_app";

void ConfigureLed() {
    gpio_config_t io_conf = {};
    io_conf.pin_bit_mask = 1ULL << CONFIG_IDENTIFIER_LED_GPIO;
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.intr_type = GPIO_INTR_DISABLE;
    gpio_config(&io_conf);
    gpio_set_level(static_cast<gpio_num_t>(CONFIG_IDENTIFIER_LED_GPIO), 0);
}

void SetLed(bool on) {
    gpio_set_level(static_cast<gpio_num_t>(CONFIG_IDENTIFIER_LED_GPIO), on ? 1 : 0);
}

void LogResult(const IdentifierResult& result, int64_t elapsed_us) {
    ESP_LOGI(kTag,
             "class=%d (%s) detected=%s margin=%d time=%lld us scores=[%.3f %.3f %.3f %.3f] raw=[%d %d %d %d]",
             result.class_id,
             IdentifierClassName(result.class_id),
             result.detected ? "yes" : "no",
             result.raw_margin_over_absent,
             elapsed_us,
             static_cast<double>(result.scores[0]),
             static_cast<double>(result.scores[1]),
             static_cast<double>(result.scores[2]),
             static_cast<double>(result.scores[3]),
             result.raw_scores[0],
             result.raw_scores[1],
             result.raw_scores[2],
             result.raw_scores[3]);
}

}  // namespace

extern "C" void app_main(void) {
    ConfigureLed();

    IdentifierDetector detector;
    if (!detector.Begin()) {
        ESP_LOGE(kTag, "detector init failed");
        return;
    }

#if CONFIG_IDENTIFIER_USE_CAMERA
    static uint8_t image[kImageElementCount];
    if (!InitCameraProvider()) {
        ESP_LOGE(kTag, "camera init failed");
        return;
    }
#else
    const uint8_t* image = g_static_image_data;
    ESP_LOGI(kTag, "running with static image; enable camera in menuconfig later");
#endif

    while (true) {
#if CONFIG_IDENTIFIER_USE_CAMERA
        if (!CaptureCameraImage(image)) {
            SetLed(false);
            vTaskDelay(pdMS_TO_TICKS(CONFIG_IDENTIFIER_PERIOD_MS));
            continue;
        }
#endif

        IdentifierResult result;
        const int64_t start = esp_timer_get_time();
        const bool ok = detector.Run(image, &result);
        const int64_t elapsed = esp_timer_get_time() - start;

        if (ok) {
            SetLed(result.detected);
            LogResult(result, elapsed);
        } else {
            SetLed(false);
            ESP_LOGE(kTag, "inference failed");
        }

        vTaskDelay(pdMS_TO_TICKS(CONFIG_IDENTIFIER_PERIOD_MS));
    }
}
