#include "camera_provider.h"

#include <string.h>

#include "driver/ledc.h"
#include "esp_camera.h"
#include "esp_log.h"

namespace {

constexpr char kTag[] = "camera_provider";

// ESP32-CAM AI Thinker pinout.
constexpr int kPwdnGpio = 32;
constexpr int kResetGpio = -1;
constexpr int kXclkGpio = 0;
constexpr int kSiodGpio = 26;
constexpr int kSiocGpio = 27;
constexpr int kY9Gpio = 35;
constexpr int kY8Gpio = 34;
constexpr int kY7Gpio = 39;
constexpr int kY6Gpio = 36;
constexpr int kY5Gpio = 21;
constexpr int kY4Gpio = 19;
constexpr int kY3Gpio = 18;
constexpr int kY2Gpio = 5;
constexpr int kVsyncGpio = 25;
constexpr int kHrefGpio = 23;
constexpr int kPclkGpio = 22;

uint8_t Rgb565ToGray(uint8_t high, uint8_t low) {
    const uint16_t pixel = (static_cast<uint16_t>(high) << 8) | low;
    const uint8_t r = ((pixel >> 11) & 0x1f) << 3;
    const uint8_t g = ((pixel >> 5) & 0x3f) << 2;
    const uint8_t b = (pixel & 0x1f) << 3;
    return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
}

}  // namespace

bool InitCameraProvider() {
    camera_config_t config = {};
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = kY2Gpio;
    config.pin_d1 = kY3Gpio;
    config.pin_d2 = kY4Gpio;
    config.pin_d3 = kY5Gpio;
    config.pin_d4 = kY6Gpio;
    config.pin_d5 = kY7Gpio;
    config.pin_d6 = kY8Gpio;
    config.pin_d7 = kY9Gpio;
    config.pin_xclk = kXclkGpio;
    config.pin_pclk = kPclkGpio;
    config.pin_vsync = kVsyncGpio;
    config.pin_href = kHrefGpio;
    config.pin_sccb_sda = kSiodGpio;
    config.pin_sccb_scl = kSiocGpio;
    config.pin_pwdn = kPwdnGpio;
    config.pin_reset = kResetGpio;
    config.xclk_freq_hz = 20000000;
    config.frame_size = FRAMESIZE_96X96;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.jpeg_quality = 12;
    config.fb_count = 1;

    const esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "esp_camera_init failed: 0x%x", err);
        return false;
    }
    return true;
}

bool CaptureCameraImage(uint8_t* grayscale_image) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (fb == nullptr) {
        ESP_LOGE(kTag, "camera capture failed");
        return false;
    }

    bool ok = false;
    if (fb->width == kImageWidth && fb->height == kImageHeight &&
        fb->format == PIXFORMAT_GRAYSCALE && fb->len >= kImageElementCount) {
        memcpy(grayscale_image, fb->buf, kImageElementCount);
        ok = true;
    } else if (fb->width == kImageWidth && fb->height == kImageHeight &&
               fb->format == PIXFORMAT_RGB565 && fb->len >= kImageElementCount * 2) {
        for (int i = 0; i < kImageElementCount; ++i) {
            grayscale_image[i] = Rgb565ToGray(fb->buf[2 * i], fb->buf[2 * i + 1]);
        }
        ok = true;
    } else {
        ESP_LOGE(kTag, "unexpected frame: %zux%zu format=%d len=%zu",
                 static_cast<size_t>(fb->width),
                 static_cast<size_t>(fb->height),
                 fb->format,
                 static_cast<size_t>(fb->len));
    }

    esp_camera_fb_return(fb);
    return ok;
}
