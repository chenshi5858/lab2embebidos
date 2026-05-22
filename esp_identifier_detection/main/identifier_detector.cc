#include "identifier_detector.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "identifier_model_data.h"
#include "sdkconfig.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr char kTag[] = "identifier_detector";
constexpr int kTensorArenaSize = 128 * 1024;

const tflite::Model* g_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
uint8_t* g_tensor_arena = nullptr;

int32_t ClampInt32(int32_t value, int32_t lo, int32_t hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

uint8_t* AllocateTensorArena() {
    uint8_t* arena = static_cast<uint8_t*>(
        heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (arena == nullptr) {
        arena = static_cast<uint8_t*>(
            heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT));
    }
    return arena;
}

}  // namespace

const char* IdentifierClassName(int class_id) {
    switch (class_id) {
        case kAbsentClass:
            return "ausente";
        case kLeftClass:
            return "izquierda";
        case kCenterClass:
            return "centro";
        case kRightClass:
            return "derecha";
        default:
            return "desconocida";
    }
}

bool IdentifierDetector::Begin() {
    g_model = tflite::GetModel(g_identifier_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(kTag, "model schema %d != supported schema %d",
                 g_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    if (g_tensor_arena == nullptr) {
        g_tensor_arena = AllocateTensorArena();
    }
    if (g_tensor_arena == nullptr) {
        ESP_LOGE(kTag, "could not allocate %d bytes for tensor arena", kTensorArenaSize);
        return false;
    }

    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddConcatenation();
    resolver.AddReshape();
    resolver.AddFullyConnected();

    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, g_tensor_arena, kTensorArenaSize);
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return false;
    }

    input_ = g_interpreter->input(0);
    output_ = g_interpreter->output(0);

    if (input_ == nullptr || output_ == nullptr) {
        ESP_LOGE(kTag, "missing input or output tensor");
        return false;
    }

    if (input_->dims->size != 4 || input_->dims->data[1] != kImageHeight ||
        input_->dims->data[2] != kImageWidth || input_->dims->data[3] != kImageChannels) {
        ESP_LOGE(kTag, "unexpected input shape");
        return false;
    }

    if (output_->bytes < kClassCount) {
        ESP_LOGE(kTag, "unexpected output tensor size: %u", output_->bytes);
        return false;
    }

    ESP_LOGI(kTag, "model ready, arena=%d bytes, input scale=%f zero=%d",
             kTensorArenaSize, input_->params.scale, input_->params.zero_point);
    return true;
}

bool IdentifierDetector::Run(const uint8_t* grayscale_image, IdentifierResult* result) {
    if (grayscale_image == nullptr || result == nullptr || g_interpreter == nullptr) {
        return false;
    }

    if (!FillInputTensor(grayscale_image)) {
        return false;
    }

    if (g_interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        return false;
    }

    return ReadOutputTensor(result);
}

bool IdentifierDetector::FillInputTensor(const uint8_t* grayscale_image) {
    if (input_->type == kTfLiteInt8) {
        const float scale = input_->params.scale;
        const int zero_point = input_->params.zero_point;
        if (scale <= 0.0f) {
            return false;
        }
        for (int i = 0; i < kImageElementCount; ++i) {
            const float normalized = static_cast<float>(grayscale_image[i]) / 255.0f;
            const int32_t q = static_cast<int32_t>(lroundf(normalized / scale)) + zero_point;
            input_->data.int8[i] = static_cast<int8_t>(ClampInt32(q, -128, 127));
        }
        return true;
    }

    if (input_->type == kTfLiteUInt8) {
        const float scale = input_->params.scale;
        const int zero_point = input_->params.zero_point;
        if (scale <= 0.0f) {
            return false;
        }
        for (int i = 0; i < kImageElementCount; ++i) {
            const float normalized = static_cast<float>(grayscale_image[i]) / 255.0f;
            const int32_t q = static_cast<int32_t>(lroundf(normalized / scale)) + zero_point;
            input_->data.uint8[i] = static_cast<uint8_t>(ClampInt32(q, 0, 255));
        }
        return true;
    }

    if (input_->type == kTfLiteFloat32) {
        for (int i = 0; i < kImageElementCount; ++i) {
            input_->data.f[i] = static_cast<float>(grayscale_image[i]) / 255.0f;
        }
        return true;
    }

    ESP_LOGE(kTag, "unsupported input tensor type: %d", input_->type);
    return false;
}

bool IdentifierDetector::ReadOutputTensor(IdentifierResult* result) const {
    memset(result, 0, sizeof(*result));

    int best_class = 0;
    float best_score = -INFINITY;
    for (int i = 0; i < kClassCount; ++i) {
        float score = 0.0f;
        int8_t raw = 0;
        if (output_->type == kTfLiteInt8) {
            raw = output_->data.int8[i];
            score = (static_cast<int>(raw) - output_->params.zero_point) * output_->params.scale;
        } else if (output_->type == kTfLiteUInt8) {
            const uint8_t q = output_->data.uint8[i];
            raw = static_cast<int8_t>(static_cast<int>(q) - 128);
            score = (static_cast<int>(q) - output_->params.zero_point) * output_->params.scale;
        } else if (output_->type == kTfLiteFloat32) {
            score = output_->data.f[i];
            raw = static_cast<int8_t>(ClampInt32(static_cast<int32_t>(lroundf(score)), -128, 127));
        } else {
            ESP_LOGE(kTag, "unsupported output tensor type: %d", output_->type);
            return false;
        }

        result->raw_scores[i] = raw;
        result->scores[i] = score;
        if (score > best_score) {
            best_score = score;
            best_class = i;
        }
    }

    result->class_id = best_class;
    result->raw_margin_over_absent =
        static_cast<int>(result->raw_scores[best_class]) - static_cast<int>(result->raw_scores[kAbsentClass]);
    result->detected =
        best_class != kAbsentClass && result->raw_margin_over_absent >= CONFIG_IDENTIFIER_MIN_MARGIN;
    return true;
}
