#ifndef IDENTIFIER_DETECTOR_H_
#define IDENTIFIER_DETECTOR_H_

#include <stdint.h>

#include "model_settings.h"
#include "tensorflow/lite/c/common.h"

struct IdentifierResult {
    int class_id;
    int8_t raw_scores[kClassCount];
    float scores[kClassCount];
    int raw_margin_over_absent;
    bool detected;
};

class IdentifierDetector {
public:
    bool Begin();
    bool Run(const uint8_t* grayscale_image, IdentifierResult* result);

private:
    bool FillInputTensor(const uint8_t* grayscale_image);
    bool ReadOutputTensor(IdentifierResult* result) const;

    TfLiteTensor* input_ = nullptr;
    TfLiteTensor* output_ = nullptr;
};

#endif  // IDENTIFIER_DETECTOR_H_
