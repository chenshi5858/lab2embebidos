#ifndef CAMERA_PROVIDER_H_
#define CAMERA_PROVIDER_H_

#include <stdint.h>

#include "model_settings.h"

bool InitCameraProvider();
bool CaptureCameraImage(uint8_t* grayscale_image);

#endif  // CAMERA_PROVIDER_H_
