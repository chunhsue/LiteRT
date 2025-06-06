/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_PRELU_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_PRELU_H_

#include <string>

#include "tflite/delegates/gpu/common/data_type.h"
#include "tflite/delegates/gpu/common/operations.h"
#include "tflite/delegates/gpu/common/status.h"
#include "tflite/delegates/gpu/common/task/gpu_operation.h"
#include "tflite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

ElementwiseDescriptor CreatePReLU(const PReLUAttributes& attr,
                                  TensorDescriptor tensor_desc);

GPUOperation CreatePReLU(const GpuInfo& gpu_info,
                         const OperationDef& definition,
                         const PReLUAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_PRELU_H_
