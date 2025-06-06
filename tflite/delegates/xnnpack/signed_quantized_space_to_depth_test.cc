/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <functional>
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include "tflite/c/c_api_types.h"
#include "tflite/delegates/xnnpack/space_to_depth_tester.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"
#include "tflite/schema/schema_generated.h"

namespace tflite::xnnpack {
namespace {

TEST(SignedQuantizedSpaceToDepth, SinglePixel) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto block_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  const int32_t block_size = block_rng();
  SpaceToDepthTester()
      .BatchSize(batch_rng())
      .InputHeight(block_size)
      .InputWidth(block_size)
      .InputChannels(channel_rng())
      .BlockSize(block_size)
      .Test(TensorType_INT8, xnnpack_delegate.get());
}

TEST(SignedQuantizedSpaceToDepth, SingleRow) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto width_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto block_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  const int32_t block_size = block_rng();
  SpaceToDepthTester()
      .BatchSize(batch_rng())
      .InputHeight(block_size)
      .InputWidth(width_rng() * block_size)
      .InputChannels(channel_rng())
      .BlockSize(block_size)
      .Test(TensorType_INT8, xnnpack_delegate.get());
}

TEST(SignedQuantizedSpaceToDepth, SingleColumn) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto height_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto block_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  const int32_t block_size = block_rng();
  SpaceToDepthTester()
      .BatchSize(batch_rng())
      .InputHeight(height_rng() * block_size)
      .InputWidth(block_size)
      .InputChannels(channel_rng())
      .BlockSize(block_size)
      .Test(TensorType_INT8, xnnpack_delegate.get());
}

TEST(SignedQuantizedSpaceToDepth, FullImage) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto block_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  const int32_t block_size = block_rng();
  SpaceToDepthTester()
      .BatchSize(batch_rng())
      .InputHeight(size_rng() * block_size)
      .InputWidth(size_rng() * block_size)
      .InputChannels(channel_rng())
      .BlockSize(block_size)
      .Test(TensorType_INT8, xnnpack_delegate.get());
}

TEST(SignedQuantizedSpaceToDepth, MultiThreading) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto batch_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));
  auto size_rng =
      std::bind(std::uniform_int_distribution<int32_t>(5, 25), std::ref(rng));
  auto block_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 3), std::ref(rng));
  auto channel_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 16), std::ref(rng));

  const int32_t block_size = block_rng();
  SpaceToDepthTester()
      .BatchSize(batch_rng())
      .InputHeight(size_rng() * block_size)
      .InputWidth(size_rng() * block_size)
      .InputChannels(channel_rng())
      .BlockSize(block_size)
      .Test(TensorType_INT8, xnnpack_delegate.get());
}

}  // namespace
}  // namespace tflite::xnnpack
