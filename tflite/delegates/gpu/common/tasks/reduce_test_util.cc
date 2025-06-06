/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tflite/delegates/gpu/common/tasks/reduce_test_util.h"

#include <memory>
#include <vector>

#include "tflite/delegates/gpu/common/operations.h"
#include "tflite/delegates/gpu/common/status.h"
#include "tflite/delegates/gpu/common/task/testing_util.h"
#include "tflite/delegates/gpu/common/tasks/reduce.h"

namespace tflite {
namespace gpu {
namespace {
template <DataType T>
absl::Status ReduceSumChannelsIntTest(TestExecutionEnvironment* env) {
  tflite::gpu::Tensor<BHWC, T> src;
  src.shape = BHWC(1, 2, 1, 5);
  src.data = {1, 2, -5, -2, 1, 3, 4, -2, 1, 4};

  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  tflite::gpu::Tensor<BHWC, T> ref_tensor;
  ref_tensor.shape = BHWC(1, 2, 1, 1);
  ref_tensor.data = {-3, 10};

  for (auto storage : env->GetSupportedStorages(T)) {
    OperationDef op_def;
    op_def.precision = CalculationsPrecision::F32;
    op_def.src_tensors.push_back({T, storage, Layout::HWC});
    op_def.dst_tensors.push_back({T, storage, Layout::HWC});
    TensorDescriptor src_0, dst;
    src_0 = op_def.src_tensors[0];
    src_0.UploadData(src);
    dst.SetBHWCShape(BHWC(1, 2, 1, 1));
    Reduce operation = CreateReduce(axis, src.shape, OperationType::REDUCE_SUM,
                                    op_def, env->GetGpuInfo());
    RETURN_IF_ERROR(env->ExecuteGPUOperation(
        {&src_0}, {&dst}, std::make_unique<Reduce>(std::move(operation))));
    tflite::gpu::Tensor<BHWC, T> dst_tensor;
    dst.DownloadData(&dst_tensor);
    if (dst_tensor.data != ref_tensor.data) {
      return absl::InternalError("not equal");
    }
  }
  return absl::OkStatus();
}

template absl::Status ReduceSumChannelsIntTest<DataType::INT32>(
    TestExecutionEnvironment* env);
template absl::Status ReduceSumChannelsIntTest<DataType::INT16>(
    TestExecutionEnvironment* env);
template absl::Status ReduceSumChannelsIntTest<DataType::INT8>(
    TestExecutionEnvironment* env);

template <DataType T>
absl::Status ReduceProductChannelsUIntTest(TestExecutionEnvironment* env) {
  tflite::gpu::Tensor<BHWC, T> src;
  src.shape = BHWC(1, 3, 1, 2);
  src.data = {1, 2, 3, 4, 0, 7};
  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  tflite::gpu::Tensor<BHWC, T> ref_tensor;
  ref_tensor.shape = BHWC(1, 3, 1, 1);
  ref_tensor.data = {2, 12, 0};

  for (auto storage : env->GetSupportedStorages(T)) {
    OperationDef op_def;
    op_def.precision = CalculationsPrecision::F32;
    op_def.src_tensors.push_back({T, storage, Layout::HWC});
    op_def.dst_tensors.push_back({T, storage, Layout::HWC});
    TensorDescriptor src_0, dst;
    src_0 = op_def.src_tensors[0];
    src_0.UploadData(src);
    dst.SetBHWCShape(BHWC(1, 3, 1, 1));
    Reduce operation =
        CreateReduce(axis, src.shape, OperationType::REDUCE_PRODUCT, op_def,
                     env->GetGpuInfo());
    RETURN_IF_ERROR(env->ExecuteGPUOperation(
        {&src_0}, {&dst}, std::make_unique<Reduce>(std::move(operation))));
    tflite::gpu::Tensor<BHWC, T> dst_tensor;
    dst.DownloadData(&dst_tensor);
    if (dst_tensor.data != ref_tensor.data) {
      return absl::InternalError("not equal");
    }
  }
  return absl::OkStatus();
}

template absl::Status ReduceProductChannelsUIntTest<DataType::INT32>(
    TestExecutionEnvironment* env);
template absl::Status ReduceProductChannelsUIntTest<DataType::INT16>(
    TestExecutionEnvironment* env);
template absl::Status ReduceProductChannelsUIntTest<DataType::INT8>(
    TestExecutionEnvironment* env);
}  // namespace

absl::Status MeanHWTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 2, 1);
  src_tensor.data = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::set<tflite::gpu::Axis> axis{Axis::HEIGHT, Axis::WIDTH};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::MEAN, op_def,
                       env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 1, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({2.5f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

absl::Status ReduceSumChannelsTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 5);
  src_tensor.data = {1.1, 2.1, 0.7, 0.3, 1.2, 3.1, 4.1, 0.0, 1.0, 4.4};
  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_SUM,
                       op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 2, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({5.4f, 12.6f}, dst_tensor.data, eps));
    }
  }

  RETURN_IF_ERROR(ReduceSumChannelsIntTest<DataType::INT32>(env));
  RETURN_IF_ERROR(ReduceSumChannelsIntTest<DataType::INT16>(env));
  RETURN_IF_ERROR(ReduceSumChannelsIntTest<DataType::INT8>(env));
  return absl::OkStatus();
}

absl::Status ReduceProductChannelsTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 2);
  src_tensor.data = {1.1, 2.0, 3.1, 4.0};
  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_PRODUCT,
                       op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 2, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({2.2f, 12.4f}, dst_tensor.data, eps));
    }
  }

  RETURN_IF_ERROR(ReduceProductChannelsUIntTest<DataType::UINT32>(env));
  RETURN_IF_ERROR(ReduceProductChannelsUIntTest<DataType::UINT16>(env));
  RETURN_IF_ERROR(ReduceProductChannelsUIntTest<DataType::UINT8>(env));
  return absl::OkStatus();
}

absl::Status ReduceMaxChannelsTest(TestExecutionEnvironment* env) {
  const std::vector<int> num_channels = {6, 257};

  for (int channels : num_channels) {
    TensorFloat32 src_tensor;
    src_tensor.shape = BHWC(1, 2, 1, channels);
    src_tensor.data = std::vector<float>(2 * channels, -1000.0f);
    // Move the custom values to the end of the tensor to ensure masking works
    // correctly.
    std::vector<float> channel1 = {1.1, 2.0, -0.3, -100.0, 32.6, 1.1};
    std::vector<float> channel2 = {-3.1, -4.0, -5.0, -7.0, -2.0, -100.0};
    src_tensor.data.insert(src_tensor.data.begin() + (channels - 6),
                           channel1.begin(), channel1.end());
    src_tensor.data.insert(src_tensor.data.begin() + (2 * channels - 6),
                           channel2.begin(), channel2.end());

    const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

    for (auto precision : env->GetSupportedPrecisions()) {
      auto data_type = DeduceDataTypeFromPrecision(precision);
      for (auto storage : env->GetSupportedStorages(data_type)) {
        const float eps =
            precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
        OperationDef op_def;
        op_def.precision = precision;
        op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
        op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
        TensorFloat32 dst_tensor;
        Reduce operation =
            CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_MAXIMUM,
                         op_def, env->GetGpuInfo());
        RETURN_IF_ERROR(env->ExecuteGPUOperation(
            src_tensor, std::make_unique<Reduce>(std::move(operation)),
            BHWC(1, 2, 1, 1), &dst_tensor));
        RETURN_IF_ERROR(PointWiseNear({32.6f, -2.0f}, dst_tensor.data, eps));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ReduceMinChannelsTest(TestExecutionEnvironment* env) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 2, 1, 6);
  src_tensor.data = {1.1,  2.0,  -0.3, -100.0, 32.6, 1.1,
                     -3.1, -4.0, -5.0, -7.0,   -2.0, 100.0};
  const std::set<tflite::gpu::Axis> axis{Axis::CHANNELS};

  for (auto precision : env->GetSupportedPrecisions()) {
    auto data_type = DeduceDataTypeFromPrecision(precision);
    for (auto storage : env->GetSupportedStorages(data_type)) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-2f;
      OperationDef op_def;
      op_def.precision = precision;
      op_def.src_tensors.push_back({data_type, storage, Layout::HWC});
      op_def.dst_tensors.push_back({data_type, storage, Layout::HWC});
      TensorFloat32 dst_tensor;
      Reduce operation =
          CreateReduce(axis, src_tensor.shape, OperationType::REDUCE_MINIMUM,
                       op_def, env->GetGpuInfo());
      RETURN_IF_ERROR(env->ExecuteGPUOperation(
          src_tensor, std::make_unique<Reduce>(std::move(operation)),
          BHWC(1, 2, 1, 1), &dst_tensor));
      RETURN_IF_ERROR(PointWiseNear({-100.0f, -7.0f}, dst_tensor.data, eps));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
