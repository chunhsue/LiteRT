// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/c/litert_logging.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/cc/litert_any.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_options.h"
#include "litert/core/filesystem.h"
#include "litert/runtime/tensor_buffer_requirements.h"
#include "litert/test/common.h"
#include "litert/test/matchers.h"
#include "litert/test/testdata/simple_model_test_vectors.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/intel_openvino/dispatch/device_context.h"

constexpr absl::string_view kDispatchLibraryDir = "/data/local/tmp";

litert::Expected<litert::Environment> CreateDefaultEnvironment() {
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  return litert::Environment::Create(absl::MakeConstSpan(environment_options));
}

TEST(OpenVino, DispatchApi) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, CreateDefaultEnvironment());
  LITERT_ASSERT_OK_AND_ASSIGN(auto env_options, env.GetOptions());
  LITERT_ASSERT_OK_AND_ASSIGN(auto options, ::litert::Options::Create());

  ASSERT_EQ(LiteRtDispatchInitialize(env_options.Get(), options.Get()),
            kLiteRtStatusOk);

  const char* vendor_id;
  EXPECT_EQ(LiteRtDispatchGetVendorId(&vendor_id), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "vendor_id: " << vendor_id;

  const char* build_id;
  EXPECT_EQ(LiteRtDispatchGetBuildId(&build_id), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "build_id " << build_id;

  LiteRtApiVersion api_version;
  EXPECT_EQ(LiteRtDispatchGetApiVersion(&api_version), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "api_version: " << api_version.major << "."
                 << api_version.minor << "." << api_version.patch;

  int capabilities;
  EXPECT_EQ(LiteRtDispatchGetCapabilities(&capabilities), kLiteRtStatusOk);
  ABSL_LOG(INFO) << "capabilities: " << capabilities;

  LiteRtDispatchDeviceContext device_context = nullptr;
  EXPECT_EQ(LiteRtDispatchDeviceContextCreate(&device_context),
            kLiteRtStatusOk);
  EXPECT_NE(device_context, nullptr);

  auto model_file_name =
      litert::testing::GetTestFilePath(kOpenvinoModelBlobFileName);

  ABSL_LOG(INFO) << "Model file is " << model_file_name.c_str();
  auto model = litert::internal::LoadBinaryFile(model_file_name);
  EXPECT_TRUE(model) << model.Error();

  LiteRtMemBuffer exec_bytecode_buffer = {/*.fd=*/-1,
                                          /*.base_addr=*/model->Data(),
                                          /*.offset=*/0,
                                          /*.size=*/model->Size()};
  LiteRtDispatchInvocationContext invocation_context = nullptr;
  EXPECT_EQ(LiteRtDispatchInvocationContextCreate(
                device_context, kLiteRtDispatchExecutableTypeMlModel,
                &exec_bytecode_buffer, /*function_name=*/nullptr,
                /*num_inputs=*/2, /*num_outputs=*/1, &invocation_context),
            kLiteRtStatusOk);
  EXPECT_NE(invocation_context, nullptr);

  // ///////////////////////////////////////////////////////////////////////////
  // Determine tensor buffer requirements.
  // ///////////////////////////////////////////////////////////////////////////

  int num_tensor_buffer_types;
  LiteRtTensorBufferRequirements input_0_tensor_buffer_requirements;
  EXPECT_EQ(LiteRtDispatchGetInputRequirements(
                invocation_context, /*input_index=*/0, &kInput0TensorType,
                &input_0_tensor_buffer_requirements),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                input_0_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLiteRtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 2);
  LiteRtTensorBufferType input_0_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                input_0_tensor_buffer_requirements, /*type_index=*/1,
                &input_0_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(input_0_tensor_buffer_type, kLiteRtTensorBufferTypeDmaBuf);
  size_t input_0_tensor_buffer_size;
  EXPECT_EQ(
      LiteRtGetTensorBufferRequirementsBufferSize(
          input_0_tensor_buffer_requirements, &input_0_tensor_buffer_size),
      kLiteRtStatusOk);
  EXPECT_GE(input_0_tensor_buffer_size, sizeof(kTestInput0Tensor));

  LiteRtTensorBufferRequirements input_1_tensor_buffer_requirements;
  EXPECT_EQ(LiteRtDispatchGetInputRequirements(
                invocation_context, /*input_index=*/0, &kInput1TensorType,
                &input_1_tensor_buffer_requirements),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                input_1_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLiteRtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 2);
  LiteRtTensorBufferType input_1_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                input_1_tensor_buffer_requirements, /*type_index=*/1,
                &input_1_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(input_1_tensor_buffer_type, kLiteRtTensorBufferTypeDmaBuf);
  size_t input_1_tensor_buffer_size;
  EXPECT_EQ(
      LiteRtGetTensorBufferRequirementsBufferSize(
          input_1_tensor_buffer_requirements, &input_1_tensor_buffer_size),
      kLiteRtStatusOk);
  EXPECT_GE(input_1_tensor_buffer_size, sizeof(kTestInput1Tensor));

  LiteRtTensorBufferRequirements output_tensor_buffer_requirements;
  EXPECT_EQ(LiteRtDispatchGetOutputRequirements(
                invocation_context, /*output_index=*/0, &kOutputTensorType,
                &output_tensor_buffer_requirements),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(
                output_tensor_buffer_requirements, &num_tensor_buffer_types),
            kLiteRtStatusOk);
  EXPECT_GE(num_tensor_buffer_types, 2);
  LiteRtTensorBufferType output_tensor_buffer_type;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(
                output_tensor_buffer_requirements, /*type_index=*/1,
                &output_tensor_buffer_type),
            kLiteRtStatusOk);
  EXPECT_EQ(output_tensor_buffer_type, kLiteRtTensorBufferTypeDmaBuf);
  size_t output_tensor_buffer_size;
  EXPECT_EQ(LiteRtGetTensorBufferRequirementsBufferSize(
                output_tensor_buffer_requirements, &output_tensor_buffer_size),
            kLiteRtStatusOk);
  EXPECT_GE(output_tensor_buffer_size, sizeof(kTestOutputTensor));

  // ///////////////////////////////////////////////////////////////////////////
  // Allocate tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////

  LiteRtTensorBuffer input_0_tensor_buffer;
  EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                env.Get(), input_0_tensor_buffer_type, &kInput0TensorType,
                input_0_tensor_buffer_size, &input_0_tensor_buffer),
            kLiteRtStatusOk);

  LiteRtTensorBuffer input_1_tensor_buffer;
  EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                env.Get(), input_1_tensor_buffer_type, &kInput1TensorType,
                input_1_tensor_buffer_size, &input_1_tensor_buffer),
            kLiteRtStatusOk);
  LiteRtTensorBuffer output_tensor_buffer;
  EXPECT_EQ(LiteRtCreateManagedTensorBuffer(
                env.Get(), output_tensor_buffer_type, &kOutputTensorType,
                output_tensor_buffer_size, &output_tensor_buffer),
            kLiteRtStatusOk);
  // ///////////////////////////////////////////////////////////////////////////
  // Register tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////
  LiteRtTensorBufferHandle input_0_handle;
  EXPECT_EQ(LiteRtDispatchRegisterTensorBuffer(
                device_context, input_0_tensor_buffer, &input_0_handle),
            kLiteRtStatusOk);

  LiteRtTensorBufferHandle input_1_handle;
  EXPECT_EQ(LiteRtDispatchRegisterTensorBuffer(
                device_context, input_1_tensor_buffer, &input_1_handle),
            kLiteRtStatusOk);

  LiteRtTensorBufferHandle output_handle;
  EXPECT_EQ(LiteRtDispatchRegisterTensorBuffer(
                device_context, output_tensor_buffer, &output_handle),
            kLiteRtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Attach tensor buffers.
  // ///////////////////////////////////////////////////////////////////////////
  EXPECT_EQ(LiteRtDispatchAttachInput(invocation_context,
                                      /*graph_input_index=*/0, input_0_handle),
            kLiteRtStatusOk);

  EXPECT_EQ(LiteRtDispatchAttachInput(invocation_context,
                                      /*graph_input_index=*/1, input_1_handle),
            kLiteRtStatusOk);

  EXPECT_EQ(LiteRtDispatchAttachOutput(invocation_context,
                                       /*graph_output_index=*/0, output_handle),
            kLiteRtStatusOk);
  // ///////////////////////////////////////////////////////////////////////////
  // Fill the input buffers with data.
  // ///////////////////////////////////////////////////////////////////////////

  ABSL_LOG(INFO) << "Filling inputs with data";
  void* host_mem_addr;

  ASSERT_EQ(LiteRtLockTensorBuffer(input_0_tensor_buffer, &host_mem_addr,
                                   kLiteRtTensorBufferLockModeWrite),
            kLiteRtStatusOk);
  std::memcpy(host_mem_addr, kTestInput0Tensor, sizeof(kTestInput0Tensor));
  ASSERT_EQ(LiteRtUnlockTensorBuffer(input_0_tensor_buffer), kLiteRtStatusOk);

  ASSERT_EQ(LiteRtLockTensorBuffer(input_1_tensor_buffer, &host_mem_addr,
                                   kLiteRtTensorBufferLockModeWrite),
            kLiteRtStatusOk);
  std::memcpy(host_mem_addr, kTestInput1Tensor, sizeof(kTestInput1Tensor));
  ASSERT_EQ(LiteRtUnlockTensorBuffer(input_1_tensor_buffer), kLiteRtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Execute model.
  // ///////////////////////////////////////////////////////////////////////////

  ABSL_LOG(INFO) << "Invoking execution...";
  EXPECT_EQ(LiteRtDispatchInvoke(invocation_context), kLiteRtStatusOk);

  // ///////////////////////////////////////////////////////////////////////////
  // Check output for correctness.
  // ///////////////////////////////////////////////////////////////////////////

  {
    ABSL_LOG(INFO) << "Checking output...";
    void* host_mem_addr;
    ASSERT_EQ(LiteRtLockTensorBuffer(output_tensor_buffer, &host_mem_addr,
                                     kLiteRtTensorBufferLockModeRead),
              kLiteRtStatusOk);
    auto output = absl::MakeSpan(static_cast<const float*>(host_mem_addr),
                                 kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << output[i] << "\t" << kTestOutputTensor[i];
    }

    EXPECT_THAT(output, ::testing::Pointwise(testing::FloatNear(1e-3),
                                             kTestOutputTensor));
    ASSERT_EQ(LiteRtUnlockTensorBuffer(output_tensor_buffer), kLiteRtStatusOk);
  }
  // ///////////////////////////////////////////////////////////////////////////
  // Clean up resources.
  // ///////////////////////////////////////////////////////////////////////////
  EXPECT_EQ(LiteRtDispatchDetachInput(invocation_context,
                                      /*graph_input_index=*/0, input_0_handle),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDispatchDetachInput(invocation_context,
                                      /*graph_input_index=*/1, input_1_handle),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDispatchDetachOutput(invocation_context,
                                       /*graph_output_index=*/0, output_handle),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDispatchUnregisterTensorBuffer(device_context, output_handle),
            kLiteRtStatusOk);
  EXPECT_EQ(
      LiteRtDispatchUnregisterTensorBuffer(device_context, input_1_handle),
      kLiteRtStatusOk);
  EXPECT_EQ(
      LiteRtDispatchUnregisterTensorBuffer(device_context, input_0_handle),
      kLiteRtStatusOk);
  LiteRtDestroyTensorBuffer(output_tensor_buffer);
  LiteRtDestroyTensorBuffer(input_1_tensor_buffer);
  LiteRtDestroyTensorBuffer(input_0_tensor_buffer);
  EXPECT_EQ(LiteRtDispatchInvocationContextDestroy(invocation_context),
            kLiteRtStatusOk);
  EXPECT_EQ(LiteRtDispatchDeviceContextDestroy(device_context),
            kLiteRtStatusOk);
}
