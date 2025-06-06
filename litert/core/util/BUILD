# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//litert/build_common:litert_build_defs.bzl", "cc_library_with_testonly_vis")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "@mediapipe//mediapipe/calculators/tensor:__subpackages__",
        "//litert:litert_internal_users",
    ],
)

cc_library(
    name = "to_string",
    hdrs = ["to_string.h"],
)

cc_library_with_testonly_vis(
    name = "flatbuffer_tools",
    srcs = ["flatbuffer_tools.cc"],
    hdrs = [
        "flatbuffer_tools.h",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_consts",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:filesystem",
        "//tflite:model_builder",
        "//tflite:stderr_reporter",
        "//tflite/schema:schema_fbs",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/compiler/mlir/lite:allocation",
        "@org_tensorflow//tensorflow/compiler/mlir/lite/core:model_builder_base",
    ],
)

cc_test(
    name = "flatbuffer_tools_test",
    srcs = ["flatbuffer_tools_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
        "//litert/test:tflite_test_data",
    ],
    deps = [
        ":flatbuffer_tools",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_macros",
        "//litert/test:common",
        "//litert/test:matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tensor_type_util",
    srcs = [
        "tensor_type_util.cc",
    ],
    hdrs = [
        "tensor_type_util.h",
    ],
    # TODO: Either move this to cc/ or refactor users. Then, remove this visibility.
    visibility = [
        "//litert:litert_public",
    ],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_model",
        "//litert/cc:litert_expected",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "tensor_type_util_test",
    srcs = ["tensor_type_util_test.cc"],
    deps = [
        ":tensor_type_util",
        "//litert/c:litert_model",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
