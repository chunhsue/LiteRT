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

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//litert:litert_internal_users"],
)

cc_library(
    name = "compiler_plugin",
    srcs = ["compiler_plugin.cc"],
    hdrs = ["compiler_plugin.h"],
    deps = [
        ":algo",
        "//litert/c:litert_any",
        "//litert/c:litert_common",
        "//litert/c:litert_environment_options",
        "//litert/c:litert_logging",
        "//litert/cc:litert_buffer_ref",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc:litert_op_options",
        "//litert/cc:litert_shared_library",
        "//litert/core:build_stamp",
        "//litert/core:dynamic_loading",
        "//litert/core:environment",
        "//litert/core:filesystem",
        "//litert/core:version",
        "//litert/core/model",
        "//litert/vendors/c:litert_compiler_plugin",
        "//litert/vendors/c:litert_compiler_plugin_api",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

# copybara:uncomment_begin(no OSS for unique-test-directory)
# cc_test(
#     name = "compiler_plugin_test",
#     srcs = ["compiler_plugin_test.cc"],
#     data = [
#         "//litert/test:mlir_test_data",
#         "//litert/vendors/examples:example_plugin_so",
#     ],
#     tags = [
#         # Sanitizer runtimes are incompatible with RTLD_DEEPBIND.
#         "noasan",
#         "nomsan",
#         "nosan",
#         "notsan",
#     ],
#     deps = [
#         ":compiler_plugin",
#         "@com_google_googletest//:gtest_main",
#         "@com_google_absl//absl/strings:string_view",
#         "//litert/c:litert_common",
#         "//litert/c:litert_model",
#         "//litert/c:litert_op_code",
#         "//litert/cc:litert_op_options",
#         "//litert/cc/internal:litert_environment",
#         "//litert/core:build_stamp",
#         "//litert/core:filesystem",
#         "//litert/core/model",
#         "//litert/test:common",
#         "//litert/test:matchers",
#         "//litert/tools:dump",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "algo",
    srcs = ["algo.cc"],
    hdrs = ["algo.h"],
    deps = [
        "//litert/c:litert_common",
        "//litert/c:litert_layout",
        "//litert/c:litert_logging",
        "//litert/cc:litert_detail",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/core:insert_order_map",
        "//litert/core/model",
        "//litert/core/model:model_graph",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
    ],
)

cc_test(
    name = "algo_test",
    srcs = ["algo_test.cc"],
    data = [
        "//litert/test:mlir_test_data",
    ],
    deps = [
        ":algo",
        "//litert/c:litert_common",
        "//litert/c:litert_op_code",
        "//litert/cc:litert_model",
        "//litert/cc:litert_model_predicates",
        "//litert/core/model",
        "//litert/core/model:graph_validation",
        "//litert/core/model:model_serialize",
        "//litert/test:common",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest_main",
    ],
)
