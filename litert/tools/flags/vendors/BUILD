# Copyright 2025 Google LLC.
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

# QUALCOMM #########################################################################################

cc_library(
    name = "qualcomm_flags",
    srcs = ["qualcomm_flags.cc"],
    hdrs = ["qualcomm_flags.h"],
    copts = [
        "-DINCLUDE_QUALCOMM_COMPILE_FLAGS",
        "-DINCLUDE_QUALCOMM_RUNTIME_FLAGS",
    ],
    deps = [
        "//litert/c/options:litert_qualcomm_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_qualcomm_options",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:string_view",
    ],
    alwayslink = 1,
)

cc_test(
    name = "qualcomm_flags_test",
    srcs = ["qualcomm_flags_test.cc"],
    copts = [
        "-DINCLUDE_QUALCOMM_COMPILE_FLAGS",
        "-DINCLUDE_QUALCOMM_RUNTIME_FLAGS",
    ],
    deps = [
        ":qualcomm_flags",
        "//litert/c/options:litert_qualcomm_options",
        "//litert/cc:litert_expected",
        "//litert/cc/options:litert_qualcomm_options",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

# GOOGLE TENSOR ####################################################################################

cc_library(
    name = "google_tensor_flags",
    srcs = ["google_tensor_flags.cc"],
    hdrs = ["google_tensor_flags.h"],
    copts = [
        "-DINCLUDE_GOOGLE_TENSOR_COMPILE_FLAGS",
        "-DINCLUDE_GOOGLE_TENSOR_RUNTIME_FLAGS",
    ],
    deps = [
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_google_tensor_options",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:string_view",
    ],
    alwayslink = 1,
)

cc_test(
    name = "google_tensor_flags_test",
    srcs = ["google_tensor_flags_test.cc"],
    copts = [
        "-DINCLUDE_GOOGLE_TENSOR_COMPILE_FLAGS",
        "-DINCLUDE_GOOGLE_TENSOR_RUNTIME_FLAGS",
    ],
    deps = [
        ":google_tensor_flags",
        "//litert/c/options:litert_google_tensor_options",
        "@com_google_googletest//:gtest_main",
    ],
)

# MEDIATEK ####################################################################################

cc_library(
    name = "mediatek_flags",
    srcs = ["mediatek_flags.cc"],
    hdrs = ["mediatek_flags.h"],
    copts = [
        "-DINCLUDE_MEDIATEK_COMPILE_FLAGS",
        "-DINCLUDE_MEDIATEK_RUNTIME_FLAGS",
    ],
    deps = [
        "//litert/c/options:litert_mediatek_options",
        "//litert/cc:litert_expected",
        "//litert/cc:litert_macros",
        "//litert/cc/internal:litert_mediatek_options",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:string_view",
    ],
    alwayslink = 1,
)

cc_test(
    name = "mediatek_flags_test",
    srcs = ["mediatek_flags_test.cc"],
    copts = [
        "-DINCLUDE_MEDIATEK_COMPILE_FLAGS",
        "-DINCLUDE_MEDIATEK_RUNTIME_FLAGS",
    ],
    deps = [
        ":mediatek_flags",
        "//litert/c/options:litert_mediatek_options",
        "//litert/cc:litert_expected",
        "//litert/cc/options:litert_mediatek_options",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:marshalling",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
