load("@org_tensorflow//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tflite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "types",
    srcs = ["types.cc"],
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
)

cc_library(
    name = "attribute_map",
    srcs = ["attribute_map.cc"],
    hdrs = ["attribute_map.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":types",
        "//tflite/core/async/interop:attribute_map_internal",
    ],
)

cc_test(
    name = "attribute_map_test",
    srcs = ["attribute_map_test.cc"],
    deps = [
        ":attribute_map",
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "types_test",
    srcs = ["types_test.cc"],
    deps = [
        ":types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "constants",
    srcs = ["constants.cc"],
    hdrs = ["constants.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        "//tflite/core/c:c_api_types",
    ],
)
