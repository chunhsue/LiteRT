load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application", "ios_unit_test")
load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "workspace_root")
load(
    "@org_tensorflow//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("//tflite:build_def.bzl", "CXX17_BAZEL_ONLY_COPTS")
load(
    "//tflite:special_rules.bzl",
    "tflite_ios_lab_runner",
    "tflite_ios_per_kernel_test",
    "tflite_portable_test_suite",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

DEFAULT_COPTS = [
    "-Wno-shorten-64-to-32",
] + CXX17_BAZEL_ONLY_COPTS

objc_library(
    name = "buffer",
    srcs = ["buffer.cc"],
    hdrs = ["buffer.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    sdk_frameworks = ["Metal"],
    deps = [
        ":gpu_object",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common/task:buffer_desc",
        "@com_google_absl//absl/types:span",
    ],
)

objc_library(
    name = "buffer_test_lib",
    testonly = 1,
    srcs = ["buffer_test.mm"],
    sdk_frameworks = [
        "Metal",
        "XCTest",
    ],
    deps = [
        ":buffer",
        "//tflite/delegates/gpu/common:types",
    ],
)

objc_library(
    name = "buffer_convert",
    srcs = ["buffer_convert.mm"],
    hdrs = ["buffer_convert.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = ["Metal"],
    deps = [
        ":common",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:util",
    ],
)

objc_library(
    name = "common",
    srcs = ["common.mm"],
    hdrs = ["common.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = ["Metal"],
    deps = [
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "common_test_lib",
    testonly = 1,
    srcs = ["common_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [":common"],
)

ios_unit_test(
    name = "common_test",
    testonly = 1,
    minimum_os_version = "12.0",
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":common_test_lib"],
)

objc_library(
    name = "compute_task",
    srcs = ["compute_task.cc"],
    hdrs = ["compute_task.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    sdk_frameworks = ["Metal"],
    deps = [
        ":common",
        ":metal_arguments",
        ":metal_device",
        ":metal_spatial_tensor",
        "//tflite/delegates/gpu/common:kernel_info",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/task:tuning_type",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "gpu_object",
    hdrs = ["gpu_object.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = ["Metal"],
    deps = [
        "//tflite/delegates/gpu/common:access_type",
        "//tflite/delegates/gpu/common:data_type",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common/task:gpu_object_desc",
    ],
)

objc_library(
    name = "inference_context",
    srcs = ["inference_context.cc"],
    hdrs = ["inference_context.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    sdk_frameworks = ["Metal"],
    deps = [
        ":compute_task",
        ":inference_context_cc_fbs",
        ":metal_device",
        ":metal_spatial_tensor",
        "//tflite/delegates/gpu/common:gpu_model",
        "//tflite/delegates/gpu/common:gpu_model_cc_fbs",
        "//tflite/delegates/gpu/common:memory_management",
        "//tflite/delegates/gpu/common:model",
        "//tflite/delegates/gpu/common:model_hints",
        "//tflite/delegates/gpu/common:operations",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/common/selectors:operation_selector",
        "//tflite/delegates/gpu/common/selectors:special_selector",
        "//tflite/delegates/gpu/common/selectors:subgraph",
        "//tflite/delegates/gpu/common/task:profiling_info",
        "//tflite/delegates/gpu/common/task:serialization_base",
        "//tflite/delegates/gpu/common/task:tuning_type",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

flatbuffer_cc_library(
    name = "inference_context_cc_fbs",
    srcs = ["inference_context.fbs"],
    flatc_args = [
        "--scoped-enums",
        "-I " + workspace_root,
    ],
    includes = [
        "//tflite/delegates/gpu/common:gpu_model_cc_fbs_includes",
        "//tflite/delegates/gpu/common/task:serialization_base_cc_fbs_includes",
    ],
)

objc_library(
    name = "metal_arguments",
    srcs = ["metal_arguments.cc"],
    hdrs = ["metal_arguments.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    sdk_frameworks = ["Metal"],
    deps = [
        ":buffer",
        ":gpu_object",
        ":metal_device",
        ":metal_spatial_tensor",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/common/task:arguments",
        "//tflite/delegates/gpu/common/task:gpu_object_desc",
        "//tflite/delegates/gpu/common/task:util",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "metal_device",
    srcs = ["metal_device.cc"],
    hdrs = ["metal_device.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    sdk_frameworks = ["Metal"],
    deps = ["//tflite/delegates/gpu/common:gpu_info"],
)

objc_library(
    name = "metal_spatial_tensor",
    srcs = ["metal_spatial_tensor.cc"],
    hdrs = ["metal_spatial_tensor.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    sdk_frameworks = ["Metal"],
    deps = [
        ":common",
        ":gpu_object",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/common/task:buffer_desc",
        "//tflite/delegates/gpu/common/task:gpu_tensor",
        "//tflite/delegates/gpu/common/task:tensor_desc",
    ],
)

objc_library(
    name = "metal_spatial_tensor_test_lib",
    testonly = 1,
    srcs = ["metal_spatial_tensor_test.mm"],
    sdk_frameworks = [
        "Metal",
        "XCTest",
    ],
    deps = [
        ":metal_spatial_tensor",
        "//tflite/delegates/gpu/common:types",
    ],
)

objc_library(
    name = "TestBinary",
    testonly = 1,
    srcs = ["host.m"],
    sdk_frameworks = ["UIKit"],
)

# Application for local testing in XCode
ios_application(
    name = "TestApplication",
    testonly = 1,
    bundle_id = "com.tensorflow.lite.delegates.gpu.metal.ComponentsTests",
    families = [
        "iphone",
    ],
    infoplists = ["Info.plist"],
    minimum_os_version = "12.0",
    provisioning_profile = "//tflite/delegates/gpu/metal:provisioning_profile.mobileprovision",
    tags = tf_gpu_tests_tags() + [
        "local",
        "notap",
    ],
    deps = [":TestBinary"],
)

objc_library(
    name = "common_tests_lib",
    testonly = 1,
    srcs = [
        "//tflite/delegates/gpu/metal:buffer_test.mm",
        "//tflite/delegates/gpu/metal:common_test.mm",
        "//tflite/delegates/gpu/metal:metal_spatial_tensor_test.mm",
    ],
    hdrs = [
    ],
    sdk_frameworks = ["XCTest"],
    deps = [
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/metal:buffer",
        "//tflite/delegates/gpu/metal:common",
        "//tflite/delegates/gpu/metal:metal_spatial_tensor",
    ],
)

ios_unit_test(
    name = "ComponentsTests",
    families = ["iphone"],
    minimum_os_version = "12.0",
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = tf_gpu_tests_tags() + ["notap"],
    test_host = ":TestApplication",
    deps = [
        ":common_tests_lib",
        "//tflite/delegates/gpu/metal/kernels:kernel_tests_lib",
    ],
)

tflite_portable_test_suite()

tflite_ios_per_kernel_test()

cc_library(
    name = "compute_task_cc",
    srcs = ["compute_task.cc"],
    hdrs = ["compute_task.h"],
    deps = [
        ":common",
        ":metal_arguments",
        ":metal_device",
        ":metal_spatial_tensor",
        "//tflite/delegates/gpu/common:kernel_info",
        "//tflite/delegates/gpu/common:precision",
        "//tflite/delegates/gpu/common:shape",
        "//tflite/delegates/gpu/common:status",
        "//tflite/delegates/gpu/common:types",
        "//tflite/delegates/gpu/common:util",
        "//tflite/delegates/gpu/common/task:gpu_operation",
        "//tflite/delegates/gpu/common/task:tuning_type",
        "@com_google_absl//absl/strings",
    ],
)
