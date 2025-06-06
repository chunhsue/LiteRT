load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//tflite:build_def.bzl", "tflite_copts")
load("//tflite/tools/evaluation/tasks:build_def.bzl", "task_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "dummy_delegate",
    srcs = [
        "dummy_delegate.cc",
    ],
    hdrs = [
        "dummy_delegate.h",
    ],
    deps = [
        "//tflite/c:c_api_types",
        "//tflite/c:common",
        "//tflite/core/c:common",
        "//tflite/delegates/utils:simple_delegate",
    ],
)

cc_binary(
    name = "dummy_external_delegate.so",
    srcs = [
        "external_delegate_adaptor.cc",
    ],
    defines = ["TFL_EXTERNAL_DELEGATE_COMPILE_LIBRARY"],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":dummy_delegate",
        "//tflite/core/c:common",
        "//tflite/delegates/external:external_delegate_interface",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:logging",
    ],
)

#### The following are for using the dummy test delegate in TFLite tooling ####
cc_library(
    name = "dummy_delegate_provider",
    srcs = ["dummy_delegate_provider.cc"],
    copts = tflite_copts(),
    deps = [
        ":dummy_delegate",
        "//tflite/c:c_api_types",
        "//tflite/tools:command_line_flags",
        "//tflite/tools:tool_params",
        "//tflite/tools/delegates:delegate_provider_hdr",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "benchmark_model_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tflite/tools/benchmark:benchmark_model_main",
    ],
)

cc_binary(
    name = "inference_diff_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tflite/tools/evaluation/tasks:task_executor_main",
        "//tflite/tools/evaluation/tasks/inference_diff:run_eval_lib",
    ],
)

cc_binary(
    name = "imagenet_classification_eval_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tflite/tools/evaluation/tasks:task_executor_main",
        "//tflite/tools/evaluation/tasks/imagenet_image_classification:run_eval_lib",
    ],
)

cc_binary(
    name = "coco_object_detection_eval_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tflite/tools/evaluation/tasks:task_executor_main",
        "//tflite/tools/evaluation/tasks/coco_object_detection:run_eval_lib",
    ],
)

sh_test(
    name = "external_delegate_test",
    srcs = ["external_delegate_test.sh"],
    data = [
        "//tflite/delegates/utils/dummy_delegate:dummy_external_delegate.so",
        "//tflite/tools/benchmark:benchmark_model",
        "@tflite_mobilenet_float//:mobilenet_v1_1.0_224.tflite",
    ],
    visibility = ["//visibility:private"],
)
