package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "owning_vector_ref",
    srcs = ["owning_vector_ref.cc"],
    hdrs = [
        "owning_vector_ref.h",
    ],
    deps = ["//tflite/experimental/ml_adjacent:lib"],
)

cc_test(
    name = "owning_vector_ref_test",
    srcs = ["owning_vector_ref_test.cc"],
    deps = [
        ":owning_vector_ref",
        "//tflite/experimental/ml_adjacent:lib",
        "@com_google_googletest//:gtest_main",
    ],
)
