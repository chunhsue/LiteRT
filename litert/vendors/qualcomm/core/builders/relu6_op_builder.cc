// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"

namespace qnn {

std::vector<OpWrapper> BuildRelu6Op(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  // QNN_OP_RELU6 is deprecated, use QNN_OP_RELU_MIN_MAX instead.
  auto& activation_op = CreateOpWrapper(res, QNN_OP_RELU_MIN_MAX);
  activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE, 0);
  activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE, 6);
  activation_op.AddInputTensor(inputs[0]);
  activation_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
