// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"

#include <cstdint>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"

namespace qnn {

std::vector<OpWrapper> BuildElementwiseAddOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseAdd);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSubOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseSubtract);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseMulOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseMultiply);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseDivOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseDivide);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSinOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseSin);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseCosOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseCos);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseRsqrtOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseRsqrt);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSquareOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  OpWrapper& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseMultiply);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseSquaredDifferenceOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseSquaredDifference);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildElementwiseLessOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseBinary);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS);

  return res;
}

std::vector<OpWrapper> BuildElementwiseGreaterOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseBinary);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER);

  return res;
}

std::vector<OpWrapper> BuildElementwiseAndOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseBinary);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND);

  return res;
}

std::vector<OpWrapper> BuildElementwiseMinimumOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseBinary);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM);

  return res;
}

std::vector<OpWrapper> BuildElementwiseMaximumOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseBinary);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MAXIMUM);

  return res;
}

std::vector<OpWrapper> BuildElementwiseFloorDivOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseBinary);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[1]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FLOOR_DIV);

  return res;
}

std::vector<OpWrapper> BuildElementwiseNotEqualOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWiseBinary);
  elementwise_op.AddInputTensor(inputs[0]);
  elementwise_op.AddInputTensor(inputs[1]);
  elementwise_op.AddOutputTensor(outputs[0]);
  elementwise_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_BINARY_OPERATION_NOT_EQUAL);

  return res;
}

std::vector<OpWrapper> BuildElementwisePower(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& elementwise_op =
      CreateOpWrapper(res, QnnOpCode::kQnnOpCodeElementWisePower);
  for (const auto& input : inputs) {
    elementwise_op.AddInputTensor(input);
  }
  elementwise_op.AddOutputTensor(outputs[0]);
  return res;
}

}  // namespace qnn
