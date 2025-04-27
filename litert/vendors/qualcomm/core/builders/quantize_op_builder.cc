// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/quantize_op_builder.h"

#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"

namespace qnn {

std::vector<OpWrapper> BuildQuantizeOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  QnnOpCode qnn_op_code = QnnOpCode::kQnnOpCodeQuantize;
  if (inputs[0].get().IsPerTensorQuantWithOffsetDiff(outputs[0].get())) {
    qnn_op_code = QnnOpCode::kQnnOpCodeCast;
  } else if ((inputs[0].get().IsQuant8() || inputs[0].get().IsQuant16()) &&
             (outputs[0].get().IsQuant8() || outputs[0].get().IsQuant16())) {
    qnn_op_code = QnnOpCode::kQnnOpCodeConvert;
  }

  auto& quantize_op = CreateOpWrapper(res, qnn_op_code);
  quantize_op.AddInputTensor(inputs[0]);
  quantize_op.AddOutputTensor(outputs[0]);

  return res;
}

std::vector<OpWrapper> BuildDequantizeOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;
  QnnOpCode qnn_op_code = QnnOpCode::kQnnOpCodeDequantize;
  if (inputs[0].get().IsF16() && outputs[0].get().IsF32()) {
    qnn_op_code = QnnOpCode::kQnnOpCodeCast;
  }

  auto& quantize_op = CreateOpWrapper(res, qnn_op_code);
  quantize_op.AddInputTensor(inputs[0]);
  quantize_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
