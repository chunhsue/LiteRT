// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "litert/vendors/qualcomm/core/builders/convert_uint16_to_int16_builder.h"

#include <vector>
#include <variant>
#include "litert/vendors/qualcomm/core/builders/op_builder.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"

namespace qnn {

std::vector<OpWrapper> BuildConvertUint16ToInt16(
    TensorPool& tensor_pool, TensorWrapper* input, TensorWrapper** new_input) {
  std::vector<OpWrapper> res;

  // auto& op = CreateOpWrapper(res, QNN_OP_CAST);
  // op.AddInputTensor(*input);
  TensorWrapper& new_output = tensor_pool.CreateNativeTensor(
    input->GetDataType(), input->GetQuantParams(), input->GetDims());
  // op.AddOutputTensor(new_output);

  TensorWrapper& geq_output = tensor_pool.CreateNativeTensor(
    QNN_DATATYPE_BOOL_8, {}, input->GetDims());
  TensorWrapper& sub_output = tensor_pool.CreateNativeTensor(
    input->GetDataType(), input->GetQuantParams(), input->GetDims());
  TensorWrapper& add_output = tensor_pool.CreateNativeTensor(
    input->GetDataType(), input->GetQuantParams(), input->GetDims());
  const std::vector<uint32_t> dims = {1};
  const std::array<std::uint16_t, 1> const_data = {32768};
  QuantizeParamsWrapperVariant quant_param;
  ScaleOffsetQuantizeParamsWrapper* inp_quant_param = std::get_if<ScaleOffsetQuantizeParamsWrapper>(&input->GetQuantParams());
  quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(inp_quant_param->GetScale(), inp_quant_param->GetZeroPoint() + kUint16ZeroPoint);

  auto& const_tensor = tensor_pool.CreateStaticTensor(QNN_DATATYPE_UFIXED_POINT_16, quant_param, dims,
    sizeof(std::uint16_t) * dims.size(),
    const_data.data());
  
  auto& geq_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_GREATER_EQUAL);
  geq_op.AddInputTensor(*input);
  geq_op.AddInputTensor(const_tensor);
  geq_op.AddOutputTensor(geq_output);

  auto& sub_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SUBTRACT);
  sub_op.AddInputTensor(*input);
  sub_op.AddInputTensor(const_tensor);
  sub_op.AddOutputTensor(sub_output);

  auto& add_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_ADD);
  add_op.AddInputTensor(*input);
  add_op.AddInputTensor(const_tensor);
  add_op.AddOutputTensor(add_output);

  auto& select_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SELECT);
  select_op.AddInputTensor(geq_output);
  select_op.AddInputTensor(sub_output);
  select_op.AddInputTensor(add_output);
  select_op.AddOutputTensor(new_output);

  *new_input = &new_output;

  return res;
}

}  // namespace qnn
