// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/qkv.h"

#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
namespace qnn {
namespace {
constexpr size_t kFCIndex = 0;
// constexpr size_t kReshape1Index = 1;
constexpr size_t kReshape2Index = 2;
// constexpr size_t kSliceQIndex = 3;
constexpr size_t kSliceKIndex = 4;
constexpr size_t kSliceVIndex = 5;
constexpr size_t kQRmsNormIndex = 6;
// constexpr size_t kKRmsNormIndex = 7;
// constexpr size_t kQSlice1Index = 8;
// constexpr size_t kQSlice2Index = 9;
constexpr size_t kConcatIndex = 10;
constexpr size_t kMulCosIndex = 11;
constexpr size_t kMulSinIndex = 12;
constexpr size_t kAddIndex = 13;
constexpr size_t kMHAMulIndex = 25;
constexpr size_t kMHAOffset = 11;
constexpr size_t kNumHeads = 4;
constexpr size_t kNumKVHeads = 1;

void EmplaceOpWithIO(
    std::vector<OpWrapper>& new_ops, const OpWrapper& source_op,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& inputs,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& outputs) {
  OpWrapper ret = source_op;
  ret.UpdateTensors(inputs, outputs);
  new_ops.emplace_back(ret);
}

}  // namespace
size_t SplitQKV(std::function<bool(OpWrapper&)> validate_op_config,
                std::vector<OpWrapper>& ops, size_t start_index,
                TensorPool& tensor_pool, size_t pattern_size) {
  std::vector<OpWrapper> new_ops;
  // 3 FC ops for QKV Projection
  QNN_LOG_INFO("Found QKV");
  const auto& filter_tensor = ops[start_index + kFCIndex].GetInputTensor(1);
  auto filter = filter_tensor.GetStaticTensorData<int8_t>();
  auto filter_data = filter.value();
  QNN_LOG_INFO("Fliter Size: %d", filter_data.size());
  const size_t num_total_heads = kNumHeads + kNumKVHeads * 2;
  const size_t head_size = filter_tensor.GetDim(0) / num_total_heads;

  std::vector<int8_t> k_filter(
      filter_data.begin() + head_size * kNumHeads * filter_tensor.GetDim(1),
      filter_data.begin() +
          head_size * (kNumHeads + kNumKVHeads) * filter_tensor.GetDim(1));
  auto& k_filter_tensor = tensor_pool.CreateStaticTensor(
      filter_tensor.GetDataType(), filter_tensor.GetQuantParams(),
      {static_cast<uint32_t>(head_size * kNumKVHeads), filter_tensor.GetDim(1)},
      k_filter.size() * sizeof(k_filter[0]), k_filter.data());
  std::vector<int8_t> v_filter(
      filter_data.begin() +
          head_size * (kNumHeads + kNumKVHeads) * filter_tensor.GetDim(1),
      filter_data.end());
  auto& v_filter_tensor = tensor_pool.CreateStaticTensor(
      filter_tensor.GetDataType(), filter_tensor.GetQuantParams(),
      {static_cast<uint32_t>(head_size * kNumKVHeads), filter_tensor.GetDim(1)},
      v_filter.size() * sizeof(v_filter[0]), v_filter.data());

  QNN_LOG_INFO("K Fliter Size: %d", k_filter.size());
  QNN_LOG_INFO("V Fliter Size: %d", v_filter.size());

  auto& fc_output = ops[start_index + kFCIndex].GetOutputTensor(0);

  auto& k = tensor_pool.CloneNativeTensorFrom(
      fc_output,
      {fc_output.GetDim(0), static_cast<uint32_t>(head_size * kNumKVHeads)});
  auto& v = tensor_pool.CloneNativeTensorFrom(
      fc_output,
      {fc_output.GetDim(0), static_cast<uint32_t>(head_size * kNumKVHeads)});

  EmplaceOpWithIO(new_ops, ops[start_index + kFCIndex],
                  {std::nullopt, k_filter_tensor}, {k});
  EmplaceOpWithIO(new_ops, ops[start_index + kFCIndex],
                  {std::nullopt, v_filter_tensor}, {v});
  // 2 Reshape for KV
  EmplaceOpWithIO(new_ops, ops[start_index + kReshape2Index], {k},
                  {const_cast<TensorWrapper&>(
                      ops[start_index + kSliceKIndex].GetOutputTensor(0))});
  EmplaceOpWithIO(new_ops, ops[start_index + kReshape2Index], {v},
                  {const_cast<TensorWrapper&>(
                      ops[start_index + kSliceVIndex].GetOutputTensor(0))});
  // Q
  std::vector<TensorWrapper*> q_tensors;
  q_tensors.reserve(kNumHeads);
  for (int i = 1; i <= kNumHeads; ++i) {
    std::vector<int8_t> q_filter(
        filter_data.begin() + head_size * (i - 1) * filter_tensor.GetDim(1),
        filter_data.begin() + head_size * i * filter_tensor.GetDim(1));
    auto& q_filter_tensor = tensor_pool.CreateStaticTensor(
        filter_tensor.GetDataType(), filter_tensor.GetQuantParams(),
        {static_cast<uint32_t>(head_size), filter_tensor.GetDim(1)},
        q_filter.size() * sizeof(q_filter[0]), q_filter.data());
    QNN_LOG_INFO("Q Fliter Size: %d", q_filter.size());
    auto& q = tensor_pool.CloneNativeTensorFrom(
        fc_output, {fc_output.GetDim(0), static_cast<uint32_t>(head_size)});
    EmplaceOpWithIO(new_ops, ops[start_index + kFCIndex],
                    {std::nullopt, q_filter_tensor}, {q});
    QNN_LOG_INFO("Reshape %d", ops[start_index + kReshape2Index].GetOpCode());
    auto& q_rmsnorm_input =
        tensor_pool.CloneNativeTensorFrom(q, {1, 1, q.GetDim(0), q.GetDim(1)});
    q_tensors.emplace_back(&q_rmsnorm_input);
    EmplaceOpWithIO(new_ops, ops[start_index + kReshape2Index], {q},
                    {q_rmsnorm_input});
    QNN_LOG_INFO("RMSNorm %d", ops[start_index + kQRmsNormIndex].GetOpCode());
    auto& q_rmsnorm_output = tensor_pool.CloneNativeTensorFrom(q_rmsnorm_input);
    EmplaceOpWithIO(new_ops, ops[start_index + kQRmsNormIndex],
                    {q_rmsnorm_input, std::nullopt, std::nullopt},
                    {q_rmsnorm_output});
    const std::array<int32_t, 1> split_axis_data{3};
    auto& split_axis = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_INT_32, {}, {split_axis_data.size()},
        split_axis_data.size() * sizeof(split_axis_data[0]),
        split_axis_data.data());
    auto split_out_dims = q_rmsnorm_input.GetDims();
    split_out_dims[3] = split_out_dims[3] / 2;
    const uint32_t num_split = 2;
    auto& q_split_output1 =
        tensor_pool.CloneNativeTensorFrom(q_rmsnorm_input, split_out_dims);
    auto& q_split_output2 =
        tensor_pool.CloneNativeTensorFrom(q_rmsnorm_input, split_out_dims);
    auto split = BuildSplitOp(tensor_pool, {split_axis, q_rmsnorm_output},
                              {q_split_output1, q_split_output2}, num_split);
    std::move(split.begin(), split.end(), std::back_inserter(new_ops));
    QNN_LOG_INFO("Concat %d", ops[start_index + kConcatIndex].GetOpCode());
    auto& q_concat_output = tensor_pool.CloneNativeTensorFrom(q_rmsnorm_input);
    EmplaceOpWithIO(new_ops, ops[start_index + kConcatIndex],
                    {q_split_output2, q_split_output1}, {q_concat_output});
    // Additional reshape.
    const auto& cos_tensor = ops[start_index + kMulCosIndex].GetInputTensor(1);
    auto& q_mul_cos_input = tensor_pool.CloneNativeTensorFrom(q_concat_output);
    auto reshape_cos = BuildReshapeOp(
        tensor_pool, {const_cast<::qnn::TensorWrapper&>(cos_tensor)},
        {q_mul_cos_input});
    std::move(reshape_cos.begin(), reshape_cos.end(),
              std::back_inserter(new_ops));
    QNN_LOG_INFO("MulCos %d", ops[start_index + kMulCosIndex].GetOpCode());
    auto& q_mul_cos_output = tensor_pool.CloneNativeTensorFrom(q_concat_output);
    EmplaceOpWithIO(new_ops, ops[start_index + kMulCosIndex],
                    {q_rmsnorm_output, q_mul_cos_input}, {q_mul_cos_output});
    // Additional reshape.
    const auto& sin_tensor = ops[start_index + kMulSinIndex].GetInputTensor(1);
    auto& q_mul_sin_input = tensor_pool.CloneNativeTensorFrom(q_concat_output);
    auto reshape_sin = BuildReshapeOp(
        tensor_pool, {const_cast<::qnn::TensorWrapper&>(sin_tensor)},
        {q_mul_sin_input});
    std::move(reshape_sin.begin(), reshape_sin.end(),
              std::back_inserter(new_ops));
    QNN_LOG_INFO("MulSin %d", ops[start_index + kMulSinIndex].GetOpCode());
    auto& q_mul_sin_output = tensor_pool.CloneNativeTensorFrom(q_concat_output);
    EmplaceOpWithIO(new_ops, ops[start_index + kMulSinIndex],
                    {q_concat_output, q_mul_sin_input}, {q_mul_sin_output});
    QNN_LOG_INFO("Add %d abs %d", ops[start_index + kAddIndex].GetOpCode(),
                 start_index + kAddIndex);
    for (int k = 0; k < 60; ++k) {
      QNN_LOG_INFO("What %d abs %d",
                   ops[start_index + kAddIndex + k].GetOpCode(),
                   start_index + kAddIndex + k);
    }
    auto& q_add_output = tensor_pool.CloneNativeTensorFrom(q_mul_sin_output);
    EmplaceOpWithIO(new_ops, ops[start_index + kAddIndex],
                    {q_mul_cos_output, q_mul_sin_output}, {q_add_output});
    // Connect add output to each SHA input.
    QNN_LOG_INFO(
        "MatMul %d abs %d",
        ops[start_index + kMHAMulIndex + kMHAOffset * (i - 1)].GetOpCode(),
        start_index + kMHAMulIndex + kMHAOffset * (i - 1));
    EmplaceOpWithIO(new_ops,
                    ops[start_index + kMHAMulIndex + kMHAOffset * (i - 1)],
                    {q_add_output, std::nullopt}, {std::nullopt});
  }

  // Validate new graph.
  // TODO(jiunkaiy): Disable bypassing Split int16 op validator.
  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
                    return op_wrapper.IsOpCode(QnnOpCode::kSplit) ||
                           op_wrapper.IsOpCode(QnnOpCode::kRmsNorm) ||
                           validate_op_config(op_wrapper);
                  });
  if (is_valid) {
    size_t step_size = new_ops.size();
    // Erase Transpose->Reshape->Split->ElementWiseMul*4.
    for (int i = 3; i >= 0; i--) {
      ops.erase(ops.begin() + start_index + kMHAMulIndex + kMHAOffset * i);
    }
    ops.erase(ops.begin() + start_index + kMHAMulIndex - 3,
              ops.begin() + start_index + kMHAMulIndex);
    // Replace the matched pattern with a newly generated subgraph.
    ops.insert(ops.begin() + start_index + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    ops.erase(ops.begin() + start_index,
              ops.begin() + start_index + pattern_size);

    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}
}  // namespace qnn
