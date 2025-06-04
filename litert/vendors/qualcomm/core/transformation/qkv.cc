// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/qkv.h"

#include "litert/vendors/qualcomm/core/utils/log.h"

namespace qnn {
namespace {
constexpr size_t kFCIndex = 0;
// constexpr size_t kReshape1Index = 1;
constexpr size_t kReshape2Index = 2;
constexpr size_t kSliceQIndex = 3;
constexpr size_t kSliceKIndex = 4;
constexpr size_t kSliceVIndex = 5;
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
  std::vector<int8_t> q_filter(
      filter_data.begin(),
      filter_data.begin() + head_size * kNumHeads * filter_tensor.GetDim(1));
  auto& q_filter_tensor = tensor_pool.CreateStaticTensor(
      filter_tensor.GetDataType(), filter_tensor.GetQuantParams(),
      {static_cast<uint32_t>(head_size * kNumHeads), filter_tensor.GetDim(1)},
      q_filter.size() * sizeof(q_filter[0]), q_filter.data());
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

  QNN_LOG_INFO("Q Fliter Size: %d", q_filter.size());
  QNN_LOG_INFO("K Fliter Size: %d", k_filter.size());
  QNN_LOG_INFO("V Fliter Size: %d", v_filter.size());

  auto& fc_output = ops[start_index + kFCIndex].GetOutputTensor(0);
  auto& q = tensor_pool.CloneNativeTensorFrom(
      fc_output,
      {fc_output.GetDim(0), static_cast<uint32_t>(head_size * kNumHeads)});
  auto& k = tensor_pool.CloneNativeTensorFrom(
      fc_output,
      {fc_output.GetDim(0), static_cast<uint32_t>(head_size * kNumKVHeads)});
  auto& v = tensor_pool.CloneNativeTensorFrom(
      fc_output,
      {fc_output.GetDim(0), static_cast<uint32_t>(head_size * kNumKVHeads)});
  EmplaceOpWithIO(new_ops, ops[start_index + kFCIndex],
                  {std::nullopt, q_filter_tensor}, {q});
  EmplaceOpWithIO(new_ops, ops[start_index + kFCIndex],
                  {std::nullopt, k_filter_tensor}, {k});
  EmplaceOpWithIO(new_ops, ops[start_index + kFCIndex],
                  {std::nullopt, v_filter_tensor}, {v});
  // 3 Reshape
  EmplaceOpWithIO(new_ops, ops[start_index + kReshape2Index], {q},
                  {const_cast<TensorWrapper&>(
                      ops[start_index + kSliceQIndex].GetOutputTensor(0))});
  EmplaceOpWithIO(new_ops, ops[start_index + kReshape2Index], {k},
                  {const_cast<TensorWrapper&>(
                      ops[start_index + kSliceKIndex].GetOutputTensor(0))});
  EmplaceOpWithIO(new_ops, ops[start_index + kReshape2Index], {v},
                  {const_cast<TensorWrapper&>(
                      ops[start_index + kSliceVIndex].GetOutputTensor(0))});
  // Validate new graph.
  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
                    return validate_op_config(op_wrapper);
                  });
  if (is_valid) {
    // Replace the matched pattern with a newly generated subgraph.
    size_t step_size = new_ops.size();
    pattern_size = 6;
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
