// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/mha_to_sha.h"

#include <array>
#include <functional>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/split_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "QnnInterface.h"  // from @qairt
#include "QnnOpDef.h"  // from @qairt

namespace qnn {
namespace {

// Returns a boolean indicating whether the output tensor of op1 at out_id is
// connected to the input tensor of op2 at in_id.
#define IS_CONNECTED(op1, out_id, op2, in_id)     \
  (ops[start_id + op1].GetOutputTensor(out_id) == \
   ops[start_id + op2].GetInputTensor(in_id))

constexpr size_t kMulIndex = 0;
constexpr size_t kTransposePrefillIndex = 1;
constexpr size_t kReshapePrefillIndex = 2;
constexpr size_t kMatMulK1Index = 1;
constexpr size_t kMatMulK2Index = 2;
constexpr size_t kConcatIndex = 3;
constexpr size_t kReshape0Index = 4;
constexpr size_t kAddIndex = 5;
constexpr size_t kReshape1Index = 6;
constexpr size_t kSoftmaxIndex = 7;
constexpr size_t kSlice1Index = 8;
constexpr size_t kSlice2Index = 9;
constexpr size_t kMatMulV1Index = 10;
constexpr size_t kMatMulV2Index = 11;
constexpr size_t kAdd2Index = 12;
constexpr size_t kReshape2Index = 13;
constexpr size_t kTranspose2Index = 14;
constexpr size_t kReshape3Index = 15;

void CloneOpWithIO(
    std::vector<OpWrapper>& new_ops, const OpWrapper& source_op,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& inputs,
    const std::vector<std::optional<qnn::TensorWrapperRef>>& outputs) {
  OpWrapper ret = source_op;
  ret.UpdateTensors(inputs, outputs);
  new_ops.emplace_back(ret);
}

TensorWrapper& BuildSingleSHA(std::vector<OpWrapper>& ops, size_t start_id,
                              std::vector<OpWrapper>& new_ops,
                              TensorPool& tensor_pool,
                              qnn::TensorWrapper& sha_input,
                              const ::qnn::OpWrapper& first_mul,
                              size_t num_heads) {
  // Mul
  auto& mul_output = tensor_pool.CloneNativeTensorFrom(
      first_mul.GetOutputTensor(0), sha_input.GetDims());
  CloneOpWithIO(new_ops, first_mul, {sha_input, std::nullopt}, {mul_output});
  // MatMul 1
  const auto& matmulk_cache_output =
      ops[start_id + kMatMulK1Index].GetOutputTensor(0);
  std::vector<uint32_t> matmul1_output_dim = matmulk_cache_output.GetDims();
  matmul1_output_dim[2] = matmul1_output_dim[2] / num_heads;
  auto& matmul1_output = tensor_pool.CloneNativeTensorFrom(matmulk_cache_output,
                                                           matmul1_output_dim);
  CloneOpWithIO(new_ops, ops[start_id + kMatMulK1Index],
                {mul_output, std::nullopt}, {matmul1_output});
  // MatMul 2
  const auto& matmulk_slice_output =
      ops[start_id + kMatMulK2Index].GetOutputTensor(0);
  std::vector<uint32_t> matmul2_output_dim = matmulk_slice_output.GetDims();
  matmul2_output_dim[2] = matmul2_output_dim[2] / num_heads;
  auto& matmul2_output = tensor_pool.CloneNativeTensorFrom(matmulk_slice_output,
                                                           matmul2_output_dim);
  CloneOpWithIO(new_ops, ops[start_id + kMatMulK2Index],
                {mul_output, std::nullopt}, {matmul2_output});
  // Concat
  std::vector<uint32_t> concat_output_dim = matmul1_output.GetDims();
  concat_output_dim[3] += matmul2_output.GetDim(3);
  auto& concat_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_id + kConcatIndex].GetOutputTensor(0), concat_output_dim);
  CloneOpWithIO(new_ops, ops[start_id + kConcatIndex],
                {matmul1_output, matmul2_output}, {concat_output});
  // Add
  auto& add_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_id + kAddIndex].GetOutputTensor(0), concat_output.GetDims());
  CloneOpWithIO(new_ops, ops[start_id + kAddIndex],
                {concat_output, std::nullopt}, {add_output});
  // Softmax
  auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_id + kSoftmaxIndex].GetOutputTensor(0), add_output.GetDims());
  CloneOpWithIO(new_ops, ops[start_id + kSoftmaxIndex], {add_output},
                {softmax_output});
  // Slice 1
  // Create StridedSlice param ranges.
  auto mha_slice1_param =
      ops[start_id + kSlice1Index].GetTensorPararm(0).GetTensor();
  auto mha_slice1_param_data = mha_slice1_param.GetStaticTensorData<int32_t>();
  std::vector<int32_t> slice1_param(mha_slice1_param_data.value().begin(),
                                    mha_slice1_param_data.value().end());
  slice1_param[7] /= num_heads;
  auto& slice1_param_tensor = tensor_pool.CreateStaticTensor(
      mha_slice1_param.GetDataType(), mha_slice1_param.GetQuantParams(),
      mha_slice1_param.GetDims(), mha_slice1_param.GetTensorBytes(),
      slice1_param.data());
  // Create StridedSlice op.
  auto slice1_output_dims =
      ops[start_id + kSlice1Index].GetOutputTensor(0).GetDims();
  slice1_output_dims[2] /= num_heads;
  const std::vector<uint32_t> slice1_output_dim{slice1_output_dims};
  auto& slice1_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_id + kSlice1Index].GetOutputTensor(0), slice1_output_dim);
  CloneOpWithIO(new_ops, ops[start_id + kSlice1Index], {softmax_output},
                {slice1_output});
  new_ops.back().ClearParams();
  new_ops.back().AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                slice1_param_tensor);
  // MatMul 1
  const auto& matmulv_cache_output =
      ops[start_id + kMatMulV1Index].GetOutputTensor(0);

  std::vector<uint32_t> matmul1_v_output_dim = matmulv_cache_output.GetDims();
  matmul1_v_output_dim[2] = matmul1_v_output_dim[2] / num_heads;
  auto& matmul1_v_output = tensor_pool.CloneNativeTensorFrom(
      matmulv_cache_output, matmul1_v_output_dim);
  CloneOpWithIO(new_ops, ops[start_id + kMatMulV1Index],
                {slice1_output, std::nullopt}, {matmul1_v_output});
  // Slice 2
  // Create StridedSlice param ranges.
  auto mha_slice2_param =
      ops[start_id + kSlice2Index].GetTensorPararm(0).GetTensor();
  auto mha_slice2_param_data = mha_slice2_param.GetStaticTensorData<int32_t>();
  std::vector<int32_t> slice2_param(mha_slice2_param_data.value().begin(),
                                    mha_slice2_param_data.value().end());
  slice2_param[7] /= num_heads;
  auto& slice2_param_tensor = tensor_pool.CreateStaticTensor(
      mha_slice2_param.GetDataType(), mha_slice2_param.GetQuantParams(),
      mha_slice2_param.GetDims(), mha_slice2_param.GetTensorBytes(),
      slice2_param.data());
  // Create StridedSlice op.
  auto slice2_output_dims =
      ops[start_id + kSlice2Index].GetOutputTensor(0).GetDims();
  slice2_output_dims[2] /= num_heads;
  auto& slice2_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_id + kSlice2Index].GetOutputTensor(0), slice2_output_dims);
  CloneOpWithIO(new_ops, ops[start_id + kSlice2Index], {softmax_output},
                {slice2_output});
  new_ops.back().ClearParams();
  new_ops.back().AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                slice2_param_tensor);
  // MatMul 2
  const auto& matmulv_slice_output =
      ops[start_id + kMatMulV2Index].GetOutputTensor(0);
  std::vector<uint32_t> matmul2_v_output_dim = matmulv_slice_output.GetDims();
  matmul2_v_output_dim[2] = matmul2_v_output_dim[2] / num_heads;
  auto& matmul2_v_output = tensor_pool.CloneNativeTensorFrom(
      matmulv_slice_output, matmul2_v_output_dim);
  CloneOpWithIO(new_ops, ops[start_id + kMatMulV2Index],
                {slice2_output, std::nullopt}, {matmul2_v_output});
  // Add
  auto& add_final_output = tensor_pool.CloneNativeTensorFrom(
      ops[start_id + kAdd2Index].GetOutputTensor(0),
      matmul1_v_output.GetDims());
  CloneOpWithIO(new_ops, ops[start_id + kAdd2Index],
                {matmul1_v_output, matmul2_v_output}, {add_final_output});
  return add_final_output;
}

std::vector<OpWrapper> TransformToSHA(std::vector<OpWrapper>& ops,
                                      size_t start_id, TensorPool& tensor_pool,
                                      const qnn::TensorWrapper& mha_input,
                                      const qnn::TensorWrapper& mha_output,
                                      const ::qnn::OpWrapper& first_mul,
                                      size_t num_heads) {
  std::vector<OpWrapper> new_ops;

  // Split
  const std::vector<uint32_t> split_axis_dim{1};
  const std::array<int32_t, 1> split_axis_data{2};
  auto& split_axis = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, split_axis_dim,
      split_axis_data.size() * sizeof(split_axis_data[0]),
      split_axis_data.data());
  std::vector<::qnn::TensorWrapperRef> head_inputs;
  head_inputs.reserve(num_heads);
  for (int i = 0; i < num_heads; ++i) {
    auto head_input_dims = ops[start_id].GetOutputTensor(0).GetDims();
    head_input_dims[2] /= num_heads;
    auto& split_output =
        tensor_pool.CloneNativeTensorFrom(mha_input, head_input_dims);
    head_inputs.emplace_back(split_output);
  }
  auto split = BuildSplitOp(
      tensor_pool, {split_axis, const_cast<::qnn::TensorWrapper&>(mha_input)},
      head_inputs, num_heads);
  std::move(split.begin(), split.end(), std::back_inserter(new_ops));

  std::vector<::qnn::TensorWrapperRef> concat_after_mha;
  concat_after_mha.reserve(num_heads);
  for (int i = 0; i < num_heads; ++i) {
    concat_after_mha.emplace_back(
        BuildSingleSHA(ops, start_id, new_ops, tensor_pool,
                       const_cast<TensorWrapper&>(head_inputs[i].get()),
                       first_mul, num_heads));
  }
  // Concat
  auto concat_dims = mha_output.GetDims();
  concat_dims.insert(concat_dims.begin(), 1);
  auto& concat_output =
      tensor_pool.CloneNativeTensorFrom(mha_output, concat_dims);
  auto concat_final =
      BuildConcatenationOp(tensor_pool, concat_after_mha, {concat_output}, 3);
  std::move(concat_final.begin(), concat_final.end(),
            std::back_inserter(new_ops));
  // Reshape
  auto reshape =
      BuildReshapeOp(tensor_pool, {concat_output},
                     {const_cast<::qnn::TensorWrapper&>(mha_output)});
  std::move(reshape.begin(), reshape.end(), std::back_inserter(new_ops));
  return new_ops;
}

}  // namespace

size_t OptimizeMHAPrefill(std::function<bool(OpWrapper&)> validate_op_config,
                          std::vector<OpWrapper>& ops, size_t start_id,
                          TensorPool& tensor_pool, size_t pattern_size) {
  // Connection check
  if (!(IS_CONNECTED(kMulIndex, 0, kTransposePrefillIndex, 0) &&
        IS_CONNECTED(kTransposePrefillIndex, 0, kReshapePrefillIndex, 0) &&
        IS_CONNECTED(kReshapePrefillIndex, 0, kMatMulK1Index + 2, 0) &&
        IS_CONNECTED(kReshapePrefillIndex, 0, kMatMulK2Index + 2, 0) &&
        IS_CONNECTED(kMatMulK1Index + 2, 0, kConcatIndex + 2, 0) &&
        IS_CONNECTED(kMatMulK2Index + 2, 0, kConcatIndex + 2, 1) &&
        IS_CONNECTED(kConcatIndex + 2, 0, kReshape0Index + 2, 0) &&
        IS_CONNECTED(kReshape0Index + 2, 0, kAddIndex + 2, 0) &&
        IS_CONNECTED(kAddIndex + 2, 0, kReshape1Index + 2, 0) &&
        IS_CONNECTED(kReshape1Index + 2, 0, kSoftmaxIndex + 2, 0) &&
        IS_CONNECTED(kSoftmaxIndex + 2, 0, kSlice1Index + 2, 0) &&
        IS_CONNECTED(kSoftmaxIndex + 2, 0, kSlice2Index + 2, 0) &&
        IS_CONNECTED(kSlice1Index + 2, 0, kMatMulV1Index + 2, 0) &&
        IS_CONNECTED(kSlice2Index + 2, 0, kMatMulV2Index + 2, 0) &&
        IS_CONNECTED(kMatMulV1Index + 2, 0, kAdd2Index + 2, 0) &&
        IS_CONNECTED(kMatMulV2Index + 2, 0, kAdd2Index + 2, 1) &&
        IS_CONNECTED(kAdd2Index + 2, 0, kReshape2Index + 2, 0) &&
        IS_CONNECTED(kReshape2Index + 2, 0, kTranspose2Index + 2, 0) &&
        IS_CONNECTED(kTranspose2Index + 2, 0, kReshape3Index + 2, 0))) {
    return 1;
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] MHA optimization (Prefill)");
  std::vector<OpWrapper> new_ops;
  const auto& first_mul = ops[start_id + kMulIndex];
  const auto& pattern_input = first_mul.GetInputTensor(0);
  const auto& pattern_output =
      ops[start_id + pattern_size - 1].GetOutputTensor(0);

  // Transpose
  auto transpose_output_dims =
      ops[start_id + kTransposePrefillIndex].GetOutputTensor(0).GetDims();
  auto& transpose_output =
      tensor_pool.CloneNativeTensorFrom(pattern_input, transpose_output_dims);
  CloneOpWithIO(new_ops, ops[start_id + kTransposePrefillIndex],
                {const_cast<::qnn::TensorWrapper&>(pattern_input)},
                {transpose_output});

  // Reshape
  auto& reshape_output = tensor_pool.CloneNativeTensorFrom(
      pattern_input, {transpose_output_dims[0], 1,
                      transpose_output_dims[1] * transpose_output_dims[2],
                      transpose_output_dims[3]});
  CloneOpWithIO(new_ops, ops[start_id + kReshapePrefillIndex],
                {transpose_output}, {reshape_output});

  // Process MHA to SHA transformation.
  const int num_heads = pattern_input.GetDim(2);
  const auto& mha_input = new_ops.back().GetOutputTensor(0);
  auto sha_ops =
      TransformToSHA(ops, start_id + new_ops.size(), tensor_pool, mha_input,
                     pattern_output, first_mul, num_heads);
  std::move(sha_ops.begin(), sha_ops.end(), std::back_inserter(new_ops));

  // Validate new graph.
  // TODO(jiunkaiy): Disable bypassing Split int16 op validator.
  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
                    return op_wrapper.IsOpCode(QnnOpCode::kSplit) ||
                           validate_op_config(op_wrapper);
                  });
  if (is_valid) {
    // Replace the matched pattern with a newly generated subgraph.
    size_t step_size = new_ops.size();
    ops.insert(ops.begin() + start_id + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    ops.erase(ops.begin() + start_id, ops.begin() + start_id + pattern_size);
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}

size_t OptimizeMHADecode(std::function<bool(OpWrapper&)> validate_op_config,
                         std::vector<OpWrapper>& ops, size_t start_id,
                         TensorPool& tensor_pool, size_t pattern_size) {
  // Connection check
  if (!(IS_CONNECTED(kMulIndex, 0, kMatMulK1Index, 0) &&
        IS_CONNECTED(kMulIndex, 0, kMatMulK2Index, 0) &&
        IS_CONNECTED(kMatMulK1Index, 0, kConcatIndex, 0) &&
        IS_CONNECTED(kMatMulK2Index, 0, kConcatIndex, 1) &&
        IS_CONNECTED(kConcatIndex, 0, kReshape0Index, 0) &&
        IS_CONNECTED(kReshape0Index, 0, kAddIndex, 0) &&
        IS_CONNECTED(kAddIndex, 0, kReshape1Index, 0) &&
        IS_CONNECTED(kReshape1Index, 0, kSoftmaxIndex, 0) &&
        IS_CONNECTED(kSoftmaxIndex, 0, kSlice1Index, 0) &&
        IS_CONNECTED(kSoftmaxIndex, 0, kSlice2Index, 0) &&
        IS_CONNECTED(kSlice1Index, 0, kMatMulV1Index, 0) &&
        IS_CONNECTED(kSlice2Index, 0, kMatMulV2Index, 0) &&
        IS_CONNECTED(kMatMulV1Index, 0, kAdd2Index, 0) &&
        IS_CONNECTED(kMatMulV2Index, 0, kAdd2Index, 1) &&
        IS_CONNECTED(kAdd2Index, 0, kReshape2Index, 0))) {
    return 1;
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] MHA optimization (Decode)");
  std::vector<OpWrapper> new_ops;
  const auto& first_mul = ops[start_id + kMulIndex];
  const auto& pattern_input = first_mul.GetInputTensor(0);
  const auto& pattern_output =
      ops[start_id + pattern_size - 1].GetOutputTensor(0);

  // Process MHA to SHA transformation.
  const int num_heads = pattern_input.GetDim(2);
  auto sha_ops =
      TransformToSHA(ops, start_id + new_ops.size(), tensor_pool, pattern_input,
                     pattern_output, first_mul, num_heads);
  std::move(sha_ops.begin(), sha_ops.end(), std::back_inserter(new_ops));

  // Validate new graph.
  // TODO(jiunkaiy): Disable bypassing Split int16 op validator.
  const bool is_valid =
      std::all_of(new_ops.begin(), new_ops.end(),
                  [validate_op_config](::qnn::OpWrapper& op_wrapper) -> bool {
                    return op_wrapper.IsOpCode(QnnOpCode::kSplit) ||
                           validate_op_config(op_wrapper);
                  });
  if (is_valid) {
    // Replace the matched pattern with a newly generated subgraph.
    size_t step_size = new_ops.size();
    ops.insert(ops.begin() + start_id + pattern_size,
               std::make_move_iterator(new_ops.begin()),
               std::make_move_iterator(new_ops.end()));
    ops.erase(ops.begin() + start_id, ops.begin() + start_id + pattern_size);
    return step_size;
  }
  QNN_LOG_WARNING(
      "[G2G] Validation failed. Rolling back to the original graph.");
  return 1;
}
}  // namespace qnn
