// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/mha_to_sha.h"

#include <array>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/softmax_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_op_builder.h"
#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {
namespace {
constexpr size_t kMulIndex = 0;
constexpr size_t kTransposeIndex = 1;
constexpr size_t kMatMulK1Index = 1;
constexpr size_t kMatMulK2Index = 2;
constexpr size_t kConcatIndex = 3;
constexpr size_t kAddIndex = 5;
constexpr size_t kSoftmaxIndex = 7;
}  // namespace
std::vector<OpWrapper> PreprocessPrefill(
    TensorPool& tensor_pool, const qnn::TensorWrapper& input_tensor,
    const qnn::TensorWrapper& transpose_perm,
    const std::vector<uint32_t>& transpose_output_dims) {
  QNN_LOG_INFO("Prefill preprocess");
  std::vector<OpWrapper> res;

  // Transpose
  std::vector<::qnn::TensorWrapperRef> transpose_inputs;
  transpose_inputs.emplace_back(
      const_cast<::qnn::TensorWrapper&>(input_tensor));
  transpose_inputs.emplace_back(
      const_cast<::qnn::TensorWrapper&>(transpose_perm));
  std::vector<::qnn::TensorWrapperRef> transpose_outputs;
  auto& transpose_output =
      tensor_pool.CloneNativeTensorFrom(input_tensor, transpose_output_dims);
  transpose_outputs.emplace_back(transpose_output);
  auto transpose =
      BuildTransposeOp(tensor_pool, transpose_inputs, transpose_outputs);
  std::move(transpose.begin(), transpose.end(), std::back_inserter(res));

  // Reshape
  std::vector<::qnn::TensorWrapperRef> reshape_inputs;
  reshape_inputs.emplace_back(transpose_output);
  std::vector<::qnn::TensorWrapperRef> reshape_outputs;
  std::vector<uint32_t> reshape_output_dim = {
      transpose_output_dims[0], 1,
      transpose_output_dims[1] * transpose_output_dims[2],
      transpose_output_dims[3]};
  reshape_outputs.emplace_back(
      tensor_pool.CloneNativeTensorFrom(input_tensor, reshape_output_dim));
  auto reshape = BuildReshapeOp(tensor_pool, reshape_inputs, reshape_outputs);
  std::move(reshape.begin(), reshape.end(), std::back_inserter(res));

  return res;
}

bool TransformMHAToSHA(std::vector<OpWrapper>& ops, size_t start_id,
                       TensorPool& tensor_pool) {
  QNN_LOG_INFO("Start TransformMHAToSHA");
  std::vector<OpWrapper> new_ops;

  const qnn::TensorWrapper* pattern_input_ptr =
      &(ops[start_id].GetInputTensor(0));  // Mul's input
  const auto& mul_const = ops[start_id + kMulIndex].GetInputTensor(1);
  const auto& mul_output_quant_param =
      ops[start_id + kMulIndex].GetOutputTensor(0).GetQuantParams();

  const int num_heads = (*pattern_input_ptr).GetDim(2);
  int seq_len = (*pattern_input_ptr).GetDim(1);
  size_t id_offset = 0;
  if ((*pattern_input_ptr).GetDim(1) != 1) {
    auto preprocess_ops = PreprocessPrefill(
        tensor_pool, *pattern_input_ptr,
        ops[start_id + kTransposeIndex].GetPararmTensor(0),
        ops[start_id + kTransposeIndex].GetOutputTensor(0).GetDims());
    std::move(preprocess_ops.begin(), preprocess_ops.end(),
              std::back_inserter(new_ops));
    id_offset = new_ops.size();
    pattern_input_ptr = &(new_ops.back().GetOutputTensor(0));
  }

  const auto& matmulk_cache =
      ops[start_id + kMatMulK1Index + id_offset].GetInputTensor(1);
  const auto& matmulk_cache_output =
      ops[start_id + kMatMulK1Index + id_offset].GetOutputTensor(0);
  const auto& matmulk_slice =
      ops[start_id + kMatMulK2Index + id_offset].GetInputTensor(1);
  const auto& matmulk_slice_output =
      ops[start_id + kMatMulK2Index + id_offset].GetOutputTensor(0);

  const auto& concat_mha_output =
      ops[start_id + kConcatIndex + id_offset].GetOutputTensor(0);

  const auto& add_mha_mask =
      ops[start_id + kAddIndex + id_offset].GetInputTensor(1);
  const auto& add_mha_output =
      ops[start_id + kAddIndex + id_offset].GetOutputTensor(0);

  const auto& softmax_mha_output =
      ops[start_id + kSoftmaxIndex + id_offset].GetOutputTensor(0);

  const auto& matmulv_cache = ops[start_id + 10 + id_offset].GetInputTensor(1);
  const auto& matmulv_cache_output =
      ops[start_id + 10 + id_offset].GetOutputTensor(0);
  const auto& matmulv_slice = ops[start_id + 11 + id_offset].GetInputTensor(1);
  const auto& matmulv_slice_output =
      ops[start_id + 11 + id_offset].GetOutputTensor(0);

  const auto& add_after_matmulv_output =
      ops[start_id + 10 + id_offset].GetOutputTensor(0);
  const auto& add_after_matmulv_output_2 =
      ops[start_id + 11 + id_offset].GetOutputTensor(0);

  // Slice tensor for multiplied by K
  const std::vector<uint32_t> slice_size_dim{4};
  const std::array<int32_t, 4> slice_size_data{1, 1, seq_len, 256};
  auto& slice_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, slice_size_dim,
      slice_size_data.size() * sizeof(slice_size_data[0]),
      slice_size_data.data());

  // Slice tensor for multiplied by V
  const std::vector<uint32_t> slice1_begin_dim{4};
  const std::array<int32_t, 4> slice1_begin_data{0, 0, 0, 0};
  auto& slice1_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, slice1_begin_dim,
      slice1_begin_data.size() * sizeof(slice1_begin_data[0]),
      slice1_begin_data.data());
  const std::vector<uint32_t> slice1_size_dim{4};
  const std::array<int32_t, 4> slice1_size_data{1, 1, seq_len, 1280};
  auto& slice1_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, slice1_size_dim,
      slice1_size_data.size() * sizeof(slice1_size_data[0]),
      slice1_size_data.data());

  const std::vector<uint32_t> slice2_begin_dim{4};
  const std::array<int32_t, 4> slice2_begin_data{0, 0, 0, 1280};
  auto& slice2_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, slice2_begin_dim,
      slice2_begin_data.size() * sizeof(slice2_begin_data[0]),
      slice2_begin_data.data());
  const std::vector<uint32_t> slice2_size_dim{4};
  const std::array<int32_t, 4> slice2_size_data{1, 1, seq_len, seq_len};
  auto& slice2_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, slice2_size_dim,
      slice2_size_data.size() * sizeof(slice2_size_data[0]),
      slice2_size_data.data());

  std::array<::qnn::TensorWrapper*, 4> concat_aftet_mha;
  std::array<::qnn::TensorWrapper*, 4> concat_aftet_mha_2;
  for (int i = 0; i < num_heads; ++i) {
    std::vector<::qnn::TensorWrapperRef> slice_inputs;
    slice_inputs.emplace_back(
        const_cast<::qnn::TensorWrapper&>(*pattern_input_ptr));
    const std::vector<uint32_t> slice_begin_dim{4};
    const std::array<int32_t, 4> slice_begin_data{
        0, 0, static_cast<int>(i * pattern_input_ptr->GetDim(2) / num_heads),
        0};
    auto& slice_begin = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_INT_32, {}, slice_begin_dim,
        slice_begin_data.size() * sizeof(slice_begin_data[0]),
        slice_begin_data.data());
    slice_inputs.emplace_back(slice_begin);
    slice_inputs.emplace_back(slice_size);
    std::vector<::qnn::TensorWrapperRef> slice_outputs;
    const std::vector<uint32_t> slice_output_dim = {
        1, 1, static_cast<uint32_t>(seq_len), 256};
    auto& slice_output =
        tensor_pool.CloneNativeTensorFrom(*pattern_input_ptr, slice_output_dim);
    slice_outputs.emplace_back(slice_output);
    auto slice = BuildSliceOp(tensor_pool, slice_inputs, slice_outputs);
    std::move(slice.begin(), slice.end(), std::back_inserter(new_ops));

    // Mul
    std::vector<::qnn::TensorWrapperRef> mul_inputs;
    mul_inputs.emplace_back(slice_output);
    mul_inputs.emplace_back(const_cast<::qnn::TensorWrapper&>(mul_const));
    std::vector<::qnn::TensorWrapperRef> mul_outputs;
    auto& mul_output =
        tensor_pool.CloneNativeTensorFrom(slice_output, mul_output_quant_param);
    mul_outputs.emplace_back(mul_output);
    auto mul = BuildElementwiseMulOp(tensor_pool, mul_inputs, mul_outputs);
    std::move(mul.begin(), mul.end(), std::back_inserter(new_ops));
    // 2 * MatMul
    // MatMul 1
    std::vector<::qnn::TensorWrapperRef> matmul1_inputs;
    matmul1_inputs.emplace_back(mul_output);
    matmul1_inputs.emplace_back(
        const_cast<::qnn::TensorWrapper&>(matmulk_cache));
    std::vector<::qnn::TensorWrapperRef> matmul1_outputs;
    std::vector<uint32_t> matmul1_output_dim = matmulk_cache_output.GetDims();
    matmul1_output_dim[2] = matmul1_output_dim[2] / num_heads;
    auto& matmul1_output = tensor_pool.CloneNativeTensorFrom(
        matmulk_cache_output, matmul1_output_dim);
    matmul1_outputs.emplace_back(matmul1_output);
    auto matmul1 = BuildMatmulOp(tensor_pool, matmul1_inputs, matmul1_outputs,
                                 false, true);
    std::move(matmul1.begin(), matmul1.end(), std::back_inserter(new_ops));
    // MatMul 2
    std::vector<::qnn::TensorWrapperRef> matmul2_inputs;
    matmul2_inputs.emplace_back(mul_output);
    matmul2_inputs.emplace_back(
        const_cast<::qnn::TensorWrapper&>(matmulk_slice));
    std::vector<::qnn::TensorWrapperRef> matmul2_outputs;
    std::vector<uint32_t> matmul2_output_dim = matmulk_slice_output.GetDims();
    matmul2_output_dim[2] = matmul2_output_dim[2] / num_heads;
    auto& matmul2_output = tensor_pool.CloneNativeTensorFrom(
        matmulk_slice_output, matmul2_output_dim);
    matmul2_outputs.emplace_back(matmul2_output);
    auto matmul2 = BuildMatmulOp(tensor_pool, matmul2_inputs, matmul2_outputs,
                                 false, true);
    std::move(matmul2.begin(), matmul2.end(), std::back_inserter(new_ops));
    // Concat
    std::vector<::qnn::TensorWrapperRef> concat_inputs;
    concat_inputs.emplace_back(matmul1_output);
    concat_inputs.emplace_back(matmul2_output);
    std::vector<::qnn::TensorWrapperRef> concat_outputs;
    std::vector<uint32_t> concat_output_dim = matmul1_output.GetDims();
    concat_output_dim[3] += matmul2_output.GetDim(3);
    auto& concat_output =
        tensor_pool.CloneNativeTensorFrom(concat_mha_output, concat_output_dim);
    concat_outputs.emplace_back(concat_output);
    auto concat =
        BuildConcatenationOp(tensor_pool, concat_inputs, concat_outputs, 3);
    std::move(concat.begin(), concat.end(), std::back_inserter(new_ops));
    // Add
    std::vector<::qnn::TensorWrapperRef> add_inputs;
    add_inputs.emplace_back(concat_output);
    add_inputs.emplace_back(const_cast<::qnn::TensorWrapper&>(add_mha_mask));
    std::vector<::qnn::TensorWrapperRef> add_outputs;
    auto& add_output = tensor_pool.CloneNativeTensorFrom(
        add_mha_output, concat_output.GetDims());
    add_outputs.emplace_back(add_output);
    auto add = BuildElementwiseAddOp(tensor_pool, add_inputs, add_outputs);
    std::move(add.begin(), add.end(), std::back_inserter(new_ops));
    // Softmax
    std::vector<::qnn::TensorWrapperRef> softmax_inputs;
    softmax_inputs.emplace_back(add_output);
    std::vector<::qnn::TensorWrapperRef> softmax_outputs;
    auto& softmax_output = tensor_pool.CloneNativeTensorFrom(
        softmax_mha_output, add_output.GetDims());
    softmax_outputs.emplace_back(softmax_output);
    auto softmax =
        BuildSoftmaxOp(tensor_pool, softmax_inputs, softmax_outputs, 1.0f);
    std::move(softmax.begin(), softmax.end(), std::back_inserter(new_ops));
    // 2 (Slice -> MatMul)
    // Slice 1
    std::vector<::qnn::TensorWrapperRef> slice1_inputs;
    slice1_inputs.emplace_back(softmax_output);
    slice1_inputs.emplace_back(slice1_begin);
    slice1_inputs.emplace_back(slice1_size);
    std::vector<::qnn::TensorWrapperRef> slice1_outputs;
    const std::vector<uint32_t> slice1_output_dim{
        1, 1, static_cast<uint32_t>(seq_len), 1280};
    auto& slice1_output =
        tensor_pool.CloneNativeTensorFrom(softmax_output, slice1_output_dim);
    slice1_outputs.emplace_back(slice1_output);
    auto slice1 = BuildSliceOp(tensor_pool, slice1_inputs, slice1_outputs);
    std::move(slice1.begin(), slice1.end(), std::back_inserter(new_ops));
    // MatMul 1
    std::vector<::qnn::TensorWrapperRef> matmul1_v_inputs;
    matmul1_v_inputs.emplace_back(slice1_output);
    matmul1_v_inputs.emplace_back(
        const_cast<::qnn::TensorWrapper&>(matmulv_cache));
    std::vector<::qnn::TensorWrapperRef> matmul1_v_outputs;
    std::vector<uint32_t> matmul1_v_output_dim = matmulv_cache_output.GetDims();
    matmul1_v_output_dim[2] = matmul1_v_output_dim[2] / num_heads;
    auto& matmul1_v_output = tensor_pool.CloneNativeTensorFrom(
        matmulv_cache_output, matmul1_v_output_dim);
    concat_aftet_mha[i] = &matmul1_v_output;
    matmul1_v_outputs.emplace_back(matmul1_v_output);
    auto matmul1_v = BuildMatmulOp(tensor_pool, matmul1_v_inputs,
                                   matmul1_v_outputs, false, true);
    std::move(matmul1_v.begin(), matmul1_v.end(), std::back_inserter(new_ops));
    // Slice 2
    std::vector<::qnn::TensorWrapperRef> slice2_inputs;
    slice2_inputs.emplace_back(softmax_output);
    slice2_inputs.emplace_back(slice2_begin);
    slice2_inputs.emplace_back(slice2_size);
    std::vector<::qnn::TensorWrapperRef> slice2_outputs;
    std::vector<uint32_t> slice2_output_dim{
        1, 1, static_cast<uint32_t>(seq_len), static_cast<uint32_t>(seq_len)};
    auto& slice2_output =
        tensor_pool.CloneNativeTensorFrom(softmax_output, slice2_output_dim);
    slice2_outputs.emplace_back(slice2_output);
    auto slice2 = BuildSliceOp(tensor_pool, slice2_inputs, slice2_outputs);
    std::move(slice2.begin(), slice2.end(), std::back_inserter(new_ops));
    // MatMul 2
    std::vector<::qnn::TensorWrapperRef> matmul2_v_inputs;
    matmul2_v_inputs.emplace_back(slice2_output);
    matmul2_v_inputs.emplace_back(
        const_cast<::qnn::TensorWrapper&>(matmulv_slice));
    std::vector<::qnn::TensorWrapperRef> matmul2_v_outputs;
    std::vector<uint32_t> matmul2_v_output_dim = matmulv_slice_output.GetDims();
    matmul2_v_output_dim[2] = matmul2_v_output_dim[2] / num_heads;
    auto& matmul2_v_output = tensor_pool.CloneNativeTensorFrom(
        matmulv_slice_output, matmul2_v_output_dim);
    concat_aftet_mha_2[i] = &matmul2_v_output;
    matmul2_v_outputs.emplace_back(matmul2_v_output);
    auto matmul2_v = BuildMatmulOp(tensor_pool, matmul2_v_inputs,
                                   matmul2_v_outputs, false, true);
    std::move(matmul2_v.begin(), matmul2_v.end(), std::back_inserter(new_ops));
    // // Add
    // std::vector<::qnn::TensorWrapperRef> add_final_inputs;
    // add_final_inputs.emplace_back(matmul1_v_output);
    // add_final_inputs.emplace_back(matmul2_v_output);
    // std::vector<::qnn::TensorWrapperRef> add_final_outputs;
    // std::vector<uint32_t> add_final_output_dim = matmul1_v_output.GetDims();
    // auto& add_final_output = tensor_pool.CloneNativeTensorFrom(
    //     add_after_matmulv_output, add_final_output_dim);
    // concat_aftet_mha[i] = &add_final_output;
    // add_final_outputs.emplace_back(add_final_output);
    // auto add_final =
    //     BuildElementwiseAddOp(tensor_pool, add_final_inputs,
    //     add_final_outputs);
    // std::move(add_final.begin(), add_final.end(),
    // std::back_inserter(new_ops));
  }
  // Concat
  std::vector<::qnn::TensorWrapperRef> concat_final_inputs;
  for (int i = 0; i < 4; i++) {
    concat_final_inputs.emplace_back(*(concat_aftet_mha[i]));
  }
  std::vector<::qnn::TensorWrapperRef> concat_final_outputs;
  concat_final_outputs.emplace_back(
      const_cast<::qnn::TensorWrapper&>(add_after_matmulv_output));
  auto concat_final = BuildConcatenationOp(tensor_pool, concat_final_inputs,
                                           concat_final_outputs, 2);
  std::move(concat_final.begin(), concat_final.end(),
            std::back_inserter(new_ops));
  // Concat 2
  std::vector<::qnn::TensorWrapperRef> concat_final_inputs_2;
  for (int i = 0; i < 4; i++) {
    concat_final_inputs_2.emplace_back(*(concat_aftet_mha_2[i]));
  }
  std::vector<::qnn::TensorWrapperRef> concat_final_outputs_2;
  concat_final_outputs_2.emplace_back(
      const_cast<::qnn::TensorWrapper&>(add_after_matmulv_output_2));
  auto concat_final_2 = BuildConcatenationOp(tensor_pool, concat_final_inputs_2,
                                             concat_final_outputs_2, 2);
  std::move(concat_final_2.begin(), concat_final_2.end(),
            std::back_inserter(new_ops));
  QNN_LOG_INFO("Add new ops");
  ops.insert(ops.begin() + start_id + 12 + id_offset,
             std::make_move_iterator(new_ops.begin()),
             std::make_move_iterator(new_ops.end()));
  QNN_LOG_INFO("Remove useless ops");
  // And then remove 0~14
  ops.erase(ops.begin() + start_id, ops.begin() + start_id + 12 + id_offset);
  return true;
}
}  // namespace qnn
