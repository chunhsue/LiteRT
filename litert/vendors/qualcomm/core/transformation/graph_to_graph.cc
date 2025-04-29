// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/matmul_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/op_code.h"
#include "litert/vendors/qualcomm/core/builders/reshape_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/slice_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/softmax_op_builder.h"
#include "litert/vendors/qualcomm/core/builders/transpose_op_builder.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
namespace qnn {

namespace {

constexpr size_t kQnnOpCodeSize =
    static_cast<size_t>(QnnOpCode::kQnnOpCodeUnknown);

constexpr std::array<size_t, kQnnOpCodeSize> create_bad_match_table(
    const QnnOpCode* op_codes, size_t pattern_length) {
  std::array<size_t, kQnnOpCodeSize> table{};
  for (size_t i = 0; i < table.size(); ++i) {
    table[i] = pattern_length;
  }
  for (size_t i = 0; i < pattern_length - 1; ++i) {
    table[static_cast<size_t>(op_codes[i])] = pattern_length - i - 1;
  }
  return table;
}

bool MatchPattern(size_t& end_id, const std::vector<OpWrapper>& ops,
                  const QnnOpCode* op_codes, size_t pattern_length,
                  const std::array<size_t, kQnnOpCodeSize>& bad_match_table) {
  if (end_id < pattern_length - 1) {
    QNN_LOG_DEBUG("INIT");
    end_id = pattern_length - 1;
    return false;
  }
  if (end_id >= ops.size()) {
    QNN_LOG_DEBUG("END");
    end_id = ops.size();
    return false;
  }
  QNN_LOG_DEBUG("%d/%d", end_id, ops.size());
  for (size_t i = 0; i < pattern_length; ++i) {
    if (!ops[end_id - i].IsOpType(op_codes[pattern_length - i - 1])) {
      QNN_LOG_DEBUG("Miss %d %s", end_id - i,
                    GetQnnOpType(ops[end_id - i].GetOpCode()));
      end_id += bad_match_table[static_cast<size_t>(ops[end_id].GetOpCode())];
      return false;
    }
  }
  QNN_LOG_DEBUG("Get it");
  return true;
}

typedef bool (*G2GTransform)(std::vector<OpWrapper>&, size_t, TensorPool&);
void Transform(std::vector<OpWrapper>& ops, TensorPool& tensor_pool,
               const QnnOpCode* op_codes, size_t pattern_length,
               const std::array<size_t, kQnnOpCodeSize>& bad_match_table,
               G2GTransform custom_transform) {
  for (int i = 0; i < pattern_length; ++i) {
    QNN_LOG_INFO("Bad table value of %s is %d.", GetQnnOpType(op_codes[i]),
                 bad_match_table[static_cast<int>(op_codes[i])])
  }
  size_t end_id = 0;
  while (end_id != ops.size()) {
    QNN_LOG_DEBUG("Current end_id %d", end_id);
    if (MatchPattern(end_id, ops, op_codes, pattern_length, bad_match_table)) {
      if (!custom_transform(ops, end_id - (pattern_length - 1), tensor_pool)) {
        end_id += pattern_length;
      } else {
        end_id += 1;
        QNN_LOG_INFO("G2G Match Success!");
      }
    }
  }
}

// MatMul-Convert Fusion
constexpr std::array<QnnOpCode, 2> kMatMulConvertPattern1 = {
    QnnOpCode::kQnnOpCodeMatMul,
    QnnOpCode::kQnnOpCodeConvert,
};
constexpr auto kBadMatchTableMatMulConvertPattern1 = create_bad_match_table(
    kMatMulConvertPattern1.data(), kMatMulConvertPattern1.size());

bool FuseMatMulConvert1(std::vector<OpWrapper>& ops, size_t start_id,
                        TensorPool& tensor_pool) {
  if (&ops[start_id].GetOutputTensor(0) ==
      &ops[start_id + 1].GetInputTensor(0)) {
    ops[start_id].StealOutputs(ops[start_id + 1]);
    ops.erase(ops.begin() + start_id + 1);
    QNN_LOG_INFO("FuseMatMulConvert1");
    return true;
  } else {
    return false;
  }
}

constexpr std::array<QnnOpCode, 3> kMatMulConvertPattern2 = {
    QnnOpCode::kQnnOpCodeMatMul,
    QnnOpCode::kQnnOpCodeMatMul,
    QnnOpCode::kQnnOpCodeConvert,
};
constexpr auto kBadMatchTableMatMulConvertPattern2 = create_bad_match_table(
    kMatMulConvertPattern2.data(), kMatMulConvertPattern2.size());

bool FuseMatMulConvert2(std::vector<OpWrapper>& ops, size_t start_id,
                        TensorPool& tensor_pool) {
  if (&ops[start_id].GetOutputTensor(0) ==
      &ops[start_id + 2].GetInputTensor(0)) {
    ops[start_id].StealOutputs(ops[start_id + 2]);
    ops.erase(ops.begin() + start_id + 2);
    QNN_LOG_INFO("FuseMatMulConvert2");
    return true;
  } else {
    return false;
  }
}
constexpr std::array<QnnOpCode, 15> kGemma3MHAToSHA = {
    QnnOpCode::kQnnOpCodeElementWiseMultiply,
    QnnOpCode::kQnnOpCodeTranspose,
    QnnOpCode::kQnnOpCodeReshape,
    QnnOpCode::kQnnOpCodeMatMul,
    QnnOpCode::kQnnOpCodeMatMul,
    QnnOpCode::kQnnOpCodeConcat,
    QnnOpCode::kQnnOpCodeReshape,
    QnnOpCode::kQnnOpCodeElementWiseAdd,
    QnnOpCode::kQnnOpCodeReshape,
    QnnOpCode::kQnnOpCodeSoftmax,
    QnnOpCode::kQnnOpCodeStridedSlice,
    QnnOpCode::kQnnOpCodeStridedSlice,
    QnnOpCode::kQnnOpCodeMatMul,
    QnnOpCode::kQnnOpCodeMatMul,
    QnnOpCode::kQnnOpCodeElementWiseAdd,
};

constexpr auto kBadMatchTableGemma3MHAToSHA =
    create_bad_match_table(kGemma3MHAToSHA.data(), kGemma3MHAToSHA.size());

bool TransformMHAToSHA(std::vector<OpWrapper>& ops, size_t start_id,
                       TensorPool& tensor_pool) {
  QNN_LOG_INFO("TransformMHAToSHA");
  std::vector<OpWrapper> new_ops;
  // -> {        Mul -> Transpose -> Reshape         } -> { 2*MatMul }
  // -> { Transpose -> Reshape -> 4 * (Slice -> Mul) } -> { 8*MatMul }
  //
  const auto& mul_input = ops[start_id].GetInputTensor(0);
  const auto& mul_const = ops[start_id].GetInputTensor(1);
  const auto& transpose_perm = ops[start_id + 1].GetPararmTensor(0);
  const auto& matmul1_k_cache = ops[start_id + 3].GetInputTensor(1);
  const auto& matmul1_hma_output = ops[start_id + 3].GetOutputTensor(0);
  const auto& matmul2_k_slice = ops[start_id + 4].GetInputTensor(1);
  const auto& matmul2_hma_output = ops[start_id + 4].GetOutputTensor(0);
  const auto& concat_mha_output = ops[start_id + 5].GetOutputTensor(0);
  const auto& add_mha_mask = ops[start_id + 7].GetInputTensor(1);
  const auto& add_mha_output = ops[start_id + 7].GetOutputTensor(0);
  const auto& softmax_mha_output = ops[start_id + 9].GetOutputTensor(0);

  const auto& matmul1_v_cache = ops[start_id + 12].GetInputTensor(1);
  const auto& matmul1_next_hma_output = ops[start_id + 12].GetOutputTensor(0);
  const auto& matmul2_v_slice = ops[start_id + 13].GetInputTensor(1);
  const auto& matmul2_next_hma_output = ops[start_id + 13].GetOutputTensor(0);

  const auto& add_after_matmul_v_output = ops[start_id + 14].GetOutputTensor(0);

  // Mul: Transpose
  std::vector<::qnn::TensorWrapperRef> transpose_inputs;
  transpose_inputs.emplace_back(const_cast<::qnn::TensorWrapper&>(mul_input));
  transpose_inputs.emplace_back(
      const_cast<::qnn::TensorWrapper&>(transpose_perm));
  std::vector<::qnn::TensorWrapperRef> transpose_outputs;
  std::vector<uint32_t> transpose_output_dim = mul_input.GetDims();
  uint32_t tmp = transpose_output_dim[1];
  transpose_output_dim[1] = transpose_output_dim[2];
  transpose_output_dim[2] = tmp;
  auto& transpose_output = tensor_pool.CreateNativeTensor(
      ops[start_id].GetInputTensor(0).GetDataType(),
      ops[start_id].GetInputTensor(0).GetQuantParams(), transpose_output_dim);
  transpose_outputs.emplace_back(transpose_output);
  auto transpose =
      BuildTransposeOp(tensor_pool, transpose_inputs, transpose_outputs);
  std::move(transpose.begin(), transpose.end(), std::back_inserter(new_ops));

  // Mul: Reshape
  std::vector<::qnn::TensorWrapperRef> reshape_inputs;
  reshape_inputs.emplace_back(transpose_output);
  std::vector<::qnn::TensorWrapperRef> reshape_outputs;
  std::vector<uint32_t> reshape_output_dim = {
      transpose_output_dim[0], 1,
      transpose_output_dim[1] * transpose_output_dim[2],
      transpose_output_dim[3]};
  for (int i = 0; i < transpose_output_dim[1]; ++i) {
    reshape_outputs.emplace_back(tensor_pool.CreateNativeTensor(
        ops[start_id].GetInputTensor(0).GetDataType(),
        ops[start_id].GetInputTensor(0).GetQuantParams(), reshape_output_dim));
  }
  auto reshape = BuildReshapeOp(tensor_pool, reshape_inputs, reshape_outputs);
  std::move(reshape.begin(), reshape.end(), std::back_inserter(new_ops));
  // Mul: 4*(Slice->Mul)
  const std::vector<uint32_t> slice_size_dim{4};
  const std::array<int32_t, 4> slice_size_data{1, 1, 128, 256};
  auto& slice_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, slice_size_dim,
      slice_size_data.size() * sizeof(slice_size_data[0]),
      slice_size_data.data());

  // slice_begin for slice-matmul in for-loop
  const std::vector<uint32_t> slice1_begin_dim{4};
  const std::array<int32_t, 4> slice1_begin_data{0, 0, 0, 0};
  auto& slice1_begin = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, slice1_begin_dim,
      slice1_begin_data.size() * sizeof(slice1_begin_data[0]),
      slice1_begin_data.data());
  const std::vector<uint32_t> slice1_size_dim{4};
  const std::array<int32_t, 4> slice1_size_data{1, 1, 128, 1280};
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
  const std::array<int32_t, 4> slice2_size_data{1, 1, 128, 128};
  auto& slice2_size = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, {}, slice2_size_dim,
      slice2_size_data.size() * sizeof(slice2_size_data[0]),
      slice2_size_data.data());
  std::array<::qnn::TensorWrapper*, 4> concat_inputs_ptr;
  for (int i = 0; i < transpose_output_dim[1]; ++i) {
    std::vector<::qnn::TensorWrapperRef> slice_inputs;
    slice_inputs.emplace_back(reshape_outputs[0]);
    const std::vector<uint32_t> slice_begin_dim{4};
    const std::array<int32_t, 4> slice_begin_data{0, 0, 0, 0};
    auto& slice_begin = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_INT_32, {}, slice_begin_dim,
        slice_begin_data.size() * sizeof(slice_begin_data[0]),
        slice_begin_data.data());
    slice_inputs.emplace_back(slice_begin);
    slice_inputs.emplace_back(slice_size);
    std::vector<::qnn::TensorWrapperRef> slice_outputs;
    std::vector<uint32_t> slice_output_dim = reshape_outputs[0].get().GetDims();
    slice_output_dim[2] = slice_output_dim[2] / transpose_output_dim[1];
    auto& slice_output = tensor_pool.CreateNativeTensor(
        reshape_outputs[0].get().GetDataType(),
        reshape_outputs[0].get().GetQuantParams(), slice_output_dim);
    slice_outputs.emplace_back(slice_output);
    auto slice = BuildSliceOp(tensor_pool, slice_inputs, slice_outputs);
    std::move(slice.begin(), slice.end(), std::back_inserter(new_ops));

    // Mul
    std::vector<::qnn::TensorWrapperRef> mul_inputs;
    mul_inputs.emplace_back(slice_output);
    mul_inputs.emplace_back(const_cast<::qnn::TensorWrapper&>(mul_const));
    std::vector<::qnn::TensorWrapperRef> mul_outputs;
    auto& mul_output = tensor_pool.CreateNativeTensor(
        mul_inputs[0].get().GetDataType(), mul_inputs[0].get().GetQuantParams(),
        mul_inputs[0].get().GetDims());
    mul_outputs.emplace_back(mul_output);
    auto mul = BuildElementwiseMulOp(tensor_pool, mul_inputs, mul_outputs);
    std::move(mul.begin(), mul.end(), std::back_inserter(new_ops));
    // 2 * MatMul (8 MatMul in total)
    // MatMul1
    std::vector<::qnn::TensorWrapperRef> matmul1_inputs;
    matmul1_inputs.emplace_back(mul_output);
    matmul1_inputs.emplace_back(
        const_cast<::qnn::TensorWrapper&>(matmul1_k_cache));
    std::vector<::qnn::TensorWrapperRef> matmul1_outputs;
    std::vector<uint32_t> matmul1_output_dim = matmul1_hma_output.GetDims();
    matmul1_output_dim[2] = matmul1_output_dim[2] / transpose_output_dim[1];
    auto& matmul1_output = tensor_pool.CreateNativeTensor(
        matmul1_hma_output.GetDataType(), matmul1_hma_output.GetQuantParams(),
        matmul1_output_dim);
    matmul1_outputs.emplace_back(matmul1_output);
    auto matmul1 = BuildMatmulOp(tensor_pool, matmul1_inputs, matmul1_outputs,
                                 false, true);
    std::move(matmul1.begin(), matmul1.end(), std::back_inserter(new_ops));
    // MatMul2
    std::vector<::qnn::TensorWrapperRef> matmul2_inputs;
    matmul2_inputs.emplace_back(mul_output);
    matmul2_inputs.emplace_back(
        const_cast<::qnn::TensorWrapper&>(matmul2_k_slice));
    std::vector<::qnn::TensorWrapperRef> matmul2_outputs;
    std::vector<uint32_t> matmul2_output_dim = matmul2_hma_output.GetDims();
    matmul2_output_dim[2] = matmul2_output_dim[2] / transpose_output_dim[1];
    auto& matmul2_output = tensor_pool.CreateNativeTensor(
        matmul2_hma_output.GetDataType(), matmul2_hma_output.GetQuantParams(),
        matmul2_output_dim);
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
    auto& concat_output = tensor_pool.CreateNativeTensor(
        concat_mha_output.GetDataType(), concat_mha_output.GetQuantParams(),
        concat_output_dim);
    concat_outputs.emplace_back(concat_output);
    auto concat =
        BuildConcatenationOp(tensor_pool, concat_inputs, concat_outputs, 3);
    std::move(concat.begin(), concat.end(), std::back_inserter(new_ops));
    // Add
    std::vector<::qnn::TensorWrapperRef> add_inputs;
    add_inputs.emplace_back(concat_output);
    add_inputs.emplace_back(const_cast<::qnn::TensorWrapper&>(add_mha_mask));
    std::vector<::qnn::TensorWrapperRef> add_outputs;
    auto& add_output = tensor_pool.CreateNativeTensor(
        add_mha_output.GetDataType(), add_mha_output.GetQuantParams(),
        concat_output.GetDims());
    add_outputs.emplace_back(add_output);
    auto add = BuildElementwiseAddOp(tensor_pool, add_inputs, add_outputs);
    std::move(add.begin(), add.end(), std::back_inserter(new_ops));
    // Softmax
    std::vector<::qnn::TensorWrapperRef> softmax_inputs;
    softmax_inputs.emplace_back(add_output);
    std::vector<::qnn::TensorWrapperRef> softmax_outputs;
    auto& softmax_output = tensor_pool.CreateNativeTensor(
        softmax_mha_output.GetDataType(), softmax_mha_output.GetQuantParams(),
        add_output.GetDims());
    softmax_outputs.emplace_back(softmax_output);
    auto softmax =
        BuildSoftmaxOp(tensor_pool, softmax_inputs, softmax_outputs, 1.0f);
    std::move(softmax.begin(), softmax.end(), std::back_inserter(new_ops));
    // 2 * Slice-MatMul (8 * Slice-MatMul in total)
    // Slice-MatMul1
    std::vector<::qnn::TensorWrapperRef> slice1_inputs;
    slice1_inputs.emplace_back(softmax_output);
    slice1_inputs.emplace_back(slice1_begin);
    slice1_inputs.emplace_back(slice1_size);
    std::vector<::qnn::TensorWrapperRef> slice1_outputs;
    const std::vector<uint32_t> slice1_output_dim{1, 1, 128, 1280};
    auto& slice1_output = tensor_pool.CreateNativeTensor(
        softmax_output.GetDataType(), softmax_output.GetQuantParams(),
        slice1_output_dim);
    slice1_outputs.emplace_back(slice1_output);
    auto slice1 = BuildSliceOp(tensor_pool, slice1_inputs, slice1_outputs);
    std::move(slice1.begin(), slice1.end(), std::back_inserter(new_ops));
    // --- MatMul
    std::vector<::qnn::TensorWrapperRef> matmul1_v_inputs;
    matmul1_v_inputs.emplace_back(slice1_output);
    matmul1_v_inputs.emplace_back(
        const_cast<::qnn::TensorWrapper&>(matmul1_v_cache));
    std::vector<::qnn::TensorWrapperRef> matmul1_v_outputs;
    std::vector<uint32_t> matmul1_v_output_dim =
        matmul1_next_hma_output.GetDims();
    matmul1_v_output_dim[2] = matmul1_v_output_dim[2] / transpose_output_dim[1];
    auto& matmul1_v_output = tensor_pool.CreateNativeTensor(
        matmul1_next_hma_output.GetDataType(),
        matmul1_next_hma_output.GetQuantParams(), matmul1_v_output_dim);
    matmul1_v_outputs.emplace_back(matmul1_v_output);
    auto matmul1_v = BuildMatmulOp(tensor_pool, matmul1_v_inputs,
                                   matmul1_v_outputs, false, true);
    std::move(matmul1_v.begin(), matmul1_v.end(), std::back_inserter(new_ops));
    // Slice-MatMul2
    std::vector<::qnn::TensorWrapperRef> slice2_inputs;
    slice2_inputs.emplace_back(softmax_output);
    slice2_inputs.emplace_back(slice2_begin);
    slice2_inputs.emplace_back(slice2_size);
    std::vector<::qnn::TensorWrapperRef> slice2_outputs;
    const std::vector<uint32_t> slice2_output_dim{1, 1, 128, 128};
    auto& slice2_output = tensor_pool.CreateNativeTensor(
        softmax_output.GetDataType(), softmax_output.GetQuantParams(),
        slice2_output_dim);
    slice2_outputs.emplace_back(slice2_output);
    auto slice2 = BuildSliceOp(tensor_pool, slice2_inputs, slice2_outputs);
    std::move(slice2.begin(), slice2.end(), std::back_inserter(new_ops));
    // --- MatMul
    std::vector<::qnn::TensorWrapperRef> matmul2_v_inputs;
    matmul2_v_inputs.emplace_back(slice2_output);
    matmul2_v_inputs.emplace_back(
        const_cast<::qnn::TensorWrapper&>(matmul2_v_slice));
    std::vector<::qnn::TensorWrapperRef> matmul2_v_outputs;
    std::vector<uint32_t> matmul2_v_output_dim =
        matmul2_next_hma_output.GetDims();
    matmul2_v_output_dim[2] = matmul2_v_output_dim[2] / transpose_output_dim[1];
    auto& matmul2_v_output = tensor_pool.CreateNativeTensor(
        matmul2_next_hma_output.GetDataType(),
        matmul2_next_hma_output.GetQuantParams(), matmul2_v_output_dim);
    matmul2_v_outputs.emplace_back(matmul2_v_output);
    auto matmul2_v = BuildMatmulOp(tensor_pool, matmul2_v_inputs,
                                   matmul2_v_outputs, false, true);
    std::move(matmul2_v.begin(), matmul2_v.end(), std::back_inserter(new_ops));
    // Add
    std::vector<::qnn::TensorWrapperRef> add_final_inputs;
    add_final_inputs.emplace_back(matmul1_v_output);
    add_final_inputs.emplace_back(matmul2_v_output);
    std::vector<::qnn::TensorWrapperRef> add_final_outputs;
    std::vector<uint32_t> add_final_output_dim =
        add_after_matmul_v_output.GetDims();
    add_final_output_dim[2] = add_final_output_dim[2] / transpose_output_dim[1];
    auto& add_final_output = tensor_pool.CreateNativeTensor(
        add_after_matmul_v_output.GetDataType(),
        add_after_matmul_v_output.GetQuantParams(), add_final_output_dim);
    concat_inputs_ptr[i] = &add_final_output;
    add_final_outputs.emplace_back(add_final_output);
    auto add_final =
        BuildElementwiseAddOp(tensor_pool, add_final_inputs, add_final_outputs);
    std::move(add_final.begin(), add_final.end(), std::back_inserter(new_ops));
  }
  // Concat
  std::vector<::qnn::TensorWrapperRef> concat_final_inputs;
  for (int i = 0; i < 4; i++) {
    concat_final_inputs.emplace_back(*(concat_inputs_ptr[i]));
  }
  std::vector<::qnn::TensorWrapperRef> concat_final_outputs;
  concat_final_outputs.emplace_back(
      const_cast<::qnn::TensorWrapper&>(add_after_matmul_v_output));
  auto concat_final = BuildConcatenationOp(tensor_pool, concat_final_inputs,
                                           concat_final_outputs, 2);
  std::move(concat_final.begin(), concat_final.end(),
            std::back_inserter(new_ops));
  QNN_LOG_INFO("Add new ops");
  ops.insert(ops.begin() + start_id + 15,
             std::make_move_iterator(new_ops.begin()),
             std::make_move_iterator(new_ops.end()));
  QNN_LOG_INFO("Remove useless ops");
  // And then remove 0~14
  ops.erase(ops.begin() + start_id, ops.begin() + start_id + 15);
  return true;
}

}  // namespace

// TODO (jiunkaiy): Add more G2G transformation.
void GraphToGraphTransform(std::vector<OpWrapper>& ops,
                           TensorPool& tensor_pool) {
  // MatMul-Convert Fusion
  QNN_LOG_INFO("===== MatMul-Convert 1 ===== ");
  Transform(ops, tensor_pool, kMatMulConvertPattern1.data(),
            kMatMulConvertPattern1.size(), kBadMatchTableMatMulConvertPattern1,
            FuseMatMulConvert1);
  QNN_LOG_INFO("===== MatMul-Convert 2 ===== ");
  Transform(ops, tensor_pool, kMatMulConvertPattern2.data(),
            kMatMulConvertPattern2.size(), kBadMatchTableMatMulConvertPattern2,
            FuseMatMulConvert2);
  // TODO (jiunkaiy): MHA->SHA Transformation
  QNN_LOG_INFO("===== MHA->SHA ===== ");
  Transform(ops, tensor_pool, kGemma3MHAToSHA.data(), kGemma3MHAToSHA.size(),
            kBadMatchTableGemma3MHAToSHA, TransformMHAToSHA);
}
}  // namespace qnn
