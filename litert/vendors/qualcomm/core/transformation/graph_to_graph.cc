// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <vector>

#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/transformation/matmul_convert.h"
#include "litert/vendors/qualcomm/core/transformation/mha_to_sha.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

namespace {

constexpr size_t kQnnOpCodeSize = static_cast<size_t>(QnnOpCode::kUnknown);

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
    end_id = pattern_length - 1;
    return false;
  }
  if (end_id >= ops.size()) {
    end_id = ops.size();
    return false;
  }
  for (size_t i = 0; i < pattern_length; ++i) {
    if (!ops[end_id - i].IsOpType(op_codes[pattern_length - i - 1])) {
      end_id += bad_match_table[static_cast<size_t>(ops[end_id].GetOpCode())];
      return false;
    }
  }
  return true;
}

typedef bool (*G2GTransform)(std::vector<OpWrapper>&, size_t, TensorPool&);
void Transform(std::vector<OpWrapper>& ops, TensorPool& tensor_pool,
               const QnnOpCode* op_codes, size_t pattern_length,
               const std::array<size_t, kQnnOpCodeSize>& bad_match_table,
               G2GTransform custom_transform) {
  size_t end_id = 0;
  while (end_id != ops.size()) {
    if (MatchPattern(end_id, ops, op_codes, pattern_length, bad_match_table)) {
      if (!custom_transform(ops, end_id - (pattern_length - 1), tensor_pool)) {
        end_id += pattern_length;
      } else {
        end_id += 1;
        QNN_LOG_INFO("[G2G] Transformation completed successfully.");
      }
    }
  }
}

// MatMul-Convert Fusion
constexpr auto kBadMatchTableMatMulConvertDecode = create_bad_match_table(
    kMatMulConvertDecode.data(), kMatMulConvertDecode.size());
constexpr auto kBadMatchTableMatMulConvertPrefill = create_bad_match_table(
    kMatMulConvertPrefill.data(), kMatMulConvertPrefill.size());

// MHA-to-SHA Transformation
constexpr auto kBadMatchTableGemma3MHAToSHAPrefill = create_bad_match_table(
    kGemma3MHAToSHAPrefill.data(), kGemma3MHAToSHAPrefill.size());
constexpr auto kBadMatchTableGemma3MHAToSHADecode = create_bad_match_table(
    kGemma3MHAToSHADecode.data(), kGemma3MHAToSHADecode.size());

}  // namespace

// TODO (jiunkaiy): Add more G2G transformation.
void GraphToGraphTransform(std::vector<OpWrapper>& ops,
                           TensorPool& tensor_pool) {
  // MatMul-Convert Fusion
  Transform(ops, tensor_pool, kMatMulConvertDecode.data(),
            kMatMulConvertDecode.size(), kBadMatchTableMatMulConvertDecode,
            FuseMatMulConvertDecode);
  Transform(ops, tensor_pool, kMatMulConvertPrefill.data(),
            kMatMulConvertPrefill.size(), kBadMatchTableMatMulConvertPrefill,
            FuseMatMulConvertPrefill);

  // TODO (jiunkaiy): MHA->SHA Transformation
  Transform(ops, tensor_pool, kGemma3MHAToSHADecode.data(),
            kGemma3MHAToSHADecode.size(), kBadMatchTableGemma3MHAToSHADecode,
            TransformMHAToSHA);
  Transform(ops, tensor_pool, kGemma3MHAToSHAPrefill.data(),
            kGemma3MHAToSHAPrefill.size(), kBadMatchTableGemma3MHAToSHAPrefill,
            TransformMHAToSHA);
}
}  // namespace qnn
