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

enum class G2GConfig : uint32_t {
  // Enable G2G
  kEnabled = 0x00000001,
  // Enable G2G MatMul-convert fusion
  kMatMulConvert = 0x00000011,
  // Enable G2G MHA optimization for prefill only
  kMHAOptPrefill = 0x00001011,
  // Enable G2G MHA optimization for decode only
  kMHAOptDecode = 0x00000111,
  // Enable G2G MHA optimization for both decode and prefill
  kMHAOpt = 0x00001111,
};

bool IsG2GOptionEQ(G2GConfig source, G2GConfig target) {
  return (static_cast<uint32_t>(source) & static_cast<uint32_t>(target)) ==
         static_cast<uint32_t>(target);
}

bool IsG2GOptionNE(G2GConfig source, G2GConfig target) {
  return !IsG2GOptionEQ(source, target);
}

}  // namespace

// TODO (jiunkaiy): Add more G2G transformation.
void GraphToGraphTransform(std::vector<OpWrapper>& ops,
                           TensorPool& tensor_pool) {
  // TODO(jiunkaiy): Move to LiteRtOption.
  const G2GConfig g2g_option = G2GConfig::kMHAOptPrefill;
  if (IsG2GOptionNE(g2g_option, G2GConfig::kEnabled)) {
    return;
  }

  // MatMul-convert Fusion
  if (IsG2GOptionEQ(g2g_option, G2GConfig::kMatMulConvert)) {
    Transform(ops, tensor_pool, kMatMulConvertDecode.data(),
              kMatMulConvertDecode.size(), kBadMatchTableMatMulConvertDecode,
              FuseMatMulConvertDecode);
    Transform(ops, tensor_pool, kMatMulConvertPrefill.data(),
              kMatMulConvertPrefill.size(), kBadMatchTableMatMulConvertPrefill,
              FuseMatMulConvertPrefill);
  }
  // MHA Optimization
  if (IsG2GOptionEQ(g2g_option, G2GConfig::kMHAOptPrefill)) {
    Transform(ops, tensor_pool, kGemma3MHAToSHADecode.data(),
              kGemma3MHAToSHADecode.size(), kBadMatchTableGemma3MHAToSHADecode,
              TransformMHAToSHA);
  }
  if (IsG2GOptionEQ(g2g_option, G2GConfig::kMHAOptDecode)) {
    Transform(ops, tensor_pool, kGemma3MHAToSHAPrefill.data(),
              kGemma3MHAToSHAPrefill.size(),
              kBadMatchTableGemma3MHAToSHAPrefill, TransformMHAToSHA);
  }
}
}  // namespace qnn
