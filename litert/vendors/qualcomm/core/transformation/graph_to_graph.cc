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

std::vector<size_t> CreateBadMatchTable(
    const std::vector<QnnOpCode>& op_codes) {
  std::vector<size_t> table(kQnnOpCodeSize, op_codes.size());
  for (size_t i = 0; i < op_codes.size() - 1; ++i) {
    table[static_cast<size_t>(op_codes[i])] = op_codes.size() - i - 1;
  }
  return table;
}

size_t FindPattern(size_t end_id, const std::vector<OpWrapper>& ops,
                   const std::vector<QnnOpCode>& op_codes,
                   const std::vector<size_t>& bad_match_table) {
  while (end_id >= (op_codes.size() - 1) && end_id < ops.size()) {
    bool found_pattern = true;
    for (size_t i = 0; i < op_codes.size(); ++i) {
      if (!ops[end_id - i].IsOpType(op_codes[op_codes.size() - i - 1])) {
        found_pattern = false;
        break;
      }
    }
    if (found_pattern) {
      return end_id;
    } else {
      end_id += bad_match_table[static_cast<size_t>(ops[end_id].GetOpCode())];
    }
  }
  return ops.size();
}

typedef bool (*G2GTransform)(std::vector<OpWrapper>&, size_t, TensorPool&);
void Transform(std::vector<OpWrapper>& ops, TensorPool& tensor_pool,
               const std::vector<QnnOpCode>& op_codes,
               G2GTransform custom_transform) {
  auto bad_match_table = CreateBadMatchTable(op_codes);
  for (const auto& e : op_codes) {
    QNN_LOG_DEBUG("Bad %d: %d", e, bad_match_table[static_cast<int>(e)]);
  }
  size_t end_id = op_codes.size() - 1;
  while (end_id < ops.size()) {
    end_id = FindPattern(end_id, ops, op_codes, bad_match_table);
    if (end_id < ops.size() &&
        custom_transform(ops, end_id - (op_codes.size() - 1), tensor_pool)) {
      QNN_LOG_INFO("[G2G] Transformation completed successfully.");
    }
    end_id += 1;
  }
}

enum class G2GConfig : uint32_t {
  // Enable G2G
  kEnabled = 0b0001,
  // Enable G2G MatMul-convert fusion
  kMatMulConvert = 0b0011,
  // Enable G2G MHA optimization for prefill only
  kMHAOptPrefill = 0b1011,
  // Enable G2G MHA optimization for decode only
  kMHAOptDecode = 0b0111,
  // Enable G2G MHA optimization for both decode and prefill
  kMHAOpt = 0b1111,
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
    Transform(ops, tensor_pool, kMatMulConvertDecode, FuseMatMulConvertDecode);
    Transform(ops, tensor_pool, kMatMulConvertPrefill,
              FuseMatMulConvertPrefill);
  }
  // MHA Optimization
  if (IsG2GOptionEQ(g2g_option, G2GConfig::kMHAOptDecode)) {
    Transform(ops, tensor_pool, kGemma3MHAToSHADecode, TransformMHAToSHA);
  }
  if (IsG2GOptionEQ(g2g_option, G2GConfig::kMHAOptPrefill)) {
    Transform(ops, tensor_pool, kGemma3MHAToSHAPrefill, TransformMHAToSHA);
  }
}
}  // namespace qnn
