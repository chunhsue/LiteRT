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
#include "QnnCommon.h"  // from @qairt
#include "QnnInterface.h"  // from @qairt

namespace qnn {

namespace {

constexpr size_t kQnnOpCodeSize = static_cast<size_t>(QnnOpCode::kUnknown);

std::vector<size_t> CreateBadMatchTable(
    const std::vector<QnnOpCode>& pattern_ops) {
  std::vector<size_t> table(kQnnOpCodeSize, pattern_ops.size());
  for (size_t i = 0; i < pattern_ops.size() - 1; ++i) {
    table[static_cast<size_t>(pattern_ops[i])] = pattern_ops.size() - i - 1;
  }
  return table;
}

// Returns the start ID of the matched operator pattern beginning at start_id;
// returns std::nullopt if no match is found in the remaining graph
std::optional<size_t> GetPatternStartID(
    size_t start_id, const std::vector<OpWrapper>& ops,
    const std::vector<QnnOpCode>& pattern_ops,
    const std::vector<size_t>& bad_match_table) {
  size_t end_id = start_id + (pattern_ops.size() - 1);
  while (end_id >= (pattern_ops.size() - 1) && end_id < ops.size()) {
    bool found_pattern = true;
    for (size_t i = 0; i < pattern_ops.size(); ++i) {
      if (!ops[end_id - i].IsOpCode(pattern_ops[pattern_ops.size() - i - 1])) {
        found_pattern = false;
        break;
      }
    }
    if (found_pattern) {
      return end_id - (pattern_ops.size() - 1);
    } else {
      end_id += bad_match_table[static_cast<size_t>(ops[end_id].GetOpCode())];
    }
  }
  return std::nullopt;
}

typedef size_t (*G2GTransform)(const QNN_INTERFACE_VER_TYPE* api,
                               Qnn_BackendHandle_t backend,
                               std::vector<OpWrapper>& ops, size_t start_id,
                               TensorPool& tensor_pool, size_t pattern_size);
void Transform(const QNN_INTERFACE_VER_TYPE* api, Qnn_BackendHandle_t backend,
               std::vector<OpWrapper>& ops, TensorPool& tensor_pool,
               const std::vector<QnnOpCode>& pattern_ops,
               G2GTransform custom_transform) {
  auto bad_match_table = CreateBadMatchTable(pattern_ops);
  size_t start_id = 0;
  while ((start_id + (pattern_ops.size() - 1)) < ops.size()) {
    if (auto pattern_start_id =
            GetPatternStartID(start_id, ops, pattern_ops, bad_match_table);
        pattern_start_id.has_value()) {
      start_id += custom_transform(api, backend, ops, pattern_start_id.value(),
                                   tensor_pool, pattern_ops.size());
    } else {
      break;
    }
  }
}

enum class G2GConfig {
  // Disable G2G.
  kOff,
  // Enable G2G MatMul-convert fusion.
  kMatMulConvert,
  // Enable G2G MHA optimization for prefill only.
  kMHAOptPrefill,
  // Enable G2G MHA optimization for both decode and prefill.
  kMHAOpt,
};

}  // namespace

// TODO (jiunkaiy): Add more G2G transformation.
void GraphToGraphTransform(const QNN_INTERFACE_VER_TYPE* api,
                           Qnn_BackendHandle_t backend,
                           std::vector<OpWrapper>& ops,
                           TensorPool& tensor_pool) {
  if (api == nullptr) {
    QNN_LOG_WARNING(
        "[G2G] Skip graph validation process since qnn interface is"
        "nullptr.");
  }
  // TODO(jiunkaiy): Move to LiteRtOption.
  const G2GConfig g2g_option = G2GConfig::kMHAOptPrefill;
  if (g2g_option == G2GConfig::kOff) {
    return;
  }

  // MatMul-convert Fusion
  if (g2g_option == G2GConfig::kMatMulConvert ||
      g2g_option == G2GConfig::kMHAOptPrefill ||
      g2g_option == G2GConfig::kMHAOpt) {
    const std::vector<QnnOpCode> matmul_convert_decode = {
        QnnOpCode::kMatMul,
        QnnOpCode::kConvert,
    };
    Transform(api, backend, ops, tensor_pool, matmul_convert_decode,
              FuseMatMulConvertDecode);
    const std::vector<QnnOpCode> matmul_convert_prefill = {
        QnnOpCode::kMatMul,
        QnnOpCode::kMatMul,
        QnnOpCode::kConvert,
    };
    Transform(api, backend, ops, tensor_pool, matmul_convert_prefill,
              FuseMatMulConvertPrefill);
  }
  // MHA Optimization
  if (g2g_option == G2GConfig::kMHAOpt) {
    const std::vector<QnnOpCode> gemma3_mha_decode = {
        QnnOpCode::kElementWiseMultiply,
        QnnOpCode::kMatMul,
        QnnOpCode::kMatMul,
        QnnOpCode::kConcat,
        QnnOpCode::kReshape,
        QnnOpCode::kElementWiseAdd,
        QnnOpCode::kReshape,
        QnnOpCode::kSoftmax,
        QnnOpCode::kStridedSlice,
        QnnOpCode::kStridedSlice,
        QnnOpCode::kMatMul,
        QnnOpCode::kMatMul,
        QnnOpCode::kElementWiseAdd,
        QnnOpCode::kReshape,
    };
    Transform(api, backend, ops, tensor_pool, gemma3_mha_decode,
              OptimizeMHADecode);
  }
  if (g2g_option == G2GConfig::kMHAOptPrefill ||
      g2g_option == G2GConfig::kMHAOpt) {
    const std::vector<QnnOpCode> gemma3_mha_prefill = {
        QnnOpCode::kElementWiseMultiply,
        QnnOpCode::kTranspose,
        QnnOpCode::kReshape,
        QnnOpCode::kMatMul,
        QnnOpCode::kMatMul,
        QnnOpCode::kConcat,
        QnnOpCode::kReshape,
        QnnOpCode::kElementWiseAdd,
        QnnOpCode::kReshape,
        QnnOpCode::kSoftmax,
        QnnOpCode::kStridedSlice,
        QnnOpCode::kStridedSlice,
        QnnOpCode::kMatMul,
        QnnOpCode::kMatMul,
        QnnOpCode::kElementWiseAdd,
        QnnOpCode::kReshape,
        QnnOpCode::kTranspose,
        QnnOpCode::kReshape,
    };
    Transform(api, backend, ops, tensor_pool, gemma3_mha_prefill,
              OptimizeMHAPrefill);
  }
}
}  // namespace qnn
