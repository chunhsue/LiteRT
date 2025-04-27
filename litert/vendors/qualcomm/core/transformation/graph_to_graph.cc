// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "litert/vendors/qualcomm/core/builders/op_code.h"
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
    QNN_LOG_INFO("INIT");
    end_id = pattern_length - 1;
    return false;
  }
  if (end_id >= ops.size()) {
    QNN_LOG_INFO("END");
    end_id = ops.size();
    return false;
  }
  QNN_LOG_INFO("%d/%d", end_id, ops.size());
  for (size_t i = 0; i < pattern_length; ++i) {
    if (!ops[end_id - i].IsOpType(op_codes[pattern_length - i - 1])) {
      QNN_LOG_INFO("Miss %d %s", end_id - i,
                   GetQnnOpType(ops[end_id - i].GetOpCode()));
      end_id += bad_match_table[static_cast<size_t>(ops[end_id].GetOpCode())];
      return false;
    }
  }
  QNN_LOG_INFO("Get it");
  return true;
}

typedef bool (*G2GTransform)(std::vector<OpWrapper>&, size_t);
void Transform(std::vector<OpWrapper>& ops, const QnnOpCode* op_codes,
               size_t pattern_length,
               const std::array<size_t, kQnnOpCodeSize>& bad_match_table,
               G2GTransform custom_fuse) {
  for (int i = 0; i < pattern_length; ++i) {
    QNN_LOG_INFO("Bad table value of %s is %d.", GetQnnOpType(op_codes[i]),
                 bad_match_table[static_cast<int>(op_codes[i])])
  }
  size_t end_id = 0;
  while (end_id != ops.size()) {
    QNN_LOG_INFO("Current end_id %d", end_id);
    if (MatchPattern(end_id, ops, op_codes, pattern_length, bad_match_table)) {
      if (!custom_fuse(ops, end_id - (pattern_length - 1))) {
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

bool FuseMatMulConvert1(std::vector<OpWrapper>& ops, size_t start_id) {
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

bool FuseMatMulConvert2(std::vector<OpWrapper>& ops, size_t start_id) {
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

}  // namespace

// TODO (jiunkaiy): Add more G2G transformation.
void GraphToGraphTransform(std::vector<OpWrapper>& ops) {
  // MatMul-Convert Fusion
  QNN_LOG_INFO("===== MatMul-Convert 1 ===== ");
  Transform(ops, kMatMulConvertPattern1.data(), kMatMulConvertPattern1.size(),
            kBadMatchTableMatMulConvertPattern1, FuseMatMulConvert1);
  QNN_LOG_INFO("===== MatMul-Convert 2 ===== ");
  Transform(ops, kMatMulConvertPattern2.data(), kMatMulConvertPattern2.size(),
            kBadMatchTableMatMulConvertPattern2, FuseMatMulConvert2);
  // TODO (jiunkaiy): MHA->SHA Transformation
}
}  // namespace qnn
