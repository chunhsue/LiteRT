// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <vector>

#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
namespace qnn {

namespace {
constexpr size_t kMatMulConvertRange = 2;
void FuseMatMulConvert(std::vector<OpWrapper>& ops, size_t start_id,
                       size_t end_id) {
  if (!ops[start_id].IsOpType(QNN_OP_MAT_MUL)) {
    return;
  }
  for (size_t convert_id = start_id + 1; convert_id <= end_id; ++convert_id) {
    if (convert_id >= ops.size()) {
      break;
    }
    if (ops[convert_id].IsOpType(QNN_OP_CONVERT) &&
        ops[convert_id].GetInputTensorName(0) ==
            ops[start_id].GetOutputTensorName(0)) {
      ops[start_id].StealOutputs(ops[convert_id]);
      ops.erase(ops.begin() + convert_id);
      return;
    }
  }
}
}  // namespace

// TODO (jiunkaiy): Add more G2G transformation.
void GraphToGraphTransform(std::vector<OpWrapper>& ops) {
  for (size_t op_id = 0; op_id < ops.size(); ++op_id) {
    FuseMatMulConvert(ops, op_id, op_id + kMatMulConvertRange);
  }
}
}  // namespace qnn
