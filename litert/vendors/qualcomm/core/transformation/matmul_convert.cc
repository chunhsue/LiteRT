// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "litert/vendors/qualcomm/core/transformation/matmul_convert.h"

#include <array>
#include <vector>

#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "QnnInterface.h"  // from @qairt

namespace qnn {

size_t FuseMatMulConvertDecode(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_id, TensorPool& tensor_pool,
    size_t pattern_size) {
  // Connection check
  if (ops[start_id].GetOutputTensor(0) != ops[start_id + 1].GetInputTensor(0)) {
    return 1;
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] MatMul-convert fusion (Decode)");
  ops[start_id].SwapOutputs(ops[start_id + 1]);
  if (validate_op_config(ops[start_id])) {
    ops.erase(ops.begin() + start_id + 1);
  } else {
    QNN_LOG_WARNING(
        "[G2G] Validation failed. Rolling back to the original graph.");
    ops[start_id].SwapOutputs(ops[start_id + 1]);
  }
  return 1;
}

size_t FuseMatMulConvertPrefill(
    std::function<bool(OpWrapper&)> validate_op_config,
    std::vector<OpWrapper>& ops, size_t start_id, TensorPool& tensor_pool,
    size_t pattern_size) {
  // Connection check
  if (ops[start_id].GetOutputTensor(0) != ops[start_id + 2].GetInputTensor(0)) {
    return 1;
  }
  // Graph transform
  QNN_LOG_INFO("[G2G] MatMul-convert fusion (Prefill)");
  ops[start_id].SwapOutputs(ops[start_id + 2]);
  if (validate_op_config(ops[start_id])) {
    ops.erase(ops.begin() + start_id + 2);
  } else {
    QNN_LOG_WARNING(
        "[G2G] Validation failed. Rolling back to the original graph.");
    ops[start_id].SwapOutputs(ops[start_id + 1]);
  }
  return 1;
}

}  // namespace qnn
