// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_MATMUL_CONVERT_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_MATMUL_CONVERT_H_

#include <array>
#include <vector>

#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

constexpr std::array<QnnOpCode, 2> kMatMulConvertDecode = {
    QnnOpCode::kMatMul,
    QnnOpCode::kConvert,
};

bool FuseMatMulConvertDecode(std::vector<OpWrapper>& ops, size_t start_id,
                             TensorPool& tensor_pool);

constexpr std::array<QnnOpCode, 3> kMatMulConvertPrefill = {
    QnnOpCode::kMatMul,
    QnnOpCode::kMatMul,
    QnnOpCode::kConvert,
};

bool FuseMatMulConvertPrefill(std::vector<OpWrapper>& ops, size_t start_id,
                              TensorPool& tensor_pool);

}  // namespace qnn
#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_MATMUL_CONVERT_H_
