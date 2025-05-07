// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_SHA_TO_MHA_H_
#define ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_SHA_TO_MHA_H_

#include <array>
#include <vector>

#include "litert/vendors/qualcomm/core/op_code.h"
#include "litert/vendors/qualcomm/core/tensor_pool.h"
#include "litert/vendors/qualcomm/core/utils/log.h"
#include "litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
namespace qnn {

constexpr std::array<QnnOpCode, 18> kGemma3MHAToSHAPrefill = {
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

constexpr std::array<QnnOpCode, 14> kGemma3MHAToSHADecode = {
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

bool TransformMHAToSHA(std::vector<OpWrapper>& ops, size_t start_id,
                       TensorPool& tensor_pool);

}  // namespace qnn

#endif  // ODML_LITERT_LITERT_VENDORS_QUALCOMM_CORE_TRANSFORMATION_SHA_TO_MHA_H_
