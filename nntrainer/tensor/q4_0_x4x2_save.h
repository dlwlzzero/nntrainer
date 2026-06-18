// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_0_x4x2_save.h
 * @date	28 May 2026
 * @brief	Save-time helper that quantizes an FP32 weight to plain Q4_0 and
 *              repacks it into the DSP-native x4x2 byte layout.
 *
 *              Declared with POD-only parameter types on purpose: q4_0_utils.h
 *              defines global-scope block_q4_0 / QK_0 / block templates that
 *              collide with the identically-named (but differently-typed)
 *              definitions in the ggml backend headers. Routing the x4x2 save
 *              through this thin shim keeps q4_0_utils.h out of widely-included
 *              headers such as layer_devel.h, so no translation unit ends up
 *              with both definitions.
 * @see		https://github.com/nntrainer/nntrainer
 */

#ifndef __Q4_0_X4X2_SAVE_H__
#define __Q4_0_X4X2_SAVE_H__
#ifdef __cplusplus

#include <cstdint>

namespace nntrainer {

/**
 * @brief Quantize an FP32 weight to plain Q4_0 and repack it into x4x2 bytes.
 * @param[in] weight_fp32 source FP32 weights, row-major [N rows x K cols]
 * @param[out] dst_x4x2 destination buffer for x4x2 row-strided bytes; must hold
 *             N * (K/2 + (K/256)*16) bytes (== the Q4_0 byte count)
 * @param[in] N number of output rows
 * @param[in] K number of input columns (must be divisible by 256)
 */
void quantizeAndRepackQ4_0X4x2(const float *weight_fp32, uint8_t *dst_x4x2,
                               unsigned int N, unsigned int K);

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __Q4_0_X4X2_SAVE_H__ */
