// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_swiglu_f32.h
 * @date   08 Sep 2026
 * @brief  In-place SwiGLU / GeGLU / tanh over f32 rows (hvx_swiglu_det.h)
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_SWIGLU_F32_H__
#define __NNTRAINER_HVX_SWIGLU_F32_H__

#include <stdint.h>

#include "hvx_worker_pool.h"

/**
 * @brief gate[r][j] = silu(gate[r][j]) * up[r][j], in place, for every row.
 *
 * silu(x) = x / (1 + exp(-x)), matching avx2_impl.cpp's swiglu reference
 * element for element up to the HVX exp/reciprocal's ~1e-6 relative error.
 * The result feeds a uint8 requantization downstream, so the fused layer is
 * gated on SNR, not bit equality (doc 43 §[L2]).
 *
 * Rows are independent, so @a pool splits by row range; NULL runs
 * single-threaded. @a gate and @a up must not overlap.
 *
 * @param[in,out] gate  m_valid rows by n_out columns, row-major f32
 * @param[in]     up    same shape; read-only
 * @param[in]     m_valid  rows to process
 * @param[in]     n_out    columns per row
 */
void hvx_swiglu_inplace_f32(float *gate, const float *up, uint32_t m_valid,
                            uint32_t n_out, hvx_worker_pool *pool);

/**
 * @brief gate[r][j] = gelu_tanh(gate[r][j]) * up[r][j] -- Gemma4's GeGLU --
 *        with hvx_swiglu_inplace_f32's shape, pool and aliasing contract.
 *
 * Bit-identical to swiglu_det.h's geglu_det_one (hvx_swiglu_det.h has the
 * spec), for the same reason SwiGLU is: the result feeds a requantization.
 * |gate| <= 1e12 (hvx_geglu_det_sf's domain).
 */
void hvx_geglu_inplace_f32(float *gate, const float *up, uint32_t m_valid,
                           uint32_t n_out, hvx_worker_pool *pool);

/**
 * @brief x[i] = out_scale * tanh(x[i] * in_scale), in place, on the calling
 *        thread. (1, 1) is tanh; (1/cap, cap) is Gemma's logit softcap.
 *
 * Bit-identical to swiglu_det.h's tanh_det_one. Any n; the tail runs the
 * scalar spec.
 */
void hvx_tanh_inplace_f32(float *x, uint32_t n, float in_scale,
                          float out_scale);

#endif /* __NNTRAINER_HVX_SWIGLU_F32_H__ */
