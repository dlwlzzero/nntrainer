// SPDX-License-Identifier: Apache-2.0
/**
 * @file	ref_ops.h
 * @date	18 August 2026
 * @brief	Scalar C reference implementations for hexagon-sim primitive tests
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef REF_OPS_H
#define REF_OPS_H

#include <stdint.h>

/* Same formula as htp_quant_row_fp16, so results are bit-exact. */
float ref_quant_row(const __fp16 *x, int8_t *q, uint32_t k);

/* Naive scalar int32 dot product. */
int32_t ref_dot_i8(const int8_t *w, const int8_t *x, uint32_t k);

/* Reference for MATMUL_W8A8: x fp16[m][k], w int8[n][k], sw fp32[n],
 * y fp16[m][n]. Quantizes each x row with ref_quant_row, then dots. */
void ref_matmul_w8a8(const __fp16 *x, const int8_t *w, const float *sw,
                     __fp16 *y, uint32_t m, uint32_t k, uint32_t n);

/* Reference for RMSNORM: x/gamma/y fp16[m][n] (gamma repeats every `chunk`
 * elements, chunk == n for whole-row or head_dim for PER_HEAD QK-Norm).
 * Per row, per chunk: r = 1/sqrt(mean(x^2) + eps), y_i = x_i * r * gamma. */
void ref_rmsnorm(const __fp16 *x, const __fp16 *gamma, __fp16 *y, uint32_t m,
                 uint32_t n, uint32_t chunk, float eps);

#endif /* REF_OPS_H */
