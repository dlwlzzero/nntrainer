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

/* Fills a ROPE cos/sin table: table[p][i] = cos(p*inv_freq_i),
 * table[p][64+i] = sin(p*inv_freq_i), inv_freq_i = theta^(-2i/128),
 * for p in [0,max_seq), i in [0,64). Row stride is 128 halves. */
void ref_rope_table_fill(__fp16 *table, uint32_t max_seq, float theta);

/* Reference for ROPE: x fp16[m][heads*128] in place, table fp16[*][128]
 * (row = cos[64] || sin[64]). Per token t, head, i<64: x0=x[i], x1=x[i+64]
 * rotated using the table row at pos+t (rotate-half pair (i, i+64)). */
void ref_rope(__fp16 *x, const __fp16 *table, uint32_t m, uint32_t heads,
              uint32_t pos);

/* Reference for ADD: y = a + b, fp16[count]. */
void ref_add(const __fp16 *a, const __fp16 *b, __fp16 *y, uint32_t count);

/* Reference for SILU_MUL: silu(x) = x / (1 + expf(-x)) in fp32,
 * y = (fp16)(silu(g) * u), fp16[count]. */
void ref_silu_mul(const __fp16 *g, const __fp16 *u, __fp16 *y, uint32_t count);

#endif /* REF_OPS_H */
