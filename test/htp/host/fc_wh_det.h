// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   fc_wh_det.h
 * @date   6 Oct 2026
 * @brief  [#225] Scalar spec of the u8 row on WH weights (the MoE layer's
 *         reference) and [#258] of the decode FC on WH weights (Q8_0 row)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * fc_wh_quant_row_det / fc_wh_col_det: what the MoE layer check's
 * reference computes per expert: the row quantized to u8 with one scale and
 * zero point (the range includes 0, round to nearest even), every output
 * column's int32 sum of u8 x int4 over K, then
 * (sum - zp * colsum) * scale * w_scale + bias, the products in that order
 * with no fused multiply-add. The int4 value of a tile is wh_value
 * (standin/hvx_scalar.h, the WH layout of htp_wh_layout.h).
 *
 * fc_wh_m1_col_det: [#258] what hexkl_mm_u8i4_fc_m1_run
 * (hexkl_mm_u8i4_moe.c) and hvx_gemm_i8i4_wh_col_m1 must produce for one
 * column on the row's Q8_0 (q8_0_quant_cpu_det, the CPU's and the Q4M1
 * FC's activation): per 32-block kt the int32 s = sum q * w, then
 * acc = RN(acc + RN((float)s * d[kt])) from acc = 0, then
 * RN(RN(acc * w_scale) + bias) -- every f32 op on its own (compile with
 * -ffp-contract=off, or a C mode that implies it). gemv_native_check.c
 * holds the real HVX column against it, moe_layer_host_check.c the call.
 *
 * Scope: these are the formulas of the scalar stand-ins the host checks and
 * the in-process build run (standin/hvx_scalar.c). The real HVX GEMV's
 * int32 sums are held against the emulator by gemv_native_check.c and the
 * real dequant epilogue by geglu_host_check.c; the real HVX quantizer is
 * not held against this file anywhere on the host.
 */
#ifndef __FC_WH_DET_H__
#define __FC_WH_DET_H__

#include <math.h>
#include <stdint.h>

#include "hvx_scalar.h"      /* wh_value */
#include "q4_gemv_cpu_det.h" /* q8_0_quant_cpu_det, cpu_det_f16_to_f32 */

/** @brief One row of @a k floats to u8 with its scale and zero point. */
static inline void fc_wh_quant_row_det(const float *x, uint32_t k, uint8_t *q,
                                       float *scale, int32_t *zp) {
  float lo = 0.f, hi = 0.f;
  for (uint32_t j = 0; j < k; ++j) {
    if (x[j] < lo)
      lo = x[j];
    if (x[j] > hi)
      hi = x[j];
  }
  float s = (hi - lo) / 255.f;
  if (s <= 0.f)
    s = 1e-8f;
  *scale = s;
  long z = lrintf(-lo / s);
  if (z < 0)
    z = 0;
  if (z > 255)
    z = 255;
  *zp = (int32_t)z;
  for (uint32_t j = 0; j < k; ++j) {
    long v = lrintf(x[j] / s) + *zp;
    if (v < 0)
      v = 0;
    if (v > 255)
      v = 255;
    q[j] = (uint8_t)v;
  }
}

/** @brief Column @a col of a [k_tiles x n_tiles] WH weight on the quantized
 *  row: the int32 sum, then the dequant. */
static inline float fc_wh_col_det(const uint8_t *q, float scale, int32_t zp,
                                  const uint8_t *wh, uint32_t k_tiles,
                                  uint32_t n_tiles, uint32_t col,
                                  int32_t colsum, float w_scale, float bias) {
  const uint32_t nt = col / 32u, c = col % 32u;
  int32_t s = 0;
  for (uint32_t kt = 0; kt < k_tiles; ++kt)
    for (uint32_t k = 0; k < 32u; ++k)
      s += (int32_t)q[kt * 32u + k] *
           wh_value(wh + ((size_t)kt * n_tiles + nt) * 512u, k, c);
  return ((float)(s - zp * colsum)) * scale * w_scale + bias;
}

/** @brief [#258] Column @a col of a [k_tiles x n_tiles] WH weight on the
 *  row's Q8_0 (@a q, @a d from q8_0_quant_cpu_det). */
static inline float fc_wh_m1_col_det(const int8_t *q, const uint16_t *d,
                                     const uint8_t *wh, uint32_t k_tiles,
                                     uint32_t n_tiles, uint32_t col,
                                     float w_scale, float bias) {
  const uint32_t nt = col / 32u, c = col % 32u;
  float acc = 0.f;
  for (uint32_t kt = 0; kt < k_tiles; ++kt) {
    int32_t s = 0;
    for (uint32_t k = 0; k < 32u; ++k)
      s += (int32_t)q[kt * 32u + k] *
           wh_value(wh + ((size_t)kt * n_tiles + nt) * 512u, k, c);
    const float p = (float)s * cpu_det_f16_to_f32(d[kt]);
    acc = acc + p;
  }
  const float r = acc * w_scale;
  return r + bias;
}

#endif /* __FC_WH_DET_H__ */
