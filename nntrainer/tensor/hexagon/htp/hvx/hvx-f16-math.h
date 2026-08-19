// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-f16-math.h
 * @date	18 August 2026
 * @brief	fp16 widening dot / sum-of-squares and scale-multiply HVX
 *		helpers, shared by RMSNORM (whole-row and QK-Norm) and ATTN
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef HVX_F16_MATH_H
#define HVX_F16_MATH_H

#include <stdint.h>

#include "hvx-base.h"
#include "hvx-types.h"

/* Dot product sum(a_i * b_i) in fp32, widening via Q6_Wqf32_vmpy_VhfVhf
 * (one Vhf vector is VLEN_FP16=64 halves; the widened product pair holds
 * 32+32 qf32 lanes). n must be a multiple of 64 (validator guarantees
 * hidden%64==0 and head_dim=128) and a/b must be 128B aligned. */
static inline float hvx_dot_fp16(const __fp16 *a, const __fp16 *b,
                                 uint32_t n) {
  HVX_VectorPair p0 = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(a), hvx_vmem(b));
  HVX_Vector acc_lo = Q6_V_lo_W(p0);
  HVX_Vector acc_hi = Q6_V_hi_W(p0);

  for (uint32_t i = VLEN_FP16; i < n; i += VLEN_FP16) {
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(a + i), hvx_vmem(b + i));
    acc_lo = Q6_Vqf32_vadd_Vqf32Vqf32(acc_lo, Q6_V_lo_W(p));
    acc_hi = Q6_Vqf32_vadd_Vqf32Vqf32(acc_hi, Q6_V_hi_W(p));
  }

  float __attribute__((aligned(VLEN))) buf[VLEN_FP32];
  float sum = 0.f;
  hvx_vec_store_a(buf, VLEN, Q6_Vsf_equals_Vqf32(acc_lo));
  for (uint32_t i = 0; i < VLEN_FP32; ++i)
    sum += buf[i];
  hvx_vec_store_a(buf, VLEN, Q6_Vsf_equals_Vqf32(acc_hi));
  for (uint32_t i = 0; i < VLEN_FP32; ++i)
    sum += buf[i];
  return sum;
}

/* Sum of x_i^2 in fp32: dot of x with itself. Same alignment/n rules. */
static inline float hvx_sumsq_fp16(const __fp16 *x, uint32_t n) {
  return hvx_dot_fp16(x, x, n);
}

/* y_i = (fp16)(x_i * r * g_i): r splatted to hf, two-step qf16 multiply
 * (x*r then *g) so the fp16 rounding only happens once, at the end.
 * n must be a multiple of 64; x/g/y must be 128B aligned. */
static inline void hvx_scale_mul_fp16(const __fp16 *x, const __fp16 *g, float r,
                                      __fp16 *y, uint32_t n) {
  HVX_Vector rv = hvx_vec_splat_f16((__fp16)r);
  for (uint32_t i = 0; i < n; i += VLEN_FP16) {
    HVX_Vector xv = hvx_vmem(x + i);
    HVX_Vector gv = hvx_vmem(g + i);
    HVX_Vector xr = Q6_Vqf16_vmpy_VhfVhf(xv, rv);
    HVX_Vector xrg = Q6_Vqf16_vmpy_Vqf16Vhf(xr, gv);
    hvx_vmem(y + i) = Q6_Vhf_equals_Vqf16(xrg);
  }
}

#endif /* HVX_F16_MATH_H */
