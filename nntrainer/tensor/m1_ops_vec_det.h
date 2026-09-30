// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   m1_ops_vec_det.h
 * @date   30 Sep 2026
 * @brief  [#194 L2 / L3] The vector numerics of the M=1 router, norms,
 *         conv gate and SwiGLU as a scalar spec (htp_moe_ppl only)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * WHY THIS EXISTS
 *
 * m1_ops_det.h is the Android CPU's order, bought with scalar sffma chains
 * and scalar exp / divide (#164, rule 55); on htp_moe_ppl the gate is the
 * decode PPL, so these ops run on 32-lane vectors with one IEEE f32
 * operation per step (a Q6_Vsf_* pair on HVX, silicon-checked, rule 24).
 * This header is what hvx_m1_ops_f32.c's *_vec kernels compute, bit for bit
 * on hvx_emu (test/htp/host/m1_ops_host_check.c, and graph_host_check.c
 * through the graph's HTP_GRAPH_OP_VEC ops); the host check also measures
 * the distance to m1_ops_det.h (SNR) and counts routing flips. No device
 * gtest holds the kernels to it yet: on silicon they are read through the
 * sitting's decode PPL (plan 194 P1 / P2), which is this branch's gate.
 *
 * THE PIECES (every op RN; "lanes" are the 32 lanes of one vector):
 *   sumsq    per lane l, acc_l += x[32 j + l]^2 over j (a multiply, then
 *            an add); then a rotate-add tree over the lanes at distances
 *            16, 8, 4, 2, 1 (lane l += lane (l + d) mod 32); lane 0
 *   rmsnorm  r = m1_norm_scale_det(sumsq) (the CPU's exact sqrt and
 *            reciprocal, one scalar a row), y = (x * r) * gamma
 *   sigmoid  swiglu_det_recip(1 + swiglu_det_exp(0 - v)): the exp and the
 *            Newton reciprocal of swiglu_det.h, whose HVX form
 *            (hvx_swiglu_det.h) HvxSwigluDet already holds bit for bit on
 *            silicon (in place of the CPU's exp_ps / expf and true divide)
 *   swiglu   swiglu_det_one: (y * sigmoid(y)) * z
 *   conv     g = a * c; y = w0 * g; y += w1 * s1; y += w2 * s0; out = b * y
 *            (the CPU's two fmas become a multiply and an add each)
 *   router   logits: four partial sums over k = 4 i + j, acc_j += x[k] *
 *            w[k][e], then (acc0 + acc1) + (acc2 + acc3); sig = sigmoid,
 *            score = sig + bias, then m1_router_cpu_pick unchanged
 *
 * DOMAIN. Finite inputs; the norm's as m1_ops_det.h's, the exp's as
 * swiglu_det.h's (its argument clamped to [-88, 85]).
 */

#ifndef __NNTRAINER_M1_OPS_VEC_DET_H__
#define __NNTRAINER_M1_OPS_VEC_DET_H__

#include "m1_ops_det.h"
#include "swiglu_det.h"

/** @brief Lanes of one HVX vector of f32. */
#define M1V_LANES 32u

/** @brief The rotate-add tree: lane l += lane (l + d) mod 32, d = 16 .. 1,
 *         all lanes at once; the sum lands in every lane, lane 0 returned. */
static inline float m1v_tree(const float *lane) {
  float t[M1V_LANES], u[M1V_LANES];
  for (uint32_t l = 0; l < M1V_LANES; ++l) {
    t[l] = lane[l];
  }
  for (uint32_t d = M1V_LANES / 2u; d >= 1u; d >>= 1) {
    for (uint32_t l = 0; l < M1V_LANES; ++l) {
      u[l] = m1_det_add(t[l], t[(l + d) % M1V_LANES]);
    }
    for (uint32_t l = 0; l < M1V_LANES; ++l) {
      t[l] = u[l];
    }
  }
  return t[0];
}

/** @brief Sum of squares of @a chunk floats (chunk % 32 == 0). */
static inline float m1v_sumsq(const float *x, uint32_t chunk) {
  float acc[M1V_LANES];
  for (uint32_t l = 0; l < M1V_LANES; ++l) {
    acc[l] = 0.0f;
  }
  for (uint32_t j = 0; j < chunk / M1V_LANES; ++j) {
    for (uint32_t l = 0; l < M1V_LANES; ++l) {
      const float v = x[j * M1V_LANES + l];
      acc[l] = m1_det_add(acc[l], m1_det_mul(v, v));
    }
  }
  return m1v_tree(acc);
}

/** @brief RMSNorm over n floats in chunks of @a chunk (m1_rmsnorm_det's
 *         shape), the vector sum of squares. */
static inline void m1v_rmsnorm(const float *x, const float *gamma, float *y,
                               uint32_t n, uint32_t chunk, float eps) {
  for (uint32_t c = 0; c * chunk < n; ++c) {
    const float *xc = x + (size_t)c * chunk;
    const float r = m1_norm_scale_det(m1v_sumsq(xc, chunk), chunk, eps);
    for (uint32_t i = 0; i < chunk; ++i) {
      y[(size_t)c * chunk + i] = m1_det_mul(m1_det_mul(xc[i], r), gamma[i]);
    }
  }
}

/** @brief 1 / (1 + exp(-v)): swiglu_det.h's exp and reciprocal, the
 *         silicon-checked vector SwiGLU's own (HvxSwigluDet). */
static inline float m1v_sigmoid(float v) {
  return swiglu_det_recip(
    swiglu_det_add(1.0f, swiglu_det_exp(swiglu_det_sub(0.0f, v))));
}

/** @brief out = silu(y) * z, swiglu_det_one per element
 *         (m1_swiglu_cpu_det's arguments). */
static inline void m1v_swiglu(const float *y, const float *z, float *out,
                              uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    out[i] = swiglu_det_one(y[i], z[i]);
  }
}

/** @brief m1_conv_gate_det's arguments, unfused taps. */
static inline void m1v_conv_gate(const float *abc, float *state, const float *w,
                                 float *out, uint32_t C) {
  const float *a = abc, *b = abc + C, *c = abc + 2u * C;
  const float *w0 = w, *w1 = w + C, *w2 = w + 2u * C;
  float *s0 = state, *s1 = state + C;
  for (uint32_t j = 0; j < C; ++j) {
    const float g = m1_det_mul(a[j], c[j]);
    float y = m1_det_mul(w0[j], g);
    y = m1_det_add(y, m1_det_mul(w1[j], s1[j]));
    y = m1_det_add(y, m1_det_mul(w2[j], s0[j]));
    out[j] = m1_det_mul(b[j], y);
    s0[j] = s1[j];
    s1[j] = g;
  }
}

/** @brief The router's logit of expert e over w [K][E] (m1_router_cpu_det's
 *         layout), K % 4 == 0. */
static inline float m1v_router_logit(const float *x, const float *w, uint32_t K,
                                     uint32_t E, uint32_t e) {
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint32_t k = 0; k < K; ++k) {
    acc[k % 4u] =
      m1_det_add(acc[k % 4u], m1_det_mul(x[k], w[(size_t)k * E + e]));
  }
  return m1_det_add(m1_det_add(acc[0], acc[1]), m1_det_add(acc[2], acc[3]));
}

/** @brief m1_router_cpu_det's arguments and outputs, the vector numerics. */
static inline void m1v_router(const float *x, const float *w, const float *bias,
                              uint32_t K, uint32_t E, uint32_t top_k,
                              float *logits, uint32_t *sel, float *weight) {
  float sig[M1_DET_ROUTER_MAX_E], score[M1_DET_ROUTER_MAX_E];
  for (uint32_t e = 0; e < E; ++e) {
    logits[e] = m1v_router_logit(x, w, K, E, e);
    sig[e] = m1v_sigmoid(logits[e]);
    score[e] = m1_det_add(sig[e], bias[e]);
  }
  m1_router_cpu_pick(sig, score, E, top_k, sel, weight);
}

#endif /* __NNTRAINER_M1_OPS_VEC_DET_H__ */
