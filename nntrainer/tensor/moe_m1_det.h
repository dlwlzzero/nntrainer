// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   moe_m1_det.h
 * @date   29 Sep 2026
 * @brief  The scalar specification of one routed expert of the M=1 MoE
 *         call (hexkl_mm_u8i4_moe.c's use_m1 path), with a NEON twin that
 *         produces the same bits, so the CPU can compute some of a decode
 *         call's experts and the sum stays the DSP's (#157)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * WHY THIS EXISTS
 *
 * A decode MoE call on the DSP reads four experts' weights, 22 MB, while
 * the CPU waits. The CPU can read DDR at the same time, so the call can be
 * split: the DSP keeps the first k active experts in ascending expert id
 * and returns their partial sum S_k, and the CPU computes the rest and
 * continues the sum in the same order. That is bit-identical to the DSP
 * alone only if every CPU step rounds exactly as the DSP's does. Plan
 * docs/plans/157-expert-split.md section 0 lists the DSP's steps and shows
 * (ISS, 0 of ~1 M) that the skel's qf32-lowered code equals plain IEEE f32
 * with two roundings per multiply-add on normal values. This header is
 * that IEEE sequence.
 *
 * THE SPECIFICATION, per expert, one token row x[K] (plan section 0.1)
 *
 *   1  act params (hvx_quant_u8.c quant_row_params_one):
 *        mn, mx = min, max of x;  rmin = mn < 0 ? mn : 0;  rmax = mx > 0 ?
 *        mx : 0;  rmin == rmax -> s = 1, z = 0;  else s = (rmax - rmin) /
 *        255, z = clamp(nearbyint(-rmin / s), 0, 255)
 *   2  act quant (quant_pack_group4): inv = 1 / s;  q = clamp(rne(x * inv)
 *        + z, 0, 255),  rne = round to nearest even (|x * inv| <= 256)
 *   3  GEMV (hvx_gemm_u8i4_wh.c): acc[c] = sum_k q[k] * w[k][c], int32,
 *        exact in any order (|acc| <= 255 * 8 * K < 2^24 for K <= 8192)
 *   4  dequant (hvx_dequant_i32.c dq_row_sf / DQ_TILE_ROW):
 *        c = (f32)acc - (f32)z * (f32)colsum  (exact: every term < 2^24),
 *        r = ((c * s) * w_scale) + bias       (three roundings, this order)
 *   5  SwiGLU on the gate/up pair (gate column j, up column inter + j):
 *        g = swiglu_det_one(gate, up)          (swiglu_det.h)
 *   6  requant of g[inter]: steps 1 and 2
 *   7  down GEMV and dequant: steps 3 and 4 -> res[N_out]
 *   8  merge (hvx_scale_add_rows_f32), experts in ascending id from +0:
 *        out = out + (res * weight)            (two roundings, no FMA)
 *
 * The divides (steps 1, 2) are single correctly rounded f32 divides: the
 * skel's inline divide is, and so is the CPU's fdiv. Everything else is
 * one IEEE multiply, add or subtract, each through a volatile (scalar) or
 * swiglu_det.h's empty-asm barrier (NEON), so neither -ffp-contract nor
 * the app's -ffast-math (reassociation, reciprocal math) can fuse or
 * reorder them: the aarch64 object of the one TU that runs this in the app
 * (htp_moe_cpu.cpp) is gated at 0 fmla / fmls.
 *
 * THE GEMV ON NEON. A WH quarter (128 bytes, rows 8g .. 8g+7 of a 32 x 32
 * tile) holds column c's four bytes at c*4 .. c*4+3: low nibbles are rows
 * 8g .. 8g+3, high nibbles 8g+4 .. 8g+7 (htp_wh_layout.h). A 16-byte load
 * is four columns x four k, the operand shape of sdot by element.
 * (v << 4) and (v & 0xF0), read as s8, are 16*w for the two halves; the u8
 * activation enters as q ^ 0x80 = q - 128 (s8). So the sdot sum over K is
 * 16 * sum (q - 128) w, and acc = (sum >> 4) + 128 * colsum, exactly --
 * which needs colsum to be the true column sum, as the dequant's does.
 *
 * OUT OF SCOPE: subnormals (LEDGER rules 24, 37; plan section 0.1 row 5)
 * and NaN / inf inputs. The device dump gate of #157 would catch the first.
 */

#ifndef __NNTRAINER_MOE_M1_DET_H__
#define __NNTRAINER_MOE_M1_DET_H__

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "swiglu_det.h"

#if defined(__aarch64__) && defined(__ARM_NEON) &&                             \
  defined(__ARM_FEATURE_DOTPROD)
#define MOE_M1_DET_HAS_SDOT 1
#endif

/** @brief WH tile side and bytes (htp_wh_layout.h, WEIGHT_TILE_BYTES_U8I4). */
#define MOE_M1_TILE 32u
#define MOE_M1_TILE_BYTES 512u
/** @brief Tiles per kt-outer pass of the GEMV: its accumulators (2 KB)
 *         stay in L1 while one k-tile row of the weight streams by. */
#define MOE_M1_BLOCK 16u

/** @brief One expert's two WH weights, as the arena and the DSP hold them:
 *         gate_up is K x 2*inter (gate columns, then up), down inter x N. */
typedef struct {
  const uint8_t *gu_wh;
  const float *gu_ws;
  const int32_t *gu_cs;
  const float *gu_b;
  const uint8_t *dn_wh;
  const float *dn_ws;
  const int32_t *dn_cs;
  const float *dn_b;
} moe_m1_weights;

/**
 * @brief One f32 divide, rounded once. On aarch64 the fdiv is written out:
 *        under -ffast-math (the app's flags) a volatile result is not
 *        enough, since reciprocal math may still turn x / 255 into
 *        x * (1 / 255) before the store -- two roundings, other bits (seen
 *        as 3 fdiv fewer in this header's object at -O3 -ffast-math).
 */
static inline float moe_m1_div(float a, float b) {
#if defined(__aarch64__)
  float r;
  __asm__("fdiv %s0, %s1, %s2" : "=w"(r) : "w"(a), "w"(b));
  return r;
#else
  volatile float r = a / b;
  return r;
#endif
}

/** @brief rne(v) as the DSP writes it (hvx_convert.h): the bits of
 *         v + 1.5 * 2^23 minus those of 1.5 * 2^23. Valid for |v| <= 2^22. */
static inline int32_t moe_m1_rne(float v) {
  volatile float b = v + 12582912.0f;
  float bb = b;
  int32_t i;
  memcpy(&i, &bb, sizeof(i));
  return i - 0x4B400000;
}

/** @brief Step 1: the row's u8 scale and zero point. */
static inline void moe_m1_act_params(const float *x, uint32_t k, float *scale,
                                     int32_t *zp) {
  float mn = x[0], mx = x[0];
  uint32_t i = 0;
#ifdef SWIGLU_DET_HAS_NEON
  if (k >= 4u) {
    float32x4_t vmn = vld1q_f32(x), vmx = vmn;
    for (i = 4u; i + 4u <= k; i += 4u) {
      const float32x4_t v = vld1q_f32(x + i);
      vmn = vminq_f32(vmn, v);
      vmx = vmaxq_f32(vmx, v);
    }
    mn = vminvq_f32(vmn);
    mx = vmaxvq_f32(vmx);
  }
#endif
  for (; i < k; ++i) {
    if (x[i] < mn)
      mn = x[i];
    if (x[i] > mx)
      mx = x[i];
  }
  const float rmin = mn < 0.0f ? mn : 0.0f;
  const float rmax = mx > 0.0f ? mx : 0.0f;
  *scale = 1.0f;
  *zp = 0;
  if (rmin == rmax)
    return;
  const float s = moe_m1_div(swiglu_det_sub(rmax, rmin), 255.0f);
  int32_t z = (int32_t)nearbyintf(moe_m1_div(-rmin, s));
  if (z < 0)
    z = 0;
  if (z > 255)
    z = 255;
  *scale = s;
  *zp = z;
}

/** @brief Step 2: q[i] (u8) and qs[i] = q[i] ^ 0x80 (the s8 the sdot
 *         reads; the scalar GEMV ignores it). */
static inline void moe_m1_act_quant(const float *x, uint32_t k, float s,
                                    int32_t z, uint8_t *q, int8_t *qs) {
  const float inv = moe_m1_div(1.0f, s);
  uint32_t i = 0;
#ifdef SWIGLU_DET_HAS_NEON
  /* vcvtnq is round-to-nearest-even, which is rne() for |v| <= 2^22; the
     two saturating narrows are the DSP's vpack :sat pair. */
  const float32x4_t vinv = vdupq_n_f32(inv);
  const int32x4_t vz = vdupq_n_s32(z);
  for (; i + 8u <= k; i += 8u) {
    const int32x4_t a =
      vaddq_s32(vcvtnq_s32_f32(swiglu_det_vmul(vld1q_f32(x + i), vinv)), vz);
    const int32x4_t b = vaddq_s32(
      vcvtnq_s32_f32(swiglu_det_vmul(vld1q_f32(x + i + 4u), vinv)), vz);
    const uint8x8_t u = vqmovun_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b)));
    vst1_u8(q + i, u);
    vst1_s8(qs + i, vreinterpret_s8_u8(veor_u8(u, vdup_n_u8(0x80))));
  }
#endif
  for (; i < k; ++i) {
    int32_t v = moe_m1_rne(swiglu_det_mul(x[i], inv)) + z;
    if (v < 0)
      v = 0;
    if (v > 255)
      v = 255;
    q[i] = (uint8_t)v;
    qs[i] = (int8_t)(v - 128);
  }
}

/** @brief Steps 1 and 2 on one row. */
static inline void moe_m1_act(const float *x, uint32_t k, float *scale,
                              int32_t *zp, uint8_t *q, int8_t *qs) {
  moe_m1_act_params(x, k, scale, zp);
  moe_m1_act_quant(x, k, *scale, *zp, q, qs);
}

/** @brief One i4 weight of a WH matrix with n_col tile columns: row k,
 *         column col (htp_wh_layout.h's byte order). */
static inline int32_t moe_m1_wh(const uint8_t *wh, uint32_t n_col, uint32_t k,
                                uint32_t col) {
  const uint32_t r = k % 32u, c = col % 32u;
  const uint8_t b = wh[((size_t)(k / 32u) * n_col + col / 32u) * 512u +
                       (r / 8u) * 128u + c * 4u + r % 4u];
  const int32_t v = (r % 8u) < 4u ? (b & 15) : (b >> 4);
  return v > 7 ? v - 16 : v;
}

/**
 * @brief Step 3 for the tiles tiles[0 .. n) (n <= MOE_M1_BLOCK) of a WH
 *        matrix k_tiles deep: acc[t * 32 + c] = sum_k q[k] * w[k][col],
 *        col = tiles[t] * 32 + c. k-tile outer, so every k-tile row is
 *        read as runs of adjacent tiles.
 * @param qs q ^ 0x80, read by the NEON path
 * @param cs the matrix's column sums, read by the NEON path
 */
static inline void moe_m1_gemv_block(const uint8_t *q, const int8_t *qs,
                                     uint32_t k_tiles, const uint8_t *wh,
                                     uint32_t n_col, const uint32_t *tiles,
                                     uint32_t n, const int32_t *cs,
                                     int32_t *acc) {
#ifdef MOE_M1_DET_HAS_SDOT
  int32x4_t a[MOE_M1_BLOCK][8];
  for (uint32_t t = 0; t < n; ++t)
    for (uint32_t v = 0; v < 8u; ++v)
      a[t][v] = vdupq_n_s32(0);
  const int8x16_t hi_mask = vdupq_n_s8((int8_t)0xF0);
  for (uint32_t kt = 0; kt < k_tiles; ++kt) {
    const int8x16_t a0 = vld1q_s8(qs + (size_t)kt * 32u);
    const int8x16_t a1 = vld1q_s8(qs + (size_t)kt * 32u + 16u);
    const int8_t *row = (const int8_t *)wh + (size_t)kt * n_col * 512u;
    for (uint32_t t = 0; t < n; ++t) {
      const int8_t *tile = row + (size_t)tiles[t] * 512u;
      for (uint32_t v = 0; v < 8u; ++v) {
        int32x4_t s = a[t][v];
        const int8x16_t w0 = vld1q_s8(tile + v * 16u);
        const int8x16_t w1 = vld1q_s8(tile + 128u + v * 16u);
        const int8x16_t w2 = vld1q_s8(tile + 256u + v * 16u);
        const int8x16_t w3 = vld1q_s8(tile + 384u + v * 16u);
        s = vdotq_laneq_s32(s, vshlq_n_s8(w0, 4), a0, 0);
        s = vdotq_laneq_s32(s, vandq_s8(w0, hi_mask), a0, 1);
        s = vdotq_laneq_s32(s, vshlq_n_s8(w1, 4), a0, 2);
        s = vdotq_laneq_s32(s, vandq_s8(w1, hi_mask), a0, 3);
        s = vdotq_laneq_s32(s, vshlq_n_s8(w2, 4), a1, 0);
        s = vdotq_laneq_s32(s, vandq_s8(w2, hi_mask), a1, 1);
        s = vdotq_laneq_s32(s, vshlq_n_s8(w3, 4), a1, 2);
        s = vdotq_laneq_s32(s, vandq_s8(w3, hi_mask), a1, 3);
        a[t][v] = s;
      }
    }
  }
  for (uint32_t t = 0; t < n; ++t) {
    const int32_t *c = cs + (size_t)tiles[t] * 32u;
    for (uint32_t v = 0; v < 8u; ++v)
      vst1q_s32(acc + t * 32u + v * 4u,
                vaddq_s32(vshrq_n_s32(a[t][v], 4),
                          vshlq_n_s32(vld1q_s32(c + v * 4u), 7)));
  }
  (void)q;
#else
  (void)qs;
  (void)cs;
  for (uint32_t t = 0; t < n; ++t)
    for (uint32_t c = 0; c < 32u; ++c) {
      int32_t s = 0;
      for (uint32_t k = 0; k < k_tiles * 32u; ++k)
        s += (int32_t)q[k] * moe_m1_wh(wh, n_col, k, tiles[t] * 32u + c);
      acc[t * 32u + c] = s;
    }
#endif
}

/** @brief Step 4 on 32 columns: r[c] = ((c * s) * ws[c]) + b[c]. */
static inline void moe_m1_dq32(const int32_t *acc, float s, int32_t z,
                               const int32_t *cs, const float *ws,
                               const float *b, float *r) {
  uint32_t c = 0;
#ifdef SWIGLU_DET_HAS_NEON
  const float32x4_t vs = vdupq_n_f32(s);
  for (; c < 32u; c += 4u) {
    /* acc - z * cs in int32 is the DSP's f32 subtraction: every term is an
       integer below 2^24, so both are exact. */
    const float32x4_t cf = vcvtq_f32_s32(
      vsubq_s32(vld1q_s32(acc + c), vmulq_n_s32(vld1q_s32(cs + c), z)));
    vst1q_f32(r + c, swiglu_det_vadd(swiglu_det_vmul(swiglu_det_vmul(cf, vs),
                                                     vld1q_f32(ws + c)),
                                     vld1q_f32(b + c)));
  }
#endif
  for (; c < 32u; ++c) {
    const float cf = (float)(acc[c] - z * cs[c]);
    r[c] = swiglu_det_add(swiglu_det_mul(swiglu_det_mul(cf, s), ws[c]), b[c]);
  }
}

/**
 * @brief Steps 3-5 for the gate/up pairs [j0, j1) of one expert: gate[j*32
 *        .. j*32+32) for each pair j (gate tile j, up tile inter/32 + j).
 * @param q, qs, s, z the token row's step 1-2 output (K = k_tiles * 32)
 */
static inline void moe_m1_gate_range(const moe_m1_weights *w, uint32_t K,
                                     uint32_t inter, const uint8_t *q,
                                     const int8_t *qs, float s, int32_t z,
                                     uint32_t j0, uint32_t j1, float *gate) {
  const uint32_t k_tiles = K / 32u, inter_nt = inter / 32u;
  const uint32_t n_col = 2u * inter_nt;
  uint32_t tiles[MOE_M1_BLOCK];
  int32_t acc[MOE_M1_BLOCK * 32u];
  float g[32], u[32];
  for (uint32_t b0 = j0; b0 < j1; b0 += MOE_M1_BLOCK / 2u) {
    const uint32_t np =
      j1 - b0 < MOE_M1_BLOCK / 2u ? j1 - b0 : MOE_M1_BLOCK / 2u;
    for (uint32_t p = 0; p < np; ++p) {
      tiles[p] = b0 + p;
      tiles[np + p] = inter_nt + b0 + p;
    }
    moe_m1_gemv_block(q, qs, k_tiles, w->gu_wh, n_col, tiles, 2u * np, w->gu_cs,
                      acc);
    for (uint32_t p = 0; p < np; ++p) {
      const uint32_t cg = (b0 + p) * 32u, cu = inter + cg;
      moe_m1_dq32(acc + p * 32u, s, z, w->gu_cs + cg, w->gu_ws + cg,
                  w->gu_b + cg, g);
      moe_m1_dq32(acc + (np + p) * 32u, s, z, w->gu_cs + cu, w->gu_ws + cu,
                  w->gu_b + cu, u);
      swiglu_det(32u, gate + cg, g, u);
    }
  }
}

/** @brief Steps 3, 4 and 7 for the down tiles [nt0, nt1) of one expert:
 *         res[nt*32 .. nt*32+32). mid / mid_s / rs / rz: step 6's output. */
static inline void moe_m1_down_range(const moe_m1_weights *w, uint32_t inter,
                                     uint32_t N_out, const uint8_t *mid,
                                     const int8_t *mid_s, float rs, int32_t rz,
                                     uint32_t nt0, uint32_t nt1, float *res) {
  const uint32_t k_tiles = inter / 32u, n_col = N_out / 32u;
  uint32_t tiles[MOE_M1_BLOCK];
  int32_t acc[MOE_M1_BLOCK * 32u];
  for (uint32_t b0 = nt0; b0 < nt1; b0 += MOE_M1_BLOCK) {
    const uint32_t n = nt1 - b0 < MOE_M1_BLOCK ? nt1 - b0 : MOE_M1_BLOCK;
    for (uint32_t t = 0; t < n; ++t)
      tiles[t] = b0 + t;
    moe_m1_gemv_block(mid, mid_s, k_tiles, w->dn_wh, n_col, tiles, n, w->dn_cs,
                      acc);
    for (uint32_t t = 0; t < n; ++t) {
      const uint32_t c0 = (b0 + t) * 32u;
      moe_m1_dq32(acc + t * 32u, rs, rz, w->dn_cs + c0, w->dn_ws + c0,
                  w->dn_b + c0, res + c0);
    }
  }
}

/** @brief Step 8: dst = dst + (src * weight), n elements. */
static inline void moe_m1_scale_add(float *dst, const float *src, float weight,
                                    uint32_t n) {
  uint32_t i = 0;
#ifdef SWIGLU_DET_HAS_NEON
  const float32x4_t vw = vdupq_n_f32(weight);
  for (; i + 4u <= n; i += 4u)
    vst1q_f32(dst + i,
              swiglu_det_vadd(vld1q_f32(dst + i),
                              swiglu_det_vmul(vld1q_f32(src + i), vw)));
#endif
  for (; i < n; ++i)
    dst[i] = swiglu_det_add(dst[i], swiglu_det_mul(src[i], weight));
}

/**
 * @brief One whole expert, single-threaded: res[N_out] from the token row
 *        x[K]. The reference the host check and the device gtest run;
 *        the app runs the same range functions on its thread pool.
 * @param gate, mid, mid_s scratch of inter floats / bytes each
 * @param q, qs scratch of K bytes each
 */
static inline void moe_m1_expert(const moe_m1_weights *w, const float *x,
                                 uint32_t K, uint32_t inter, uint32_t N_out,
                                 uint8_t *q, int8_t *qs, float *gate,
                                 uint8_t *mid, int8_t *mid_s, float *res) {
  float s, rs;
  int32_t z, rz;
  moe_m1_act(x, K, &s, &z, q, qs);
  moe_m1_gate_range(w, K, inter, q, qs, s, z, 0u, inter / 32u, gate);
  moe_m1_act(gate, inter, &rs, &rz, mid, mid_s);
  moe_m1_down_range(w, inter, N_out, mid, mid_s, rs, rz, 0u, N_out / 32u, res);
}

#endif /* __NNTRAINER_MOE_M1_DET_H__ */
