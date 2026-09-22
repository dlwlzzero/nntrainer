// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_fwht_f32.c
 * @date   22 Sep 2026
 * @brief  In-place block-256 Walsh-Hadamard rotation of f32 rows (HVX)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * The DSP half of fwht_det.h: the SwiGLU output is rotated by H/16 in
 * blocks of 256 right before hexkl_mm_u8i4_moe.c requantizes it to uint8
 * for the down matmul, whose weight the converter folded by H^T/16
 * (QS4CX_WH_HAD). The two cancel exactly; what changes is the shape of
 * the u8 requantization error (issue #95).
 *
 * Bit identity with the scalar reference is the gate on this file
 * (HvxFwht.MatchesScalarBitExact), not SNR: the rotated values go through
 * a quantizer, and any one-ulp disagreement with the host reference
 * eventually lands an element on the other side of a u8 level.
 *
 * Lane arrangement. A block is eight vectors of 32 lanes. Stages s = 1,
 * 2, 4, 8, 16 pair lanes inside a vector: the partner of lane L is lane
 * L ^ s, fetched with a byte rotate in each direction, and a lane-index
 * mask picks a + c for lanes with bit s clear and a - c for the others.
 * Stages 32, 64, 128 pair whole vectors. Every output is exactly one
 * Q6_Vsf_vadd or Q6_Vsf_vsub of two stage inputs, in the reference's
 * stage order, so the bits are the reference's regardless of how the
 * lanes were moved to meet each other. The final 1/16 is one
 * Q6_Vsf_vmpy by a power of two: exact.
 */

#include "hvx_fwht_f32.h"

#include <stddef.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

#include "hvx_convert.h"

/** @brief f32 lanes per 128-byte vector; vectors per block. */
#define LANES 32u
#define VECS_PER_BLOCK (HVX_FWHT_BLOCK / LANES)

/** @brief One in-vector butterfly stage: lanes L and L ^ s. */
static inline HVX_Vector hvx_fwht_stage_lanes(HVX_Vector v, uint32_t s,
                                              HVX_Vector lane_idx) {
  const HVX_Vector vs = Q6_V_vsplat_R((int)s);
  /* Lanes whose index has bit s set hold the "c" of their pair. */
  const HVX_VectorPred is_c = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(lane_idx, vs), vs);
  /* vror by n bytes: out.b[i] = in.b[(i + n) mod 128], so +s lanes brings
     lane L+s down to L and 128 - 4s bytes brings lane L-s up to L. */
  const HVX_Vector up = Q6_V_vror_VR(v, (int)(4u * s));
  const HVX_Vector down = Q6_V_vror_VR(v, (int)(128u - 4u * s));
  const HVX_Vector sum = Q6_Vsf_vadd_VsfVsf(v, up);   /* a + c at "a" */
  const HVX_Vector dif = Q6_Vsf_vsub_VsfVsf(down, v); /* a - c at "c" */
  return Q6_V_vmux_QVV(is_c, dif, sum);
}

/** @brief The whole specification on one 256-float block, in registers. */
static inline void hvx_fwht_block(float *p, HVX_Vector lane_idx,
                                  HVX_Vector sixteenth) {
  HVX_UVector *vp = (HVX_UVector *)p;
  HVX_Vector v[VECS_PER_BLOCK];
  for (uint32_t j = 0; j < VECS_PER_BLOCK; ++j) {
    v[j] = vp[j];
  }
  for (uint32_t s = 1; s < LANES; s <<= 1) {
    for (uint32_t j = 0; j < VECS_PER_BLOCK; ++j) {
      v[j] = hvx_fwht_stage_lanes(v[j], s, lane_idx);
    }
  }
  for (uint32_t t = 1; t < VECS_PER_BLOCK; t <<= 1) {
    for (uint32_t j = 0; j < VECS_PER_BLOCK; ++j) {
      if (j & t) {
        continue;
      }
      const HVX_Vector a = v[j];
      const HVX_Vector c = v[j + t];
      v[j] = Q6_Vsf_vadd_VsfVsf(a, c);
      v[j + t] = Q6_Vsf_vsub_VsfVsf(a, c);
    }
  }
  for (uint32_t j = 0; j < VECS_PER_BLOCK; ++j) {
    vp[j] = Q6_Vsf_vmpy_VsfVsf(v[j], sixteenth);
  }
}

typedef struct {
  float *x;
  uint32_t rows;
  uint32_t k;
} hvx_fwht_ctx;

/** @brief hvx_worker_pool_func body: rows are independent, so the pool
 *         splits by row range, the same contiguous split the SwiGLU and
 *         quant workers use. */
static void hvx_fwht_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  const hvx_fwht_ctx *c = (const hvx_fwht_ctx *)vctx;
  const uint32_t lo = (uint32_t)((uint64_t)c->rows * i / n_threads);
  const uint32_t hi = (uint32_t)((uint64_t)c->rows * (i + 1) / n_threads);
  static const uint32_t lane_index[LANES] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  const HVX_Vector lane_idx = ((const HVX_UVector *)lane_index)[0];
  const HVX_Vector sixteenth = hvx_splat_sf(FWHT_DET_SCALE);
  for (uint32_t r = lo; r < hi; ++r) {
    float *row = c->x + (size_t)r * c->k;
    for (uint32_t b = 0; b + HVX_FWHT_BLOCK <= c->k; b += HVX_FWHT_BLOCK) {
      hvx_fwht_block(row + b, lane_idx, sixteenth);
    }
  }
}

void hvx_fwht_rows_f32(float *x, uint32_t rows, uint32_t k,
                       hvx_worker_pool *pool) {
  if (!x || rows == 0u || k < HVX_FWHT_BLOCK) {
    return;
  }
  hvx_fwht_ctx c = {x, rows, k};
  hvx_worker_pool_run(pool, hvx_fwht_worker, &c, rows);
}
