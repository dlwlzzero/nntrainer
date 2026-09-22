// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   fwht_det.h
 * @date   22 Sep 2026
 * @brief  The deterministic block-256 Walsh-Hadamard rotation, scalar form
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * One specification, two implementations that must agree bit for bit:
 * this scalar loop (the quantizer's weight fold and every host reference)
 * and hvx/hvx_fwht_f32.c (the DSP, on the SwiGLU output right before its
 * uint8 requantization). The rotation h -> h*H/16 on the down_proj input
 * and W -> H^T*W/16 folded into the down weight cancel exactly, because
 * H*H^T = 256*I; what changes is only how the u8 requantization of the
 * intermediate spreads its error (issue #95).
 *
 * Specification, per row and per 256-float block v[0..255] (k is a
 * multiple of 256; 1792 = 7 blocks, no inter-block mixing):
 *
 *   for s in 1, 2, 4, 8, 16, 32, 64, 128:           # ascending, fixed
 *     for i in 0..255 with (i & s) == 0:
 *       a = v[i]; c = v[i + s]
 *       v[i]     = a + c                             # one IEEE f32 add, RNE
 *       v[i + s] = a - c                             # one IEEE f32 sub, RNE
 *   for i in 0..255: v[i] = v[i] * 0.0625f           # exact (power of two)
 *
 * Every stage output is exactly one f32 add or sub of two stage inputs,
 * so the HVX side is free to arrange lanes however it likes (rotate and
 * mux inside a vector for s < 32, whole vectors for s >= 32) and still
 * produce these bits. No FMA can form (there is no multiply until the
 * end), no reassociation is allowed, and qf32 is forbidden on the DSP
 * side: v75 and v79 disagree on qf32 -> sf.
 *
 * Denormals: HVX sf arithmetic flushes them to zero, so this reference
 * flushes every loaded input and every result too (fwht_det_ftz). The
 * sign of a flushed zero is the one thing the host cannot decide for the
 * DSP; this side keeps the sign, and HvxFwht.MatchesScalarBitExact feeds
 * +-subnormals to settle it on silicon.
 *
 * The transform is its own inverse under this scaling, which is why the
 * same function serves both the activation and the weight fold.
 */

#ifndef __NNTRAINER_FWHT_DET_H__
#define __NNTRAINER_FWHT_DET_H__

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Rotation block, in floats. Fixed at 256 (1792 = 7 x 256, the
 *        expert intermediate of LFM2-8B-A1B) and shared with the HVX
 *        kernel and the quantizer's converter refusal.
 *
 * ponytail: one constant, not a parameter. Both implementations, the
 * model file's dtype tag and the converter's K % 256 refusal assume it;
 * the upgrade path is to lift it to a parameter carried by the weight
 * header (and a new dtype tag), not to edit this number.
 */
#define FWHT_DET_BLOCK 256u

/** @brief 1/16, the block-256 normalisation applied on BOTH sides
 *         (activation here, weight in the converter), so the pair
 *         composes to the identity: (h*H/16)*(H^T*W/16) = h*W. */
#define FWHT_DET_SCALE 0.0625f

/** @brief Flush a subnormal to a zero of the same sign, as HVX sf does. */
static inline float fwht_det_ftz(float x) {
  const float a = fabsf(x);
  if (a != 0.0f && a < FLT_MIN) {
    return copysignf(0.0f, x);
  }
  return x;
}

/**
 * @brief One f32 operation, forced to round on its own.
 *
 * Applications/CausalLM builds with -ffast-math, under which a compiler may
 * reassociate the eight unrolled stages; storing through a volatile is what
 * no such rewrite can cross (the same device swiglu_det.h uses).
 */
static inline float fwht_det_add(float a, float b) {
  volatile float r = a + b;
  return fwht_det_ftz(r);
}
static inline float fwht_det_sub(float a, float b) {
  volatile float r = a - b;
  return fwht_det_ftz(r);
}

/** @brief The specification above on one 256-float block, in place. */
static inline void fwht_det_block(float *v) {
  for (uint32_t i = 0; i < FWHT_DET_BLOCK; ++i) {
    v[i] = fwht_det_ftz(v[i]);
  }
  for (uint32_t s = 1; s < FWHT_DET_BLOCK; s <<= 1) {
    for (uint32_t i = 0; i < FWHT_DET_BLOCK; ++i) {
      if (i & s) {
        continue;
      }
      const float a = v[i];
      const float c = v[i + s];
      v[i] = fwht_det_add(a, c);
      v[i + s] = fwht_det_sub(a, c);
    }
  }
  for (uint32_t i = 0; i < FWHT_DET_BLOCK; ++i) {
    volatile float r = v[i] * FWHT_DET_SCALE;
    v[i] = fwht_det_ftz(r);
  }
}

/**
 * @brief x[r][0..k) -> its block-256 Hadamard rotation, every row in place.
 *
 * @param x     rows x k floats, row-major, rewritten
 * @param rows  row count
 * @param k     columns per row; must be a multiple of FWHT_DET_BLOCK (the
 *              caller checks and refuses -- the converter and the DSP
 *              kernel both do -- because a partial block here would be a
 *              silently different rotation on each side)
 */
static inline void fwht_rows_f32_ref(float *x, uint32_t rows, uint32_t k) {
  for (uint32_t r = 0; r < rows; ++r) {
    float *row = x + (size_t)r * k;
    for (uint32_t b = 0; b + FWHT_DET_BLOCK <= k; b += FWHT_DET_BLOCK) {
      fwht_det_block(row + b);
    }
  }
}

#endif /* __NNTRAINER_FWHT_DET_H__ */
