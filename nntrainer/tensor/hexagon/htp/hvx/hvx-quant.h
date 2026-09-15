// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-quant.h
 * @date	18 August 2026
 * @brief	Per-token dynamic int8 quantization (scalar; the vrmpy tile
 *		kernel lives in ops/hvx-matmul.c)
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef HVX_QUANT_H
#define HVX_QUANT_H

#include <math.h>
#include <stdint.h>

/* Scalar per-token quantization; rows are spread over the worker pool by
 * the caller. Returns scale (absmax / 127). */
static inline float htp_quant_row_fp16(const __fp16 *x, int8_t *q, uint32_t k) {
  float amax = 0.f;
  for (uint32_t i = 0; i < k; ++i) {
    float v = fabsf((float)x[i]);
    if (v > amax)
      amax = v;
  }
  float inv = amax > 0.f ? 127.f / amax : 0.f;
  for (uint32_t i = 0; i < k; ++i)
    q[i] = (int8_t)lrintf((float)x[i] * inv);
  return amax / 127.f;
}

#endif /* HVX_QUANT_H */
