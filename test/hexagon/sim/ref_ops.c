// SPDX-License-Identifier: Apache-2.0
/**
 * @file	ref_ops.c
 * @date	18 August 2026
 * @brief	Scalar C reference implementations for hexagon-sim primitive tests
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "ref_ops.h"

#include <math.h>
#include <stdlib.h>

float ref_quant_row(const __fp16 *x, int8_t *q, uint32_t k) {
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

int32_t ref_dot_i8(const int8_t *w, const int8_t *x, uint32_t k) {
  int32_t acc = 0;
  for (uint32_t i = 0; i < k; ++i)
    acc += (int32_t)w[i] * (int32_t)x[i];
  return acc;
}

void ref_matmul_w8a8(const __fp16 *x, const int8_t *w, const float *sw,
                     __fp16 *y, uint32_t m, uint32_t k, uint32_t n) {
  int8_t *xq = malloc((size_t)k);
  for (uint32_t t = 0; t < m; ++t) {
    float sx = ref_quant_row(x + (size_t)t * k, xq, k);
    for (uint32_t j = 0; j < n; ++j) {
      int32_t dot = ref_dot_i8(w + (size_t)j * k, xq, k);
      y[(size_t)t * n + j] = (__fp16)((float)dot * sw[j] * sx);
    }
  }
  free(xq);
}
