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

void ref_rmsnorm(const __fp16 *x, const __fp16 *gamma, __fp16 *y, uint32_t m,
                 uint32_t n, uint32_t chunk, float eps) {
  for (uint32_t t = 0; t < m; ++t) {
    const __fp16 *xrow = x + (size_t)t * n;
    __fp16 *yrow = y + (size_t)t * n;
    for (uint32_t c0 = 0; c0 < n; c0 += chunk) {
      float sumsq = 0.f;
      for (uint32_t i = 0; i < chunk; ++i) {
        float v = (float)xrow[c0 + i];
        sumsq += v * v;
      }
      float r = 1.0f / sqrtf(sumsq / (float)chunk + eps);
      for (uint32_t i = 0; i < chunk; ++i)
        yrow[c0 + i] = (__fp16)((float)xrow[c0 + i] * r * (float)gamma[i]);
    }
  }
}

void ref_rope_table_fill(__fp16 *table, uint32_t max_seq, float theta) {
  for (uint32_t p = 0; p < max_seq; ++p) {
    __fp16 *row = table + (size_t)p * 128;
    for (uint32_t i = 0; i < 64; ++i) {
      float inv_freq = powf(theta, -2.0f * (float)i / 128.0f);
      row[i] = (__fp16)cosf((float)p * inv_freq);
      row[64 + i] = (__fp16)sinf((float)p * inv_freq);
    }
  }
}

void ref_rope(__fp16 *x, const __fp16 *table, uint32_t m, uint32_t heads,
              uint32_t pos) {
  for (uint32_t t = 0; t < m; ++t) {
    const __fp16 *row = table + (size_t)(pos + t) * 128;
    for (uint32_t h = 0; h < heads; ++h) {
      __fp16 *xh = x + ((size_t)t * heads + h) * 128;
      for (uint32_t i = 0; i < 64; ++i) {
        float x0 = (float)xh[i], x1 = (float)xh[64 + i];
        float cs = (float)row[i], sn = (float)row[64 + i];
        xh[i] = (__fp16)(x0 * cs - x1 * sn);
        xh[64 + i] = (__fp16)(x1 * cs + x0 * sn);
      }
    }
  }
}

void ref_add(const __fp16 *a, const __fp16 *b, __fp16 *y, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i)
    y[i] = (__fp16)((float)a[i] + (float)b[i]);
}

void ref_silu_mul(const __fp16 *g, const __fp16 *u, __fp16 *y, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    float gf = (float)g[i];
    float silu = gf / (1.0f + expf(-gf));
    y[i] = (__fp16)(silu * (float)u[i]);
  }
}
