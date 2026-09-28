// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   geglu_tanh_host_check.c
 * @date   28 Sep 2026
 * @brief  Host check: the real HVX GeGLU and tanh kernels, compiled against
 *         hvx_emu/, are bit-identical to swiglu_det.h and within tolerance
 *         of a double reference
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Same two halves as m1_ops_host_check.c:
 *  1. BIT IDENTITY. hvx_swiglu_f32.c -- the skel's own source -- on the
 *     lane-by-lane emulation, memcmp'd against geglu_det_one / tanh_det_one
 *     at row lengths with and without a scalar tail. It rests on one Vsf op
 *     = one IEEE op, device-confirmed by HvxSwigluDet (rule 24) and
 *     re-checked for these two by HvxGegluTanhDet.*.
 *  2. TOLERANCE. The spec against double gelu_tanh / tanh (asserted) and,
 *     for comparison, the CPU's own float path __fallback_tanh_gelu's
 *     formula with tanhf (printed only).
 *
 * No Gemma4 MoE model runs in this tree, so the shapes are generic: 1792
 * and 2048 (vector-only), 2051 (a 3-element tail), and the 262144 vocab
 * row for the softcap.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hvx_swiglu_f32.h"
#include "swiglu_det.h"

/* The pool's NULL path: func(1, 0, ctx) once on the caller. */
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units) {
  (void)pool;
  if (n_units) {
    func(1u, 0u, ctx);
  }
}

static int g_fail = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL: " __VA_ARGS__);                                            \
      printf("\n");                                                            \
      g_fail = 1;                                                              \
    }                                                                          \
  } while (0)

static uint32_t g_seed = 0x6e6d0001u;
static float frand(float lo, float hi) {
  g_seed = g_seed * 1664525u + 1013904223u;
  return lo + (hi - lo) * ((float)(g_seed >> 8) / 16777216.0f);
}

/** @brief HvxSwigluDet's spread (both clamps, then a dense band) plus the
 *         edges this spec adds: near zero, and |g| up to the 1e12 domain. */
static float spread(uint32_t i, float band) {
  if (i < 64u) {
    return -200.0f + 3.0f * (float)i;
  }
  if (i < 128u) {
    return 100.0f - 3.0f * (float)(i - 64u);
  }
  if (i < 160u) {
    return ldexpf((i & 1u) ? -1.0f : 1.0f, -(int)(i - 128u)); /* 2^-k */
  }
  if (i < 176u) {
    return ((i & 1u) ? -1.0f : 1.0f) *
           powf(10.0f, (float)(i - 164u)); /* ~1e12 */
  }
  return frand(-band, band);
}

static double gelu_ref(double g) {
  return 0.5 * g *
         (1.0 + tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)));
}

/** @brief |a - b| over max(|b|, floor): relative, with an absolute floor. */
static double err(double a, double b, double floor) {
  const double d = fabs(b) > floor ? fabs(b) : floor;
  return fabs(a - b) / d;
}

static void check_geglu(uint32_t m, uint32_t n) {
  const size_t N = (size_t)m * n;
  float *g = malloc(N * sizeof(float)), *u = malloc(N * sizeof(float));
  float *y = malloc(N * sizeof(float));
  for (size_t i = 0; i < N; ++i) {
    g[i] = spread((uint32_t)(i % 8192u), 8.0f);
    u[i] = frand(-8.0f, 8.0f);
  }
  memcpy(y, g, N * sizeof(float));
  hvx_geglu_inplace_f32(y, u, m, n, NULL);

  uint32_t bad = 0;
  double max_det = 0.0, max_cpu = 0.0;
  for (size_t i = 0; i < N; ++i) {
    const float ref = geglu_det_one(g[i], u[i]);
    bad += memcmp(&y[i], &ref, sizeof(float)) ? 1u : 0u;
    if (fabsf(g[i]) > 1e3f) {
      continue; /* double's g^3 is fine there, the float CPU formula's is not */
    }
    /* gelu alone (u = 1), relative above |gelu| = 1e-3 and absolute
       below: where a*K passes the exp clamp (g < ~-9.5) the spec returns a
       value near 1e-36 instead of the true 1e-40, which is right in every
       sense that reaches a quantizer and wrong by 1e14 as a ratio. */
    const double r = gelu_ref(g[i]);
    const double e_det = err(geglu_det_one(g[i], 1.0f), r, 1e-3);
    const float x = g[i];
    const float cpu =
      0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    const double e_cpu = err(cpu, r, 1e-3);
    if (e_det > max_det) {
      max_det = e_det;
    }
    if (e_cpu > max_cpu && fabsf(g[i]) < 8.0f) {
      max_cpu = e_cpu;
    }
  }
  printf("GEGLU m=%u n=%u bad=%u max_rel(det vs double)=%.3g "
         "max_rel(cpu tanhf vs double, |g|<8)=%.3g (both: abs below 1e-3)\n",
         m, n, bad, max_det, max_cpu);
  CHECK(bad == 0u, "GeGLU m=%u n=%u: HVX differs from geglu_det_one", m, n);
  /* Measured 1.18e-6 at g = -3.35 on a 2M-point sweep of [-12, 12]: the
     exp argument's own rounding, times |a*K| ~ 8. 2e-6 is the bound. */
  CHECK(max_det <= 2e-6, "GeGLU: %.3g relative from the double reference",
        max_det);
  free(g);
  free(u);
  free(y);
}

static void check_tanh(uint32_t n, float in, float out, const char *name) {
  float *x = malloc(n * sizeof(float)), *y = malloc(n * sizeof(float));
  const float band = (out == 1.0f) ? 8.0f : 100.0f;
  for (uint32_t i = 0; i < n; ++i) {
    x[i] = spread(i % 8192u, band);
  }
  memcpy(y, x, n * sizeof(float));
  hvx_tanh_inplace_f32(y, n, in, out);

  uint32_t bad = 0;
  double max_abs = 0.0, max_abs_cpu = 0.0, max_rel_big = 0.0;
  for (uint32_t i = 0; i < n; ++i) {
    const float ref = tanh_det_one(x[i], in, out);
    bad += memcmp(&y[i], &ref, sizeof(float)) ? 1u : 0u;
    const double r = (double)out * tanh((double)x[i] * (double)in);
    const float cpu = tanhf(x[i] * in) * out;
    const double a = fabs((double)ref - r) / fabs((double)out);
    const double ac = fabs((double)cpu - r) / fabs((double)out);
    if (a > max_abs) {
      max_abs = a;
    }
    if (ac > max_abs_cpu) {
      max_abs_cpu = ac;
    }
    if (fabs(r) >= 0.1 * fabs((double)out) && err(ref, r, 0.0) > max_rel_big) {
      max_rel_big = err(ref, r, 0.0);
    }
  }
  printf("TANH %s n=%u bad=%u max_abs(det, /out)=%.3g "
         "max_abs(cpu tanhf, /out)=%.3g max_rel(|tanh|>=0.1)=%.3g\n",
         name, n, bad, max_abs, max_abs_cpu, max_rel_big);
  CHECK(bad == 0u, "tanh %s n=%u: HVX differs from tanh_det_one", name, n);
  /* Measured 3.46e-7 at x = 2.5 on a 2M-point sweep of [-10, 10], ~6 ulp
     of 0.99: 1 + e rounds away e's low bits, and 2s - 1 doubles what is
     left. tanhf is 1.0e-7. Enough for a softcap (30 * 3.5e-7 = 1e-5 on a
     logit); tighten the spec, not the bound, if a use needs better. */
  CHECK(max_abs <= 5e-7, "tanh %s: %.3g absolute from double", name, max_abs);
  CHECK(max_rel_big <= 2e-6, "tanh %s: %.3g relative where |tanh| >= 0.1", name,
        max_rel_big);
  free(x);
  free(y);
}

int main(void) {
  check_geglu(3u, 1792u);
  check_geglu(2u, 2048u);
  check_geglu(3u, 2051u);
  check_tanh(8192u, 1.0f, 1.0f, "plain");
  check_tanh(8195u, 1.0f, 1.0f, "plain+tail");
  check_tanh(262144u, 1.0f / 30.0f, 30.0f, "softcap30");
  if (g_fail) {
    printf("GEGLU TANH HOST CHECK FAILED\n");
    return 1;
  }
  printf("GEGLU TANH HOST CHECK OK\n");
  return 0;
}
