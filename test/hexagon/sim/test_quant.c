// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_quant.c
 * @date	18 August 2026
 * @brief	Hexagon-sim test for per-token dynamic quantization
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdio.h>
#include <string.h>

#include "hvx-quant.h"
#include "ref_ops.h"
#include "sim_test_util.h"

#define KQ 3072

static __fp16 x_row[KQ] __attribute__((aligned(128)));
static int8_t q_got[KQ] __attribute__((aligned(128)));
static int8_t q_ref[KQ] __attribute__((aligned(128)));

/* Quantize x_row with the kernel and with the scalar reference. scale must
 * always be equal; q must be byte-identical unless pm1_out is given, in which
 * case +/-1 differences are counted there instead (the caller bounds the
 * rate). Prints the first mismatch on failure. */
static int check_row(const char *tag, uint32_t k, uint32_t *pm1_out) {
  float scale_got = htp_quant_row_fp16(x_row, q_got, k);
  float scale_ref = ref_quant_row(x_row, q_ref, k);

  if (scale_got != scale_ref) {
    printf("SIM_TEST quant FAIL %s scale got=%f ref=%f\n", tag,
           (double)scale_got, (double)scale_ref);
    return 1;
  }
  if (memcmp(q_got, q_ref, k) != 0) {
    uint32_t n = 0, first = 0;
    int pm1 = 1;
    for (uint32_t i = 0; i < k; ++i)
      if (q_got[i] != q_ref[i]) {
        if (n == 0)
          first = i;
        ++n;
        if (q_got[i] - q_ref[i] != 1 && q_got[i] - q_ref[i] != -1)
          pm1 = 0;
      }
    if (pm1_out && pm1) {
      *pm1_out += n;
      return 0;
    }
    printf("SIM_TEST quant FAIL %s n=%u/%u all_pm1=%d i=%u got=%d ref=%d "
           "x=%.9g x/scale=%.9g\n",
           tag, (unsigned)n, (unsigned)k, pm1, (unsigned)first, q_got[first],
           q_ref[first], (double)x_row[first],
           (double)x_row[first] / (double)scale_ref);
    return 1;
  }
  return 0;
}

static int test_quant_row(void) {
  char tag[32];
  int fails = 0;

  /* (a) 16 random rows, k=1024, amplitude cycling over the exponent range. */
  static const float amp[4] = {8.f, 0.01f, 1000.f, 1e-4f};
  for (int r = 0; r < 16; ++r) {
    for (uint32_t i = 0; i < 1024; ++i)
      x_row[i] = (__fp16)(frand() * amp[r & 3]);
    snprintf(tag, sizeof(tag), "rand%d", r);
    fails += check_row(tag, 1024, NULL);
  }

  /* (b) all-zero row: inv = 0 branch. */
  memset(x_row, 0, sizeof(x_row));
  fails += check_row("zero", 1024, NULL);

  /* (c) tie row: x[0] = 127 makes inv exactly 1.0, the rest are +/-(n + 0.5)
   * so every element lands on a rounding tie (ties-to-even like lrintf). */
  x_row[0] = (__fp16)127.f;
  for (uint32_t i = 1; i < 1024; ++i)
    x_row[i] = (__fp16)((i & 1u ? -1.f : 1.f) * ((float)(i % 127u) + 0.5f));
  fails += check_row("tie", 1024, NULL);

  /* (d) widest and narrowest row widths. */
  for (uint32_t i = 0; i < KQ; ++i)
    x_row[i] = (__fp16)(frand() * 8.f);
  fails += check_row("k3072", KQ, NULL);
  fails += check_row("k128", 128, NULL);

  /* (e) negative-only row, and a row whose absmax is the last element. */
  for (uint32_t i = 0; i < 1024; ++i)
    x_row[i] = (__fp16)(-1.f - fabsf(frand()) * 7.f);
  fails += check_row("neg", 1024, NULL);
  for (uint32_t i = 0; i < 1024; ++i)
    x_row[i] = (__fp16)(frand() * 0.5f);
  x_row[1023] = (__fp16)9.5f;
  fails += check_row("last", 1024, NULL);

  /* (f) generic (non-dyadic) rows: the kernel rounds at a 2^-14 resolution, so
   * a true product just above a .5 tie may round to even instead of up. Only
   * the +/-1 rate is bounded here. */
  static const float gamp[4] = {3.7f, 0.013f, 731.f, 2.9e-4f};
  uint32_t pm1 = 0;
  for (int r = 0; r < 64; ++r) {
    for (uint32_t i = 0; i < 1024; ++i)
      x_row[i] = (__fp16)(frand() * gamp[r & 3]);
    x_row[0] = (__fp16)(gamp[r & 3] * 0.973f);
    snprintf(tag, sizeof(tag), "generic%d", r);
    fails += check_row(tag, 1024, &pm1);
  }
  printf("SIM_TEST quant_generic STAT pm1=%u/%u\n", (unsigned)pm1,
         (unsigned)(64u * 1024u));
  if (pm1 > 13u) {
    printf("SIM_TEST quant FAIL generic rate %u > 13\n", (unsigned)pm1);
    ++fails;
  }

  return fails;
}

int test_quant(void) {
  if (test_quant_row())
    return 1;

  printf("SIM_TEST quant PASS\n");
  return 0;
}
