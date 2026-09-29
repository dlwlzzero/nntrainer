// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   moe_m1_split_host_check.c
 * @date   29 Sep 2026
 * @brief  #157: the M=1 MoE call split between the DSP and the CPU is
 *         bit-identical to the DSP alone, for every split point k = 0..4
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * The DSP side is the REAL skel code: hexkl_mm_u8i4_moe.c's use_m1 path
 * with the real hvx_quant_u8.c, hvx_dequant_i32.c (+ hvx_swiglu_det.h),
 * hvx_gemm_u8i4_wh.c and hvx_scale_add_f32.c, compiled against hvx_emu/
 * (one IEEE f32 op per lane; the integer ops it adds for these sources
 * were checked against the SDK's libnative when they were written). Only
 * the DMA ring (copies at once), the worker pool (lanes one after
 * another) and the HMX entry points (never reached at M = 1) are stubs.
 *
 * The CPU side is nntrainer/tensor/moe_m1_det.h, the header the app's
 * split runs. For each token row and each k: the DSP call with the first k
 * active experts (ascending id) gives S_k, moe_m1_expert() computes the
 * rest and moe_m1_scale_add() continues the sum in id order; the result
 * must memcmp equal to the DSP call with all four. k = 0 is the CPU alone
 * from +0 (the app does not call the DSP then). The LFM2.5 shape (K 2048,
 * inter 1792, N 2048), the default call flags (0x303e1: GEMV, one-row
 * loop, VTCM feed), activation amplitudes 0.3 / 3 / 30 with outliers, an
 * all-positive row (zero point 0), an all-zero row (scale 1) and a row
 * large enough to push the gate past the exp clamp.
 *
 * What this cannot see: the device's rounding (the premise of hvx_emu is
 * the ISS result of plan 157 section 0.3; MoeM1CpuSplitMatchesDsp is the
 * silicon check), the NEON path (x86 runs the header's scalar path; the
 * device gtest runs NEON) and subnormals.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>

#include "hexkl_acc_tile.h"
#include "hexkl_dma_ring.h"
#include "hexkl_mm_u8i4_moe.h"
#include "hexkl_probe.h"
#include "hvx_dequant_i32.h"
#include "hvx_gather_ah_u8.h"
#include "hvx_quant_u8.h"
#include "hvx_scale_add_f32.h"
#include "hvx_worker_pool.h"
#include "moe_m1_det.h"

#define SK 2048u
#define INTER 1792u
#define NOUT 2048u
#define NE 8u
#define NACT 4u
#define FLAGS 0x303e1u

/* ---- stubs: what the M=1 path links but a host has no unit for ---- */
static hexkl_acc_layout g_layout = {1, 1, 0, 32};
const hexkl_acc_layout *hexkl_acc_layout_get(uint8_t *b, uint32_t off) {
  (void)b;
  (void)off;
  return &g_layout;
}
static void never(const char *what) {
  printf("MOE SPLIT CHECK: %s reached at M=1\n", what);
  exit(1);
}
int hexkl_micro_hmx_acc_clear_int32(void) {
  never("HMX");
  return 0;
}
int hexkl_micro_hmx_mm_u8i4(uint8_t *b, uint32_t a, uint32_t w) {
  (void)b;
  (void)a;
  (void)w;
  never("HMX");
  return 0;
}
int hexkl_micro_hmx_acc_read_int32(uint8_t *b, uint32_t c, uint32_t o) {
  (void)b;
  (void)c;
  (void)o;
  never("HMX");
  return 0;
}
static uint32_t g_idx;
void hexkl_dma_ring_reset(void) {}
uint32_t hexkl_dma_ring_next_idx(void) { return g_idx++; }
void hexkl_dma_ring_wait(uint32_t idx) { (void)idx; }
void hexkl_dma_ring_drain(void) {}
int hexkl_dma_ring_is_done(uint32_t idx) {
  (void)idx;
  return 1;
}
void hexkl_dma_ring_push2d(void *dst, const void *src, uint32_t ds, uint32_t ss,
                           uint32_t rs, uint32_t nrows, int sv, int dv) {
  (void)sv;
  (void)dv;
  for (uint32_t r = 0; r < nrows; ++r)
    memcpy((uint8_t *)dst + (size_t)r * ds,
           (const uint8_t *)src + (size_t)r * ss, rs);
}
/* The device's six lanes, one after another: each lane's slice is the one
   the device computes, and every column is independent of the others. */
void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units) {
  (void)pool;
  const uint32_t n = n_units < 6u ? n_units : 6u;
  if (n <= 1u) {
    func(1u, 0, ctx);
    return;
  }
  for (uint32_t i = 0; i < n; ++i)
    func(n, i, ctx);
}
void hvx_worker_pool_submit(hvx_worker_pool *pool, hvx_worker_pool_func func,
                            void *ctx, uint32_t n_units) {
  (void)pool;
  if (n_units != 0u)
    func(1u, 0, ctx);
}
void hvx_worker_pool_wait(hvx_worker_pool *pool) { (void)pool; }
void hvx_worker_pool_submit_bg(hvx_worker_pool *pool, hvx_bg_job *job) {
  (void)pool;
  for (uint32_t u = 0; u < job->n_units; ++u) {
    job->func(job->n_units, u, job->ctx);
    job->done[u] = 1;
  }
}
void hvx_worker_pool_wait_bg(hvx_worker_pool *pool, hvx_bg_job *job,
                             uint32_t n) {
  (void)pool;
  (void)job;
  (void)n;
}
void hvx_copy_ah_block(uint8_t *dst, const uint8_t *src, uint32_t k,
                       hvx_worker_pool *pool) {
  (void)pool;
  memcpy(dst, src, (size_t)(k / 32u) * 2048u);
}
uint64_t hexkl_probe_us[HEXKL_PROBE_N];
int hexkl_probe_on = 1;

/* ---- fixture ---- */
static uint32_t lcg_s = 12345u;
static uint32_t lcg(void) {
  lcg_s = lcg_s * 1664525u + 1013904223u;
  return lcg_s;
}
static float urand(void) { return (float)(lcg() >> 8) * (1.0f / 16777216.0f); }
static float grand(void) {
  return (urand() + urand() + urand() + urand() - 2.0f) * 1.7f;
}

static hexkl_weight_u8i4_table g_tbl;
static moe_m1_weights g_w[NE];

/* One WH weight with true column sums (the dequant and the NEON GEMV's
   sdot correction both read them), scales ~2-6e-3 and a bias that is 0 in
   production (@a bias_on 0) or small and random. */
static void make_weight(uint32_t slot, uint32_t k, uint32_t n, int bias_on,
                        const uint8_t **wh, const float **ws,
                        const int32_t **cs, const float **b) {
  const size_t bytes = (size_t)(k / 32u) * (n / 32u) * 512u;
  uint8_t *w = (uint8_t *)malloc(bytes);
  float *s = (float *)malloc(sizeof(float) * n);
  int32_t *c = (int32_t *)calloc(n, sizeof(int32_t));
  float *bb = (float *)malloc(sizeof(float) * n);
  for (size_t i = 0; i < bytes; ++i)
    w[i] = (uint8_t)(lcg() >> 24);
  for (uint32_t col = 0; col < n; ++col) {
    for (uint32_t r = 0; r < k; ++r)
      c[col] += moe_m1_wh(w, n / 32u, r, col);
    s[col] = (0.5f + urand()) * 0.004f;
    bb[col] = bias_on ? (urand() - 0.5f) * 0.01f : 0.0f;
  }
  hexkl_weight_u8i4 *sl = &g_tbl.slots[slot];
  sl->in_use = 1;
  sl->K = k;
  sl->N = n;
  sl->wh_bytes = w;
  sl->w_scale = s;
  sl->colsum_w = c;
  sl->bias = bb;
  *wh = w;
  *ws = s;
  *cs = c;
  *b = bb;
}

static uint8_t g_vtcm[8u << 20];
static hexkl_moe_scratch g_scratch;
static uint32_t g_hg[NE], g_hd[NE];

/* The DSP call with the first @a keep of the active experts (ascending
   id), the prefix routing the app builds. */
static int dsp_call(const float *x, const uint32_t *act, const float *rw,
                    uint32_t keep, float *out) {
  uint32_t rc[NE] = {0}, ri[NACT] = {0};
  for (uint32_t i = 0; i < keep; ++i)
    rc[act[i]] = 1u;
  return hexkl_mm_u8i4_moe_layer_run(
    &g_tbl, g_vtcm, sizeof g_vtcm, sizeof g_vtcm, 1u, SK, INTER, NOUT, NE, g_hg,
    g_hd, ri, rc, rw, x, out, NULL, &g_scratch, FLAGS);
}

/* ---- the stages alone, many values each ----------------------------------
   The whole-call comparison below rounds ~10^5 values through each
   quantizer, and a formula that differs from the DSP's only near a
   rounding tie (x / s against x * (1 / s), say) flips none of them there.
   These sweeps put the stages' own inputs where the formulas can differ:
   rows whose values sit half a level from a quantization boundary, random
   int32 accumulators through the fused dequant + SwiGLU and the down
   dequant, and random scale-adds. Returns the mismatch count. */
static uint32_t stage_sweep(void) {
  enum { ROWS = 400, QK = 2048 };
  static float x[4 * QK], sc[64], s_gate[32], s_res[32];
  static int32_t zp[64];
  static uint8_t ah[(QK / 32) * 2048], q[QK];
  static int8_t qs[QK];
  static const uint32_t map[4] = {0, 0, 0, 0};
  uint32_t bad_q = 0, bad_dq = 0, bad_sa = 0;
  for (uint32_t r = 0; r < ROWS; ++r) {
    const float amp = (r % 3u == 0u) ? 0.3f : (r % 3u == 1u) ? 3.0f : 30.0f;
    /* odd rows: range [-amp, amp] pinned by x[0], x[1], so the scale is
       s = (2 amp) / 255, and every other value (n + 1/2) * s: x / s lands
       within an ulp of a tie and the rounding formula decides the byte.
       Rows 2 and 4: a zero point of exactly 0.5 and 2.5 levels. */
    const float st = (2.0f * amp) / 255.0f;
    for (uint32_t k = 0; k < QK; ++k) {
      x[k] = (r & 1u) ? ((float)((int32_t)(lcg() % 254u) - 127) + 0.5f) * st
                      : grand() * amp;
    }
    if (r & 1u) {
      x[0] = -amp;
      x[1] = amp;
    } else if (r == 2u || r == 4u) {
      x[0] = r == 2u ? -1.0f : -5.0f; /* s = 510 / 255 = 2 */
      x[1] = r == 2u ? 509.0f : 505.0f;
      for (uint32_t k = 2; k < QK; ++k)
        x[k] = urand() * 500.0f;
    }
    hvx_quant_rows_u8_params(x, 1u, 64u, QK, sc, zp, NULL);
    hvx_quant_pack_u8_ah_rows(x, map, 0u, 1u, QK, sc, zp, ah);
    float s;
    int32_t z;
    moe_m1_act(x, QK, &s, &z, q, qs);
    bad_q += memcmp(&s, &sc[0], sizeof s) != 0 || z != zp[0];
    for (uint32_t k = 0; k < QK; ++k)
      bad_q += q[k] != ah[(k / 32u) * 2048u + k % 32u];
  }
  /* The dequant epilogues on random accumulators (|acc| < 2^23, as the
     GEMV's are): one gate/up tile pair through the fused SwiGLU worker,
     one down tile through hvx_dequant_acc_tile_to_f32. */
  static int32_t tiles[2][32], cs[64];
  static float ws[64], bias[64], g[32], u[32], dst[64], gspec[32];
  for (uint32_t it = 0; it < 20000u; ++it) {
    const float as = (0.5f + urand()) * ((it % 3u == 0u) ? 2e-3f : 3e-2f);
    const int32_t az = (int32_t)(lcg() % 256u);
    for (uint32_t c = 0; c < 64u; ++c) {
      cs[c] = (int32_t)(lcg() % 32768u) - 16384;
      ws[c] = (0.5f + urand()) * 0.004f;
      bias[c] = (it & 1u) ? (urand() - 0.5f) * 0.01f : 0.0f;
    }
    for (uint32_t c = 0; c < 32u; ++c) {
      tiles[0][c] = (int32_t)(lcg() % (1u << 23)) - (1 << 22);
      tiles[1][c] = (int32_t)(lcg() % (1u << 23)) - (1 << 22);
    }
    hvx_dequant_swiglu_acc_tiles_to_f32((const uint8_t *)tiles, 128u, 1u, 0u,
                                        32u, 1u, &as, &az, cs, ws, bias, 32u,
                                        dst, 32u, NULL);
    moe_m1_dq32(tiles[0], as, az, cs, ws, bias, g);
    moe_m1_dq32(tiles[1], as, az, cs + 32, ws + 32, bias + 32, u);
    swiglu_det(32u, gspec, g, u);
    bad_dq += memcmp(dst, gspec, sizeof gspec) != 0;
    hvx_dequant_acc_tile_to_f32(tiles[0], 32u, 1u, &as, &az, cs, ws, bias,
                                s_res, 32u, 0);
    moe_m1_dq32(tiles[0], as, az, cs, ws, bias, s_gate);
    bad_dq += memcmp(s_res, s_gate, sizeof s_res) != 0;
  }
  for (uint32_t it = 0; it < 20000u; ++it) {
    static float a[32], b[32], a2[32];
    const float w = 0.05f + urand();
    for (uint32_t c = 0; c < 32u; ++c) {
      a[c] = a2[c] = grand() * (float)(1u << (lcg() % 12u));
      b[c] = grand() * (float)(1u << (lcg() % 12u));
    }
    hvx_scale_add_rows_f32(a, b, w, 32u);
    moe_m1_scale_add(a2, b, w, 32u);
    bad_sa += memcmp(a, a2, sizeof a) != 0;
  }
  printf("moe split stages: quant rows=%u bad=%u, dequant tiles=40000 bad=%u, "
         "scale_add rows=20000 bad=%u\n",
         (unsigned)ROWS, bad_q, bad_dq, bad_sa);
  return bad_q + bad_dq + bad_sa;
}

int main(void) {
  const uint32_t act[NACT] = {1u, 3u, 4u, 6u};
  for (uint32_t e = 0; e < NE; ++e) {
    g_hg[e] = 2u * e;
    g_hd[e] = 2u * e + 1u;
  }
  for (uint32_t i = 0; i < NACT; ++i) {
    const uint32_t e = act[i];
    moe_m1_weights *w = &g_w[e];
    make_weight(g_hg[e], SK, 2u * INTER, e == 4u, &w->gu_wh, &w->gu_ws,
                &w->gu_cs, &w->gu_b);
    make_weight(g_hd[e], INTER, NOUT, e == 4u, &w->dn_wh, &w->dn_ws, &w->dn_cs,
                &w->dn_b);
  }

  static float x[SK], full[NOUT], part[NOUT], res[NACT][NOUT], gate[INTER];
  static uint8_t q[SK], mid[INTER];
  static int8_t qs[SK], mid_s[INTER];
  const float amps[] = {0.3f,  3.0f, 30.0f, 0.3f, 3.0f,
                        30.0f, 3.0f, 0.0f,  90.0f};
  const uint32_t n_rows = sizeof amps / sizeof *amps;
  uint32_t bad_k[NACT + 1] = {0}, calls = 0;
  int fail = stage_sweep() != 0u;
  for (uint32_t t = 0; t < n_rows; ++t) {
    for (uint32_t k = 0; k < SK; ++k)
      x[k] = grand() * amps[t] * ((lcg() % 97u) == 0u ? 8.0f : 1.0f);
    if (t == 6u) /* all positive: rmin = 0, zero point 0 */
      for (uint32_t k = 0; k < SK; ++k)
        x[k] = x[k] < 0.0f ? -x[k] : x[k];
    float rw[NACT], sum = 0.0f;
    for (uint32_t i = 0; i < NACT; ++i) {
      rw[i] = 0.05f + urand();
      sum += rw[i];
    }
    for (uint32_t i = 0; i < NACT; ++i)
      rw[i] = rw[i] / sum;

    int rc = dsp_call(x, act, rw, NACT, full);
    ++calls;
    if (rc != AEE_SUCCESS || hexkl_probe_us[HEXKL_PROBE_PATH] != 1u) {
      printf("MOE SPLIT row %u: DSP rc=%d path=%llu (want the M=1 path)\n", t,
             rc, (unsigned long long)hexkl_probe_us[HEXKL_PROBE_PATH]);
      return 1;
    }
    for (uint32_t i = 0; i < NACT; ++i)
      moe_m1_expert(&g_w[act[i]], x, SK, INTER, NOUT, q, qs, gate, mid, mid_s,
                    res[i]);
    for (uint32_t keep = 0; keep <= NACT; ++keep) {
      if (keep == 0u) {
        memset(part, 0, sizeof part);
      } else {
        rc = dsp_call(x, act, rw, keep, part);
        ++calls;
        if (rc != AEE_SUCCESS) {
          printf("MOE SPLIT row %u k=%u: DSP rc=%d\n", t, keep, rc);
          return 1;
        }
      }
      for (uint32_t i = keep; i < NACT; ++i)
        moe_m1_scale_add(part, res[i], rw[i], NOUT);
      uint32_t bad = 0;
      for (uint32_t c = 0; c < NOUT; ++c)
        bad += memcmp(&part[c], &full[c], sizeof(float)) != 0;
      if (bad != 0u && bad_k[keep] == 0u)
        printf("MOE SPLIT row %u (amp %g) k=%u: %u of %u differ, first "
               "[0] dsp %.9g split %.9g\n",
               t, (double)amps[t], keep, bad, NOUT, (double)full[0],
               (double)part[0]);
      bad_k[keep] += bad;
      fail |= bad != 0u;
    }
  }
  printf("moe split: rows=%u dsp_calls=%u bad k0..4 = %u %u %u %u %u\n", n_rows,
         calls, bad_k[0], bad_k[1], bad_k[2], bad_k[3], bad_k[4]);
  if (fail)
    printf("MOE SPLIT DIFFERS\n");
  else
    printf("MOE SPLIT BIT-IDENTICAL k=0..4 (M=1 path on hvx_emu vs "
           "moe_m1_det.h, K %u inter %u N %u, %u rows)\n",
           SK, INTER, NOUT, n_rows);
  return fail;
}
