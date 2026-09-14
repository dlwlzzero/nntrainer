// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_profile.c
 * @date	15 September 2026
 * @brief	Hexagon-sim profile of the graph executor at qwen3 shape
 *		(2 layers, vocab 4096): per-op-kind pcycles for a prefill
 *		chunk at pos 0, a prefill chunk at pos 512 and n=1 decode
 *		steps at pos 512, printed as SIM_PROF lines for summ_prof.py.
 *		argv: profile <prefill0|prefill512|decode512> [n_workers]
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "htp_graph.h"
#include "ref_ops.h"
#include "sim_model.h"
#include "sim_test_util.h"
#include "worker_pool.h"

/* qwen3-0.6b shape with 2 layers and a reduced vocab; only lm_head
 * (MATMUL_LOGITS) cycles scale linearly with vocab and are rescaled by
 * summ_prof.py - EMBED is a gather, its cost is O(n_tokens * hidden) and
 * independent of the vocabulary size. */
static const struct sim_model_cfg QWEN3_2L = {
  2u, 16u, 8u, 128u, 1024u, 3072u, 4096u, 2048u, 128u, 1e-6f, 1e6f,
};
#define CHUNK 128u
#define FILL_CHUNKS 4u /* pos 0..511 before the measured pos-512 work */
#define DECODE_STEPS 8u
#define ACC_TOKENS 8u /* accuracy check size: scalar ref must stay cheap */

/* Unsized on purpose: the size check below then catches a missing name
 * instead of silently leaving a NULL entry. */
static const char *const KIND_NAME[] = {
  "EMBED",    "RMSNORM", "MATMUL_W8A8",   "ROPE",         "ATTN",
  "SILU_MUL", "ADD",     "MATMUL_LOGITS", "MATMUL_W8A16",
};
/* Same idiom as the ABI size checks in nntr_htp_common.h. */
typedef char kind_name_count_check[(sizeof(KIND_NAME) / sizeof(KIND_NAME[0]) ==
                                    (size_t)NNTR_HTP_OP_KIND_COUNT)
                                     ? 1
                                     : -1];

static void *xmemalign(const char *what, size_t bytes, int *rc) {
  void *p = memalign(128, bytes);
  if (!p) {
    printf("SIM_TEST profile FAIL alloc %s %zu\n", what, bytes);
    *rc = 1;
  }
  return p;
}

static void fill_tokens(int32_t *t, uint32_t n, uint32_t vocab) {
  for (uint32_t i = 0; i < n; ++i)
    /* frand() is signed: bias into [0,1) before the unsigned cast, a
     * negative float -> uint32_t conversion is undefined. */
    t[i] = (int32_t)((uint32_t)((frand() * 0.5f + 0.5f) * 65536.f) % vocab);
}

static void print_kinds(const char *scenario, int workers, uint32_t tokens,
                        uint32_t pos, uint64_t total, const uint64_t *cyc,
                        const uint32_t *calls, uint32_t divisor) {
  printf(
    "SIM_PROF scenario=%s workers=%d tokens=%u pos=%u total_pcycles=%llu\n",
    scenario, workers, (unsigned)tokens, (unsigned)pos,
    (unsigned long long)total);
  for (uint32_t k = 0; k < (uint32_t)NNTR_HTP_OP_KIND_COUNT; ++k) {
    uint64_t c = cyc[k] / divisor;
    uint32_t n = calls[k] / divisor;
    printf("SIM_PROF kind=%s calls=%u pcycles=%llu per_call=%llu\n",
           KIND_NAME[k], (unsigned)n, (unsigned long long)c,
           (unsigned long long)(n ? c / n : 0u));
  }
}

static int cmp_u64(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return x < y ? -1 : x > y;
}

int test_profile(void) {
  const char *scenario = sim_arg(2, "prefill0");
  const int workers = atoi(sim_arg(3, "0"));
  struct sim_model_plan P;
  struct htp_graph g;
  int rc = 0, inited = 0;
  uint32_t i;

  if (strcmp(scenario, "prefill0") && strcmp(scenario, "prefill512") &&
      strcmp(scenario, "decode512")) {
    printf("SIM_TEST profile FAIL unknown scenario %s\n", scenario);
    return 2;
  }
  if (sim_model_plan_init(&P, &QWEN3_2L)) {
    printf("SIM_TEST profile FAIL plan alloc\n");
    return 1;
  }
  printf("SIM_PROF model layers=%u hidden=%u ffn=%u vocab=%u max_seq=%u "
         "weights=%u act=%u kv=%u n_ops=%u\n",
         (unsigned)P.cfg.n_layers, (unsigned)P.cfg.hidden, (unsigned)P.cfg.ffn,
         (unsigned)P.cfg.vocab, (unsigned)P.cfg.max_seq, (unsigned)P.wtotal,
         (unsigned)P.atotal, (unsigned)P.kv_bytes, (unsigned)P.n_ops);

  uint8_t *w = xmemalign("weights", P.wtotal, &rc);
  uint8_t *act = xmemalign("act", P.atotal, &rc);
  uint8_t *kv = xmemalign("kv", P.kv_bytes, &rc);
  uint8_t *ol = xmemalign("oplist", P.oplist_len, &rc);
  float *logits = xmemalign("logits", P.cfg.vocab * sizeof(float), &rc);
  int32_t *tok = xmemalign("tokens", (FILL_CHUNKS + 1u) * CHUNK * 4u, &rc);
  /* Reference executor: weights are read-only so it shares w; it needs its
   * own KV/ACT and logits. */
  uint8_t *ract = xmemalign("ref act", P.atotal, &rc);
  uint8_t *rkv = xmemalign("ref kv", P.kv_bytes, &rc);
  float *rlogits = xmemalign("ref logits", P.cfg.vocab * sizeof(float), &rc);
  if (rc)
    goto out;

  sim_model_fill_weights(&P, w);
  sim_model_build_oplist(&P, ol);
  fill_tokens(tok, (FILL_CHUNKS + 1u) * CHUNK, P.cfg.vocab);
  memset(act, 0, P.atotal);
  memset(kv, 0, P.kv_bytes);

  if (htp_graph_init_ex(&g, ol, P.oplist_len, w, P.wtotal, kv, P.kv_bytes, act,
                        P.atotal, workers)) {
    printf("SIM_TEST profile FAIL init\n");
    rc = 1;
    goto out;
  }
  inited = 1;
  const int nw = wp_size(g.ctx.pool);
  printf("SIM_PROF pool workers=%d vtcm=%u\n", nw, (unsigned)g.ctx.vtcm_size);

  uint64_t cyc[NNTR_HTP_OP_KIND_COUNT];
  uint32_t calls[NNTR_HTP_OP_KIND_COUNT];

  if (!strcmp(scenario, "prefill0")) {
    /* Accuracy first, on a small chunk the scalar reference can afford.
     * Tolerance matches tools/hexagon/find_divergence.py (atol/rtol 0.1):
     * per-token int8 re-binning amplifies 1-ulp fp16 differences ~2x per
     * layer, and at this shape the 8-token logits (rms ~0.8) reach
     * max_abs ~0.08 against the scalar reference with no single culprit op
     * (logs/hexagon/sim_diverge_qwen3_2l.log). The tiny graph test keeps
     * its 3e-2/5e-2 bound. */
    if (htp_graph_forward(&g, tok, ACC_TOKENS, 0, logits, P.cfg.vocab)) {
      printf("SIM_TEST profile FAIL forward (accuracy)\n");
      rc = 1;
      goto out;
    }
    memset(ract, 0, P.atotal);
    memset(rkv, 0, P.kv_bytes);
    ref_graph_forward(ol, w, rkv, ract, tok, ACC_TOKENS, 0, rlogits);
    rc =
      cmp_f("profile_prefill_acc", rlogits, logits, P.cfg.vocab, 1e-1f, 1e-1f);
    if (rc)
      goto out;

    /* Measured chunk: 128 tokens at pos 0 on a clean cache. */
    memset(act, 0, P.atotal);
    memset(kv, 0, P.kv_bytes);
    uint64_t pc = 0;
    htp_graph_profile_reset(&g);
    if (htp_graph_forward_upto(&g, tok, CHUNK, 0, logits, P.cfg.vocab, P.n_ops,
                               &pc)) {
      printf("SIM_TEST profile FAIL forward prefill0\n");
      rc = 1;
      goto out;
    }
    htp_graph_profile_get(&g, cyc, calls);
    print_kinds(scenario, nw, CHUNK, 0, pc, cyc, calls, 1u);
  } else {
    /* Fill positions 0..511 (not measured). */
    for (i = 0; i < FILL_CHUNKS; ++i) {
      if (htp_graph_forward(&g, tok + i * CHUNK, CHUNK, i * CHUNK, logits,
                            P.cfg.vocab)) {
        printf("SIM_TEST profile FAIL fill chunk %u\n", (unsigned)i);
        rc = 1;
        goto out;
      }
    }
    const uint32_t pos = FILL_CHUNKS * CHUNK; /* 512 */
    if (!strcmp(scenario, "prefill512")) {
      uint64_t pc = 0;
      htp_graph_profile_reset(&g);
      if (htp_graph_forward_upto(&g, tok + pos, CHUNK, pos, logits, P.cfg.vocab,
                                 P.n_ops, &pc)) {
        printf("SIM_TEST profile FAIL forward prefill512\n");
        rc = 1;
        goto out;
      }
      htp_graph_profile_get(&g, cyc, calls);
      print_kinds(scenario, nw, CHUNK, pos, pc, cyc, calls, 1u);
    } else {
      uint64_t steps[DECODE_STEPS];
      htp_graph_profile_reset(&g);
      for (i = 0; i < DECODE_STEPS; ++i) {
        if (htp_graph_forward_upto(&g, tok + pos + i, 1u, pos + i, logits,
                                   P.cfg.vocab, P.n_ops, &steps[i])) {
          printf("SIM_TEST profile FAIL decode step %u\n", (unsigned)i);
          rc = 1;
          goto out;
        }
      }
      htp_graph_profile_get(&g, cyc, calls);
      qsort(steps, DECODE_STEPS, sizeof(steps[0]), cmp_u64);
      const uint64_t median =
        (steps[DECODE_STEPS / 2u - 1u] + steps[DECODE_STEPS / 2u]) / 2u;
      print_kinds(scenario, nw, 1u, pos, median, cyc, calls, DECODE_STEPS);
    }
  }

out:
  if (inited)
    htp_graph_destroy(&g);
  free(rlogits);
  free(rkv);
  free(ract);
  free(tok);
  free(logits);
  free(ol);
  free(kv);
  free(act);
  free(w);
  sim_model_plan_free(&P);
  if (rc)
    return 1;
  printf("SIM_TEST profile PASS\n");
  return 0;
}
