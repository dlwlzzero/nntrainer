// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_matmul_dma.c
 * @date	18 August 2026
 * @brief	Hexagon-sim test for the MATMUL_W8A8 VTCM/DMA streaming path:
 *		checks it against the scalar reference and against the DDR
 *		direct-read path for bit-exact fp16 output at three VTCM sizes
 *		(4 MB, 256 KB, 64 KB fallback).
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <HAP_compute_res.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "htp_ops.h"
#include "ref_ops.h"
#include "sim_test_util.h"

static uint32_t align128(uint32_t n) { return (n + 127u) & ~127u; }

int test_matmul_dma(void) {
  const uint32_t m = 8, k = 1024, n = 3072;

  uint32_t off_x = 0;
  uint32_t off_w = align128(off_x + m * k * (uint32_t)sizeof(__fp16));
  uint32_t off_sw = align128(off_w + n * k);
  uint32_t off_y_ddr = align128(off_sw + n * (uint32_t)sizeof(float));
  uint32_t off_y_dma = align128(off_y_ddr + m * n * (uint32_t)sizeof(__fp16));
  uint32_t total = align128(off_y_dma + m * n * (uint32_t)sizeof(__fp16));

  uint8_t *act = memalign(128, total);
  __fp16 *x = (__fp16 *)(act + off_x);
  int8_t *w = (int8_t *)(act + off_w);
  float *sw = (float *)(act + off_sw);
  __fp16 *y_ddr = (__fp16 *)(act + off_y_ddr);
  __fp16 *y_dma = (__fp16 *)(act + off_y_dma);

  for (uint32_t i = 0; i < m * k; ++i)
    x[i] = (__fp16)frand();
  int8_t *wrm = malloc((size_t)n * k);
  for (uint32_t i = 0; i < n * k; ++i)
    wrm[i] = (int8_t)(frand() * 127.f);
  nntr_htp_repack_tiled32((uint8_t *)w, (const uint8_t *)wrm, n, k);
  free(wrm);
  for (uint32_t j = 0; j < n; ++j)
    sw[j] = 0.001f + 0.019f * (frand() * 0.5f + 0.5f);

  struct htp_exec_ctx c;
  memset(&c, 0, sizeof(c));
  c.buf[NNTR_HTP_BUF_ACT] = act;
  c.buf_size[NNTR_HTP_BUF_ACT] = total;
  c.pool = wp_create(0);
  c.xq = memalign(128, (size_t)m * k);
  c.xq_scale = malloc((size_t)m * sizeof(float));

  struct nntr_htp_op_desc d;
  memset(&d, 0, sizeof(d));
  d.kind = NNTR_HTP_OP_MATMUL_W8A8;
  d.m = m;
  d.k = k;
  d.n = n;
  d.in0.buf = NNTR_HTP_BUF_ACT;
  d.in0.offset = off_x;
  d.in1.buf = NNTR_HTP_BUF_ACT;
  d.in1.offset = off_w;
  d.in2.buf = NNTR_HTP_BUF_ACT;
  d.in2.offset = off_sw;
  d.out.buf = NNTR_HTP_BUF_ACT;

  /* DDR direct-read path. */
  d.out.offset = off_y_ddr;
  hvx_op_matmul_w8a8(&c, &d);

  compute_res_attr_t rattr;
  HAP_compute_res_attr_init(&rattr);
  HAP_compute_res_attr_set_vtcm_param(&rattr, 4 * 1024 * 1024, 1);
  unsigned ctx_id = HAP_compute_res_acquire(&rattr, 10000 /*us*/);
  void *vtcm = ctx_id ? HAP_compute_res_attr_get_vtcm_ptr(&rattr) : NULL;
  if (!vtcm) {
    printf("SIM_TEST matmul_dma vtcm acquire fail\n");
    if (ctx_id)
      HAP_compute_res_release(ctx_id);
    free(c.xq);
    free(c.xq_scale);
    wp_destroy(c.pool);
    free(act);
    return 1;
  }
  c.vtcm = (uint8_t *)vtcm;

  __fp16 *y_ref = malloc((size_t)m * n * sizeof(__fp16));
  ref_matmul_w8a8(x, w, sw, y_ref, m, k, n);
  float *ref_f = malloc((size_t)m * n * sizeof(float));
  float *got_f = malloc((size_t)m * n * sizeof(float));
  for (uint32_t i = 0; i < m * n; ++i)
    ref_f[i] = (float)y_ref[i];

  /* VTCM/DMA streaming path at three slab sizes: 4 MB (what htp_graph
   * acquires), 256 KB (64 KB per worker = exactly one double-buffered
   * 32-row tile at k=1024, so every chunk is a single tile and each worker
   * pipelines 24 chunks) and 64 KB (16 KB per worker holds less than two
   * tiles, so mm_worker_vtcm must fall back to the DDR path). Each run must
   * match the scalar reference and be bit-identical to the DDR run. */
  const size_t nbytes = (size_t)m * n * sizeof(__fp16);
  const uint32_t sizes[3] = {4u << 20, 256u << 10, 64u << 10};
  int rc = 0;
  d.out.offset = off_y_dma;
  for (uint32_t s = 0; s < 3u && !rc; ++s) {
    char tag[32];
    c.vtcm_size = sizes[s];
    memset(y_dma, 0, nbytes);
    hvx_op_matmul_w8a8(&c, &d);
    for (uint32_t i = 0; i < m * n; ++i)
      got_f[i] = (float)y_dma[i];
    snprintf(tag, sizeof(tag), "matmul_dma_ref_%uk",
             (unsigned)(sizes[s] >> 10));
    if (cmp_f(tag, ref_f, got_f, m * n, 2e-3f, 1e-3f))
      rc = 1;
    if (memcmp(y_ddr, y_dma, nbytes)) {
      printf("SIM_TEST matmul_dma FAIL vtcm=%uk differs from the DDR path\n",
             (unsigned)(sizes[s] >> 10));
      rc = 1;
    }
  }
  HAP_compute_res_release(ctx_id);

  free(ref_f);
  free(got_f);
  free(y_ref);
  free(c.xq);
  free(c.xq_scale);
  wp_destroy(c.pool);
  free(act);
  if (rc)
    return 1;

  printf("SIM_TEST matmul_dma PASS\n");
  return 0;
}
