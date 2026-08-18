// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-matmul.c
 * @date	18 August 2026
 * @brief	MATMUL_W8A8 kernel: fp16 activation x int8 weight. DDR
 *		direct-read path plus a VTCM/DMA double-buffered streaming
 *		path (used when c->vtcm != NULL); both produce bit-identical
 *		fp16 output since the streaming path only moves the same
 *		bytes through VTCM before the identical dot/scale math.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdlib.h>
#include <string.h>

#include "dma-queue.h"
#include "htp_ops.h"
#include "hvx-quant.h"

/* Ring depth for the per-worker DMA queue: only ever one chunk in flight
 * (kick c+1, wait c), but the ring needs room for 2 outstanding slots so
 * the push for c+1 does not collide with the not-yet-popped descriptor
 * for c. Must be a power of two (dma_queue_init rounds up regardless). */
#define MM_DMA_QUEUE_CAP 4

struct mm_job {
  struct htp_exec_ctx *c;
  const struct nntr_htp_op_desc *d;
  uint32_t m;
};

/* Task 6 compute: dot + scale + store for output columns [n0, n1), using
 * whatever xq/xrow the caller hands in (DDR pointers, or VTCM copies -
 * same bytes either way, so the result is bit-identical). */
static void mm_compute_range(const int8_t *w, uint32_t w_base, const float *sw,
                             const int8_t *xq, const float *xq_scale, __fp16 *y,
                             uint32_t m, uint32_t k, uint32_t n, uint32_t n0,
                             uint32_t n1) {
  for (uint32_t jn = n0; jn < n1; ++jn) {
    const int8_t *wrow = w + (size_t)(jn - w_base) * k;
    for (uint32_t t = 0; t < m; ++t) {
      int32_t acc = hvx_dot_i8(wrow, xq + (size_t)t * k, k);
      y[(size_t)t * n + jn] = (__fp16)((float)acc * sw[jn] * xq_scale[t]);
    }
  }
}

/* VTCM/DMA streaming path for this worker's N-slab. Returns false if the
 * slab is too small to hold the Xq copy plus at least one double-buffered
 * weight row, in which case the caller falls back to the DDR path. */
static bool mm_worker_vtcm(struct htp_exec_ctx *c, const int8_t *w,
                           const float *sw, __fp16 *y, uint32_t m, uint32_t k,
                           uint32_t n, uint32_t n0, uint32_t n1, int wid,
                           int nw) {
  /* Round the per-worker slab size down to a multiple of 128 first, so
   * every worker's slab base (and therefore buf[0]/buf[1]) stays 128B
   * aligned - hvx_dot_i8 does raw HVX_Vector loads and requires it. */
  size_t slab_sz = (c->vtcm_size / (uint32_t)nw) & ~(size_t)127;
  uint8_t *slab = c->vtcm + (size_t)wid * slab_sz;
  size_t xq_bytes = (size_t)m * k;

  if (xq_bytes + HEX_L2_LINE_SIZE + 2 * k > slab_sz)
    return false; /* not even one double-buffered row fits */

  size_t half = (slab_sz - xq_bytes - HEX_L2_LINE_SIZE) / 2;
  uint32_t rows_per_buf = (uint32_t)(half / k);
  if (rows_per_buf < 1)
    return false;

  int8_t *xq_local = (int8_t *)slab;
  memcpy(xq_local, c->xq, xq_bytes);
  uint8_t *buf[2];
  buf[0] = slab + xq_bytes + HEX_L2_LINE_SIZE;
  buf[1] = buf[0] + (size_t)rows_per_buf * k;

  void *qmem =
    memalign(dma_queue_alignof(), dma_queue_sizeof(MM_DMA_QUEUE_CAP));
  if (!qmem)
    return false;
  dma_queue_t q = dma_queue_init(qmem, MM_DMA_QUEUE_CAP, (uintptr_t)c->vtcm,
                                 c->vtcm_size, NULL);

  uint32_t total_rows = n1 - n0;
  uint32_t n_chunks = (total_rows + rows_per_buf - 1) / rows_per_buf;

  /* Kick the first chunk before entering the pipeline. */
  uint32_t rows0 = rows_per_buf < total_rows ? rows_per_buf : total_rows;
  dma_queue_push_ddr_to_vtcm(q, dma_make_ptr(buf[0], w + (size_t)n0 * k), k, k,
                             rows0);

  for (uint32_t ci = 0; ci < n_chunks; ++ci) {
    uint32_t row0 = ci * rows_per_buf;
    uint32_t rows = rows_per_buf;
    if (row0 + rows > total_rows)
      rows = total_rows - row0;

    if (ci + 1 < n_chunks) {
      uint32_t next_row0 = (ci + 1) * rows_per_buf;
      uint32_t next_rows = rows_per_buf;
      if (next_row0 + next_rows > total_rows)
        next_rows = total_rows - next_row0;
      dma_queue_push_ddr_to_vtcm(
        q, dma_make_ptr(buf[(ci + 1) & 1], w + (size_t)(n0 + next_row0) * k), k,
        k, next_rows);
    }

    dma_queue_pop(q); /* wait for this chunk's DMA (kicked one iteration ago) */

    mm_compute_range((const int8_t *)buf[ci & 1], n0 + row0, sw, xq_local,
                     c->xq_scale, y, m, k, n, n0 + row0, n0 + row0 + rows);
  }

  dma_queue_free(q);
  free(qmem);
  return true;
}

static void mm_worker(void *arg, int wid, int nw) {
  struct mm_job *j = arg;
  struct htp_exec_ctx *c = j->c;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t k = d->k, n = d->n;
  const int8_t *w = (const int8_t *)htp_ref_ptr(c, d->in1);
  const float *sw = (const float *)htp_ref_ptr(c, d->in2);
  __fp16 *y = (__fp16 *)htp_ref_ptr(c, d->out);
  /* Per-worker N-slab [n0,n1) - outer N, inner M (isomorphic to the VTCM
   * tile-reuse layout). */
  uint32_t n0 = (uint32_t)(((uint64_t)n * wid) / nw);
  uint32_t n1 = (uint32_t)(((uint64_t)n * (wid + 1)) / nw);

  if (c->vtcm && mm_worker_vtcm(c, w, sw, y, j->m, k, n, n0, n1, wid, nw))
    return;

  mm_compute_range(w, 0, sw, c->xq, c->xq_scale, y, j->m, k, n, n0, n1);
}

void hvx_op_matmul_w8a8(struct htp_exec_ctx *c,
                        const struct nntr_htp_op_desc *d) {
  const uint32_t m = htp_m(c, d), k = d->k;
  const __fp16 *x = (const __fp16 *)htp_ref_ptr(c, d->in0);
  for (uint32_t t = 0; t < m; ++t) /* one quant pass on op entry */
    c->xq_scale[t] =
      htp_quant_row_fp16(x + (size_t)t * k, c->xq + (size_t)t * k, k);
  struct mm_job j = {c, d, m};
  wp_run(c->pool, mm_worker, &j);
}
