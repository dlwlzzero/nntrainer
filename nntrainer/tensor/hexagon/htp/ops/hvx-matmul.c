// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-matmul.c
 * @date	18 August 2026
 * @brief	MATMUL_W8A8 / MATMUL_LOGITS: int8 x int8 vrmpy kernel over the
 *		tiled32 WEIGHTS layout. One n-tile (32 output rows) is a
 *		contiguous strip of k/4 vectors; vector i holds bytes
 *		k = 4i..4i+3 of all 32 rows, so one vrmpyacc against a 4-byte
 *		activation splat advances 32 dot products with no horizontal
 *		reduction and no vector->scalar traffic. DDR direct-read path
 *		plus a VTCM/DMA double-buffered streaming path (c->vtcm !=
 *		NULL); both run the same math on the same bytes, so their
 *		output is bit-identical. MATMUL_LOGITS is the same kernel with
 *		m=1 and fp32 output. MATMUL_W8A16 keeps the activation in fp16
 *		(no per-token int8 quantization), accumulates in fp32 and
 *		reads row-major weights; used for down_proj, whose SwiGLU
 *		input is too outlier-heavy for per-token int8.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdlib.h>

#include "dma-queue.h"
#include "htp_ops.h"
#include "hvx-f16-math.h"
#include "hvx-quant.h"

/* Ring depth for the per-worker DMA queue: only ever one chunk in flight
 * (kick c+1, wait c), but the ring needs room for 2 outstanding slots so
 * the push for c+1 does not collide with the not-yet-popped descriptor
 * for c. Must be a power of two (dma_queue_init rounds up regardless). */
#define MM_DMA_QUEUE_CAP 4

/* Tokens per weight-vector load (Task 2 raises it to 4). */
#define MM_TB 1u

struct mm_job {
  struct htp_exec_ctx *c;
  const struct nntr_htp_op_desc *d;
  uint32_t m;
  bool y_is_f32; /* MATMUL_LOGITS writes fp32, W8A8 fp16 */
};

/* Worker N-range in whole tiles: n0/n1 are multiples of 32 (the validator
 * guarantees n % 32 == 0), so rows [n0, n1) of a tiled tensor are the
 * contiguous bytes [n0 * k, n1 * k). */
static void mm_tile_range(uint32_t n, int wid, int nw, uint32_t *n0,
                          uint32_t *n1) {
  const uint32_t nt = n / NNTR_HTP_TILE_ROWS;
  *n0 = (uint32_t)(((uint64_t)nt * (uint32_t)wid) / (uint32_t)nw) *
        NNTR_HTP_TILE_ROWS;
  *n1 = (uint32_t)(((uint64_t)nt * (uint32_t)(wid + 1)) / (uint32_t)nw) *
        NNTR_HTP_TILE_ROWS;
}

/* One n-tile x tb tokens. wv is the tile's strip: k/4 vectors, since the
 * tiled32 offset kt*4096 + g*128 of k = kt*128 + 4g is 128 * (k/4). swv
 * holds the tile's 32 fp32 scales; xq, sx and y point at the first of the
 * tb token rows. int32 lanes are exact whatever the order, so the sums are
 * bit-exact against the scalar reference; the scale product keeps the
 * reference order ((float)acc * sw) * sx, in qf32 (HEXAGON.md section 7:
 * qf-format ops only), and narrows to fp16 once (Vhf_equals_Wqf32, exact
 * RNE). Always inlined with a constant tb so the accumulators stay in
 * registers. */
static inline __attribute__((always_inline)) void
mm_tile(const HVX_Vector *wv, HVX_Vector swv, const int8_t *xq, const float *sx,
        uint8_t *y, size_t y_stride, bool y_is_f32, uint32_t k, uint32_t tb) {
  HVX_Vector acc[MM_TB];
  for (uint32_t u = 0; u < tb; ++u)
    acc[u] = Q6_V_vzero();
  for (uint32_t i = 0; i < k / 4u; ++i) {
    const HVX_Vector w4 = wv[i];
    for (uint32_t u = 0; u < tb; ++u) {
      const uint32_t *xw = (const uint32_t *)(const void *)(xq + (size_t)u * k);
      acc[u] = Q6_Vw_vrmpyacc_VwVbVb(acc[u], w4, Q6_V_vsplat_R(xw[i]));
    }
  }
  for (uint32_t u = 0; u < tb; ++u) {
    HVX_Vector f = Q6_Vsf_equals_Vw(acc[u]);
    f = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(f, swv));
    f = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(f, hvx_vec_splat_f32(sx[u])));
    uint8_t *yu = y + u * y_stride;
    /* Masked stores: 32 floats = 128 B, 32 halves = 64 B. The fp16 y is
     * 64 B-aligned (n % 32 == 0), but the fp32 logits buffer comes from the
     * caller with no alignment guarantee, so neither store may assume one. */
    if (y_is_f32)
      hvx_vec_store_u(yu, 128u, f);
    else
      hvx_vec_store_u(yu, 64u, hvx_vec_f32_to_f16(f, f));
  }
}

/* Output rows [n0, n1) (whole tiles) x tokens [0, m). w_n0 is the strip of
 * row n0 - the DDR tensor at n0*k, or a DMA'd VTCM slab that starts on that
 * tile; either way strip s of the range sits at w_n0 + s*32*k. */
static void mm_tiles(const int8_t *w_n0, const float *sw, const int8_t *xq,
                     const float *sx, uint8_t *y, bool y_is_f32, uint32_t m,
                     uint32_t k, uint32_t n, uint32_t n0, uint32_t n1) {
  const size_t esz = y_is_f32 ? 4u : 2u, y_stride = (size_t)n * esz;
  for (uint32_t jn = n0; jn < n1; jn += NNTR_HTP_TILE_ROWS) {
    const HVX_Vector *wv =
      (const HVX_Vector *)(const void *)(w_n0 + (size_t)(jn - n0) * k);
    const HVX_Vector swv = hvx_vmem(sw + jn);
    uint8_t *yj = y + (size_t)jn * esz;
    for (uint32_t t = 0; t < m; ++t)
      mm_tile(wv, swv, xq + (size_t)t * k, sx + t, yj + t * y_stride, y_stride,
              y_is_f32, k, 1u);
  }
}

/* VTCM/DMA streaming path for this worker's N-slab. Returns false if the
 * slab cannot hold two 32-row tiles, in which case the caller falls back to
 * the DDR path. Activations are not copied: the kernel reads c->xq through
 * the cache (4-byte scalar loads gain nothing from VTCM). */
static bool mm_worker_vtcm(struct htp_exec_ctx *c, const int8_t *w,
                           const float *sw, uint8_t *y, bool y_is_f32,
                           uint32_t m, uint32_t k, uint32_t n, uint32_t n0,
                           uint32_t n1, int wid, int nw) {
#ifdef HTP_MM_NO_VTCM
  return false; /* measurement-only: forces the direct DDR read path */
#endif
  /* Per-worker slab rounded down to 128 B so buf[0] is vector-aligned;
   * buf[1] = buf[0] + rows_per_buf*k is too, because rows_per_buf % 32 == 0
   * and k % 128 == 0. Whole tiles per chunk keep every DMA'd slab starting
   * on a tile, which mm_tiles relies on. */
  size_t slab_sz = (c->vtcm_size / (uint32_t)nw) & ~(size_t)127;
  uint32_t rows_per_buf =
    (uint32_t)(slab_sz / 2 / k) & ~(NNTR_HTP_TILE_ROWS - 1u);
  if (rows_per_buf < NNTR_HTP_TILE_ROWS)
    return false;
  uint8_t *buf[2];
  buf[0] = c->vtcm + (size_t)wid * slab_sz;
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

    mm_tiles((const int8_t *)buf[ci & 1], sw, c->xq, c->xq_scale, y, y_is_f32,
             m, k, n, n0 + row0, n0 + row0 + rows);
  }

  dma_queue_free(q);
  free(qmem);
  return true;
}

/* W8A8 and LOGITS worker: per-worker N-slab [n0, n1) of whole tiles, VTCM
 * streaming when it fits, DDR direct read otherwise. */
static void mm_worker(void *arg, int wid, int nw) {
  struct mm_job *j = arg;
  struct htp_exec_ctx *c = j->c;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t k = d->k, n = d->n;
  const int8_t *w = (const int8_t *)htp_ref_ptr(c, d->in1);
  const float *sw = (const float *)htp_ref_ptr(c, d->in2);
  uint8_t *y = htp_ref_ptr(c, d->out);
  uint32_t n0, n1;
  mm_tile_range(n, wid, nw, &n0, &n1);
  if (n0 == n1)
    return;
  if (c->vtcm &&
      mm_worker_vtcm(c, w, sw, y, j->y_is_f32, j->m, k, n, n0, n1, wid, nw))
    return;
  mm_tiles(w + (size_t)n0 * k, sw, c->xq, c->xq_scale, y, j->y_is_f32, j->m, k,
           n, n0, n1);
}

/* fp16 x . int8 w dot in fp32: each 128B of w is sign-extended to two
 * int16 vectors, converted to hf (exact for |w| <= 127) and multiply-
 * accumulated with the matching 64-half x vectors in IEEE sf (see
 * hvx_dot_fp16 for why not qf32). k%128==0, w/x 128B aligned (validator +
 * lowering guarantee both). */
static inline float hvx_dot_fp16_i8(const __fp16 *x, const int8_t *w,
                                    uint32_t k) {
  HVX_VectorPair acc = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
  for (uint32_t i = 0; i < k; i += 128u) {
    HVX_VectorPair wh = Q6_Wh_vunpack_Vb(hvx_vmem(w + i));
    acc = hvx_vec_mpyacc_f32_f16(acc, hvx_vmem(x + i),
                                 Q6_Vhf_equals_Vh(Q6_V_lo_W(wh)));
    acc = hvx_vec_mpyacc_f32_f16(acc, hvx_vmem(x + i + 64u),
                                 Q6_Vhf_equals_Vh(Q6_V_hi_W(wh)));
  }
  return hvx_sum_sf_pair(acc);
}

/* MATMUL_W8A16 worker: same N-slab split, DDR direct read.
 * ponytail: no VTCM/DMA streaming for this kind yet - add by generalizing
 * mm_worker_vtcm if the M5 measurements show down_proj prefill needs it. */
static void mm_w8a16_worker(void *arg, int wid, int nw) {
  struct mm_job *j = arg;
  struct htp_exec_ctx *c = j->c;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t k = d->k, n = d->n, m = j->m;
  const __fp16 *x = (const __fp16 *)htp_ref_ptr(c, d->in0);
  const int8_t *w = (const int8_t *)htp_ref_ptr(c, d->in1);
  const float *sw = (const float *)htp_ref_ptr(c, d->in2);
  __fp16 *y = (__fp16 *)htp_ref_ptr(c, d->out);
  uint32_t n0 = (uint32_t)(((uint64_t)n * wid) / nw);
  uint32_t n1 = (uint32_t)(((uint64_t)n * (wid + 1)) / nw);

  for (uint32_t jn = n0; jn < n1; ++jn) {
    const int8_t *wrow = w + (size_t)jn * k;
    for (uint32_t t = 0; t < m; ++t)
      y[(size_t)t * n + jn] =
        (__fp16)(hvx_dot_fp16_i8(x + (size_t)t * k, wrow, k) * sw[jn]);
  }
}

void hvx_op_matmul_w8a16(struct htp_exec_ctx *c,
                         const struct nntr_htp_op_desc *d) {
  struct mm_job j = {c, d, htp_m(c, d), false};
  wp_run(c->pool, mm_w8a16_worker, &j);
}

void hvx_op_matmul_w8a8(struct htp_exec_ctx *c,
                        const struct nntr_htp_op_desc *d) {
  /* validator guarantees this for op-lists; unit tests build descriptors
   * by hand */
  if (d->n % NNTR_HTP_TILE_ROWS != 0u)
    return;
  const uint32_t m = htp_m(c, d), k = d->k;
  const __fp16 *x = (const __fp16 *)htp_ref_ptr(c, d->in0);
  for (uint32_t t = 0; t < m; ++t) /* one quant pass on op entry */
    c->xq_scale[t] =
      htp_quant_row_fp16(x + (size_t)t * k, c->xq + (size_t)t * k, k);
  struct mm_job j = {c, d, m, false};
  wp_run(c->pool, mm_worker, &j);
}

void hvx_op_matmul_logits(struct htp_exec_ctx *c,
                          const struct nntr_htp_op_desc *d) {
  /* validator guarantees this for op-lists; unit tests build descriptors
   * by hand */
  if (d->n % NNTR_HTP_TILE_ROWS != 0u)
    return;
  const uint32_t k = d->k;
  /* in0 is the full X fp16[n_tokens][k]; only the last token row feeds the
   * logits. The desc carries m=1, so the row offset comes from the runtime
   * chunk size c->n_tokens, not from htp_m(). */
  const __fp16 *x = (const __fp16 *)htp_ref_ptr(c, d->in0);
  const __fp16 *x_last = x + (size_t)(c->n_tokens - 1) * k;
  c->xq_scale[0] = htp_quant_row_fp16(x_last, c->xq, k);
  struct mm_job j = {c, d, 1, true};
  wp_run(c->pool, mm_worker, &j);
}
