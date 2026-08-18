// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-eltwise.c
 * @date	18 August 2026
 * @brief	ADD and SILU_MUL kernels: per-row, 64-half HVX strips
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "htp_ops.h"
#include "hvx-base.h"
#include "hvx-exp.h"
#include "hvx-inverse.h"

struct eltwise_job {
  struct htp_exec_ctx *c;
  const struct nntr_htp_op_desc *d;
  uint32_t m;
};

static void add_worker(void *arg, int wid, int nw) {
  struct eltwise_job *j = arg;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t n = d->n;
  const __fp16 *a = (const __fp16 *)htp_ref_ptr(j->c, d->in0);
  const __fp16 *b = (const __fp16 *)htp_ref_ptr(j->c, d->in1);
  __fp16 *y = (__fp16 *)htp_ref_ptr(j->c, d->out);

  uint32_t t0 = (uint32_t)(((uint64_t)j->m * wid) / nw);
  uint32_t t1 = (uint32_t)(((uint64_t)j->m * (wid + 1)) / nw);

  for (uint32_t t = t0; t < t1; ++t) {
    const __fp16 *arow = a + (size_t)t * n;
    const __fp16 *brow = b + (size_t)t * n;
    __fp16 *yrow = y + (size_t)t * n;
    for (uint32_t i = 0; i < n; i += VLEN_FP16) {
      HVX_Vector av = hvx_vmem(arow + i);
      HVX_Vector bv = hvx_vmem(brow + i);
      /* Q6_Vqf16_vadd_VhfVhf produces an unnormalized qf16 result whose
       * precision is fixed at the larger operand's ulp: for the residual
       * add, near-cancelling a/-b of similar magnitude (~4-8) quantize the
       * small true result to 2^-8 steps, failing the 1e-3/1e-4 tolerance.
       * Q6_Vhf_vadd_VhfVhf is a true IEEE fp16 add (properly normalized
       * per-lane), so use it directly instead. */
      hvx_vmem(yrow + i) = Q6_Vhf_vadd_VhfVhf(av, bv);
    }
  }
}

void hvx_op_add(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d) {
  struct eltwise_job j = {c, d, htp_m(c, d)};
  wp_run(c->pool, add_worker, &j);
}

/* silu(g) = g / (1 + exp(-g)); y = silu(g) * up. Strip = 64 halves,
 * widened to two 32-lane fp32 vectors (hvx_vec_f16_to_f32) so the vendor
 * exp/inverse register primitives (hvx-exp.h, hvx-inverse.h) can run in
 * fp32, then narrowed back (hvx_vec_f32_to_f16). No scratch buffers. */
static inline HVX_Vector silu_f32(HVX_Vector g) {
  static const float kMaxExp = 88.7228f;
  const HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
  const HVX_Vector inf = hvx_vec_splat_f32(INFINITY);
  const HVX_Vector one = hvx_vec_splat_f32(1.0f);

  HVX_Vector neg_g = hvx_vec_neg_f32(g);
  HVX_Vector e = hvx_vec_exp_f32_guard(neg_g, max_exp, inf);
  HVX_Vector denom = hvx_vec_add_f32_f32(one, e);
  HVX_Vector inv = hvx_vec_inverse_f32(denom);
  return hvx_vec_mul_f32_f32(g, inv);
}

static void silu_mul_worker(void *arg, int wid, int nw) {
  struct eltwise_job *j = arg;
  const struct nntr_htp_op_desc *d = j->d;
  const uint32_t n = d->n;
  const __fp16 *g = (const __fp16 *)htp_ref_ptr(j->c, d->in0);
  const __fp16 *u = (const __fp16 *)htp_ref_ptr(j->c, d->in1);
  __fp16 *y = (__fp16 *)htp_ref_ptr(j->c, d->out);

  uint32_t t0 = (uint32_t)(((uint64_t)j->m * wid) / nw);
  uint32_t t1 = (uint32_t)(((uint64_t)j->m * (wid + 1)) / nw);

  for (uint32_t t = t0; t < t1; ++t) {
    const __fp16 *grow = g + (size_t)t * n;
    const __fp16 *urow = u + (size_t)t * n;
    __fp16 *yrow = y + (size_t)t * n;
    for (uint32_t i = 0; i < n; i += VLEN_FP16) {
      HVX_Vector gv = hvx_vmem(grow + i);
      HVX_Vector uv = hvx_vmem(urow + i);

      HVX_VectorPair g32 = hvx_vec_f16_to_f32(gv);
      HVX_Vector s_lo = silu_f32(Q6_V_lo_W(g32));
      HVX_Vector s_hi = silu_f32(Q6_V_hi_W(g32));
      HVX_Vector silu_hf = hvx_vec_f32_to_f16(s_lo, s_hi);

      HVX_Vector prod = Q6_Vqf16_vmpy_VhfVhf(silu_hf, uv);
      hvx_vmem(yrow + i) = Q6_Vhf_equals_Vqf16(prod);
    }
  }
}

void hvx_op_silu_mul(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d) {
  struct eltwise_job j = {c, d, htp_m(c, d)};
  wp_run(c->pool, silu_mul_worker, &j);
}
