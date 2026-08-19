// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-attn.c
 * @date	19 August 2026
 * @brief	Fused ATTN kernel: KV-cache append, causal SDPA and GQA.
 *		Workers split by kv head; scores are fp32 in per-worker
 *		scratch, softmax uses the borrowed HVX exp, and the output
 *		is accumulated in qf32 vector pairs (qf16 adds are known to
 *		lose precision under cancellation and are not used).
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <math.h>
#include <string.h>

#include "htp_ops.h"
#include "hvx-exp.h"
#include "hvx-f16-math.h"

struct attn_job {
  struct htp_exec_ctx *c;
  const struct nntr_htp_op_desc *d;
  uint32_t m;
};

static void attn_worker(void *arg, int wid, int nw) {
  struct attn_job *j = arg;
  struct htp_exec_ctx *c = j->c;
  const struct nntr_htp_op_desc *d = j->d;
  const struct nntr_htp_oplist_header *cfg = c->cfg;
  const uint32_t hd = cfg->head_dim; /* 128 by design: 2 hf vectors/row */
  const uint32_t n_heads = cfg->n_heads, n_kv = cfg->n_kv_heads;
  const uint32_t max_seq = cfg->max_seq, group = n_heads / n_kv;
  const uint32_t m = j->m, pos = c->pos;
  const __fp16 *q = (const __fp16 *)htp_ref_ptr(c, d->in0);
  const __fp16 *kin = (const __fp16 *)htp_ref_ptr(c, d->in1);
  const __fp16 *vin = (const __fp16 *)htp_ref_ptr(c, d->in2);
  __fp16 *out = (__fp16 *)htp_ref_ptr(c, d->out);
  __fp16 *kv = (__fp16 *)c->buf[NNTR_HTP_BUF_KV];
  const size_t v_off = (size_t)cfg->n_layers * n_kv * max_seq * hd;
  float *scores = c->attn_scratch + (size_t)wid * max_seq;
  float scale;
  memcpy(&scale, &d->param0, sizeof(scale));

  uint32_t h0 = (uint32_t)(((uint64_t)n_kv * wid) / nw);
  uint32_t h1 = (uint32_t)(((uint64_t)n_kv * (wid + 1)) / nw);

  for (uint32_t h = h0; h < h1; ++h) {
    __fp16 *kh = kv + ((size_t)d->layer * n_kv + h) * max_seq * hd;
    __fp16 *vh = kh + v_off;

    /* 1) KV append: rows [pos, pos+m) of this kv head. */
    for (uint32_t t = 0; t < m; ++t) {
      memcpy(kh + (size_t)(pos + t) * hd, kin + ((size_t)t * n_kv + h) * hd,
             hd * sizeof(__fp16));
      memcpy(vh + (size_t)(pos + t) * hd, vin + ((size_t)t * n_kv + h) * hd,
             hd * sizeof(__fp16));
    }

    /* 2) SDPA for every q head in this kv head's GQA group. */
    for (uint32_t g = 0; g < group; ++g) {
      const uint32_t hq = h * group + g;
      for (uint32_t t = 0; t < m; ++t) {
        const __fp16 *qrow = q + ((size_t)t * n_heads + hq) * hd;
        const uint32_t L = pos + t + 1; /* causal: attend to [0, pos+t] */

        float mx = -INFINITY;
        for (uint32_t p = 0; p < L; ++p) {
          float s = scale * hvx_dot_fp16(qrow, kh + (size_t)p * hd, hd);
          scores[p] = s;
          if (s > mx)
            mx = s;
        }
        for (uint32_t p = 0; p < L; ++p)
          scores[p] -= mx;
        hvx_exp_f32((uint8_t *)scores, (const uint8_t *)scores, (int)L, false);
        float sum = 0.f;
        for (uint32_t p = 0; p < L; ++p)
          sum += scores[p];
        const float inv = 1.0f / sum;

        /* out[t,hq] = sum_p scores[p] * V[h][p], accumulated as two
         * qf32 vector pairs (V row = 2 hf vectors), initialized from
         * p=0 (L >= 1 always). qf16 adds are forbidden here. */
        HVX_Vector pv = hvx_vec_splat_f16((__fp16)scores[0]);
        HVX_VectorPair w0 = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(vh), pv);
        HVX_VectorPair w1 =
          Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(vh + VLEN_FP16), pv);
        HVX_Vector a0l = Q6_V_lo_W(w0), a0h = Q6_V_hi_W(w0);
        HVX_Vector a1l = Q6_V_lo_W(w1), a1h = Q6_V_hi_W(w1);
        for (uint32_t p = 1; p < L; ++p) {
          const __fp16 *vrow = vh + (size_t)p * hd;
          pv = hvx_vec_splat_f16((__fp16)scores[p]);
          w0 = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(vrow), pv);
          w1 = Q6_Wqf32_vmpy_VhfVhf(hvx_vmem(vrow + VLEN_FP16), pv);
          a0l = Q6_Vqf32_vadd_Vqf32Vqf32(a0l, Q6_V_lo_W(w0));
          a0h = Q6_Vqf32_vadd_Vqf32Vqf32(a0h, Q6_V_hi_W(w0));
          a1l = Q6_Vqf32_vadd_Vqf32Vqf32(a1l, Q6_V_lo_W(w1));
          a1h = Q6_Vqf32_vadd_Vqf32Vqf32(a1h, Q6_V_hi_W(w1));
        }

        /* hf convert (inverse of the vmpy interleave), then 1/sum. */
        __fp16 *orow = out + ((size_t)t * n_heads + hq) * hd;
        HVX_Vector iv = hvx_vec_splat_f16((__fp16)inv);
        HVX_Vector o0 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a0h, a0l));
        HVX_Vector o1 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(a1h, a1l));
        hvx_vmem(orow) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(o0, iv));
        hvx_vmem(orow + VLEN_FP16) =
          Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(o1, iv));
      }
    }
  }
}

void hvx_op_attn(struct htp_exec_ctx *c, const struct nntr_htp_op_desc *d) {
  struct attn_job j = {c, d, htp_m(c, d)};
  wp_run(c->pool, attn_worker, &j);
}
