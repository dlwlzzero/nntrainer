// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_decode_hook.h
 * @date   28 Sep 2026
 * @brief  The CPU layers' one-line entry to the HTP per-token decode hook
 *         (#130): each resident op kind as one inline call, 0 without HTP
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * A layer calls its function at one decode row (to - from == 1, FP32)
 * before its own kernel and skips that kernel on 1. Everything about
 * the graph -- which op this call is, whether the parameter is bound,
 * the stretch the op belongs to -- is the backend's
 * (HtpComputeOps::decode_op_fp32); the layers only hand over the row and
 * their weights. Without ENABLE_HEXKL every call is 0 and the layers are
 * byte-for-byte what they were.
 */

#ifndef __CAUSALLM_HTP_DECODE_HOOK_H__
#define __CAUSALLM_HTP_DECODE_HOOK_H__

#ifdef ENABLE_HEXKL
#include <compute_ops.h>
#include <htp_graph_desc.h>
#endif

#include <cstdio>
#include <cstdlib>

namespace causallm {

/** @brief Measurement only (dev/norm-shadow): NNTR_NORM_SHADOW=<file> makes
 *  the decode norms also run the CPU path and append one record per call:
 *  u32 tag (0 RMSNORM on HTP, 1 QK_NORM on HTP, 2 RMSNORM on CPU), pos,
 *  n_in, n_out, then f32 in[n_in], cpu[n_out], other[n_out] (the HTP output
 *  for tag 0, the layer's real output for tag 2, and for tag 1 the DSP's
 *  normed q | k from the rmsnorm_det_f32 entry, #164 -- zeros without an
 *  accelerator). */
inline std::FILE *normShadowFile() {
  static std::FILE *f = [] {
    const char *p = std::getenv("NNTR_NORM_SHADOW");
    return p ? std::fopen(p, "wb") : nullptr;
  }();
  return f;
}

inline void normShadowWrite(unsigned tag, unsigned pos, const float *in,
                            unsigned n_in, const float *cpu,
                            const float *other, unsigned n_out) {
  std::FILE *f = normShadowFile();
  if (!f)
    return;
  const unsigned h[4] = {tag, pos, n_in, n_out};
  std::fwrite(h, sizeof(unsigned), 4, f);
  std::fwrite(in, sizeof(float), n_in, f);
  std::fwrite(cpu, sizeof(float), n_out, f);
  if (other) {
    std::fwrite(other, sizeof(float), n_out, f);
  } else {
    for (unsigned i = 0; i < n_out; ++i) {
      const float z = 0.0f;
      std::fwrite(&z, sizeof(float), 1, f);
    }
  }
  std::fflush(f);
}

/** @brief Measurement only (dev/attn-shadow-170, never merged):
 *  NNTR_ATTN_SHADOW=<file> makes every decode attention the HTP runs also
 *  run the CPU's fp16 attention on the same row (its own cache kept
 *  current) and append one record per call: u32 tag 3, pos, the layer's
 *  ordinal, n_row, n_out, then f32 row[n_row] (q | k | v before RoPE),
 *  cpu[n_out], htp[n_out]. */
inline std::FILE *attnShadowFile() {
  static std::FILE *f = [] {
    const char *p = std::getenv("NNTR_ATTN_SHADOW");
    return p ? std::fopen(p, "wb") : nullptr;
  }();
  return f;
}

inline void attnShadowWrite(unsigned pos, unsigned ordinal, const float *row,
                            unsigned n_row, const float *cpu, const float *htp,
                            unsigned n_out) {
  std::FILE *f = attnShadowFile();
  if (!f)
    return;
  const unsigned h[5] = {3u, pos, ordinal, n_row, n_out};
  std::fwrite(h, sizeof(unsigned), 5, f);
  std::fwrite(row, sizeof(float), n_row, f);
  std::fwrite(cpu, sizeof(float), n_out, f);
  std::fwrite(htp, sizeof(float), n_out, f);
  std::fflush(f);
}

#ifdef ENABLE_HEXKL
inline int htpDecodeOp(unsigned kind, unsigned pos, const float *in,
                       unsigned in_len, float *out, unsigned out_len,
                       const float *param, unsigned param_len,
                       const float *state, unsigned state_len, float eps) {
  return nntrainer::get_htp_ops()->decode_op_fp32(kind, pos, in, in_len, out,
                                                  out_len, param, param_len,
                                                  state, state_len, eps);
}
/** @brief dev/norm-shadow: the DSP's RMSNorm of n floats in chunks. */
inline int htpDevRmsNorm(const float *x, const float *gamma, float *y,
                         unsigned n, unsigned chunk, float eps) {
  return nntrainer::get_htp_ops()->dev_rmsnorm_det_fp32(x, gamma, y, n, chunk,
                                                        eps);
}
/** @brief dev/fc-shadow only (#132 PR 2): compute_ops.h's dev_* hooks. */
inline bool fcShadowOn() { return nntrainer::get_htp_ops()->dev_shadow_on(); }
inline void fcShadowRecord(unsigned tag, const float *in, unsigned n_in,
                           const float *cpu, const float *dsp, unsigned n_out) {
  nntrainer::get_htp_ops()->dev_shadow_record(tag, in, n_in, cpu, dsp, n_out);
}
inline int htpDevAdd(const float *a, const float *b, float *c, unsigned n) {
  return nntrainer::get_htp_ops()->dev_add_f32(a, b, c, n);
}
inline int htpDevRouter(const float *x, const float *w, const float *bias,
                        unsigned K, unsigned E, unsigned top_k, float *logits,
                        unsigned *sel, float *weight) {
  return nntrainer::get_htp_ops()->dev_router_topk(x, w, bias, K, E, top_k,
                                                   logits, sel, weight);
}
#else
inline int htpDevRmsNorm(const float *, const float *, float *, unsigned,
                         unsigned, float) {
  return 0;
}
inline bool fcShadowOn() { return false; }
inline void fcShadowRecord(unsigned, const float *, unsigned, const float *,
                           const float *, unsigned) {}
inline int htpDevAdd(const float *, const float *, float *, unsigned) {
  return 0;
}
inline int htpDevRouter(const float *, const float *, const float *, unsigned,
                        unsigned, unsigned, float *, unsigned *, float *) {
  return 0;
}
inline int htpDecodeOp(unsigned, unsigned, const float *, unsigned, float *,
                       unsigned, const float *, unsigned, const float *,
                       unsigned, float) {
  return 0;
}
#define HTP_OP_RMSNORM 0
#define HTP_OP_CONV1D_GATE 2
#define HTP_OP_QK_NORM 3
#define HTP_OP_ATTN_M1 5
#define HTP_OP_ADD 6
#define HTP_OP_ROUTER_TOPK 7
#define HTP_OP_LM_HEAD 10
#endif

/** @brief RMSNorm of one row of W: y = norm(x) * gamma. */
inline int htpDecodeRmsNorm(unsigned pos, const float *x, float *y,
                            const float *gamma, unsigned W, float eps) {
  return htpDecodeOp(HTP_OP_RMSNORM, pos, x, W, y, W, gamma, W, nullptr, 0,
                     eps);
}

/** @brief Per-head q / k norm of the row q_raw | k_raw | v (len floats)
 *  with gamma = q_gamma | k_gamma (2 x head_dim). On 1 the DSP holds the
 *  normed row for the attention hook and the layer writes nothing. */
inline int htpDecodeQkNorm(unsigned pos, const float *row, unsigned len,
                           const float *gamma, unsigned gamma_len, float eps) {
  return htpDecodeOp(HTP_OP_QK_NORM, pos, row, len, nullptr, 0, gamma,
                     gamma_len, nullptr, 0, eps);
}

/** @brief Causal conv1d (L=3) + gate of one token: abc = a | b | c (3C),
 *  conv_w = w0 | w1 | w2 (3C), state = x_{t-2} | x_{t-1} (2C), y (C). */
inline int htpDecodeConvGate(unsigned pos, const float *abc, float *y,
                             unsigned C, const float *conv_w,
                             const float *state) {
  return htpDecodeOp(HTP_OP_CONV1D_GATE, pos, abc, 3 * C, y, C, conv_w, 3 * C,
                     state, 2 * C, 0.0f);
}

/** @brief Decode attention of one token: qkv = q | k | v (len floats,
 *  post-projection, pre-norm when QK_NORM is resident, else normed),
 *  out (n_heads x head_dim), rope = cos[32] | sin[32] per position. */
inline int htpDecodeAttn(unsigned pos, const float *qkv, unsigned len,
                         float *out, unsigned out_len, const float *rope,
                         unsigned rope_len) {
  return htpDecodeOp(HTP_OP_ATTN_M1, pos, qkv, len, out, out_len, rope,
                     rope_len, nullptr, 0, 0.0f);
}

/** @brief [#132] The residual add of one row: out = residual + addend
 *  (W floats). The DSP holds the residual (its slot 0), so only the
 *  addend travels; on 1 @a out is written only when the ADD ends a
 *  stretch, which the validator rules out -- the next norm hook writes. */
inline int htpDecodeAdd(unsigned pos, const float *addend, float *out,
                        unsigned W) {
  return htpDecodeOp(HTP_OP_ADD, pos, addend, W, out, W, nullptr, 0, nullptr, 0,
                     0.0f);
}

/** @brief [#132] The MoE router of one row: x (K floats, the ffn-normed
 *  row), gate_w the [K][E] gate weight, bias the E expert biases. On 1 the
 *  DSP routes and runs the experts in the same stretch; the layer skips
 *  its router, top-k and experts and writes nothing. */
inline int htpDecodeRouter(unsigned pos, const float *x, unsigned K,
                           const float *gate_w, unsigned E, const float *bias) {
  return htpDecodeOp(HTP_OP_ROUTER_TOPK, pos, x, K, nullptr, 0, gate_w, K * E,
                     bias, E, 0.0f);
}

/** @brief [#132 Part B] The tied lm_head of one row: x (K floats, the
 *  final-normed row), logits (vocab floats). Resident only with every
 *  kind (one stretch per token): this hook is its last op, so on 1 the
 *  DSP ran the whole token and @a logits holds its output. The weight was
 *  bound at load. */
inline int htpDecodeLmHead(unsigned pos, const float *x, unsigned K,
                           float *logits, unsigned vocab) {
  return htpDecodeOp(HTP_OP_LM_HEAD, pos, x, K, logits, vocab, nullptr, 0,
                     nullptr, 0, 0.0f);
}

/** @brief [#132 Part B] Whether the HTP runs the whole decode row at
 *  @a pos (every kind resident, the row handed over): a layer's FC GEMVs
 *  are then discarded work and it skips them. False without HTP. */
inline bool htpDecodeRowResident(unsigned pos) {
#ifdef ENABLE_HEXKL
  return nntrainer::get_htp_ops()->decode_row_resident(pos);
#else
  (void)pos;
  return false;
#endif
}

/** @brief [#132 Part B E3] Whether the decode tokens bring their logits
 *  back; false when the caller takes the id (htpDecodeTokenId) and nothing
 *  reads the logits. No-op without HTP. */
inline void htpDecodeWantLogits(bool want) {
#ifdef ENABLE_HEXKL
  nntrainer::get_htp_ops()->set_decode_logits(want);
#else
  (void)want;
#endif
}

/** @brief [#132 Part B E3] The bad-word ids the greedy pick sets to -inf;
 *  the NPU's pick (htpDecodeTokenId) skips them too. No-op without HTP. */
inline void htpDecodeBan(const unsigned *ids, unsigned n) {
#ifdef ENABLE_HEXKL
  nntrainer::get_htp_ops()->set_decode_ban(ids, n);
#else
  (void)ids;
  (void)n;
#endif
}

/** @brief [#132 Part B E3] The id the NPU picked for the last decode token
 *  (its argmax, first maximum) when its logits did not come back; false
 *  otherwise, and always without HTP. */
inline bool htpDecodeTokenId(unsigned *id) {
#ifdef ENABLE_HEXKL
  return nntrainer::get_htp_ops()->take_decode_token_id(id);
#else
  (void)id;
  return false;
#endif
}

/** @brief Rows [0, n_rows) of the layer whose attention hook returned 2,
 *  [n_rows][n_kv x head_dim] f32 each. */
inline bool htpDecodeKvSeed(unsigned n_rows, const float *k_rows,
                            const float *v_rows) {
#ifdef ENABLE_HEXKL
  return nntrainer::get_htp_ops()->decode_kv_seed_fp32(n_rows, k_rows, v_rows);
#else
  (void)n_rows;
  (void)k_rows;
  (void)v_rows;
  return false;
#endif
}

} // namespace causallm

#endif /* __CAUSALLM_HTP_DECODE_HOOK_H__ */
