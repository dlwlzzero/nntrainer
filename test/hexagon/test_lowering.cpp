// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_lowering.cpp
 * @date	19 August 2026
 * @brief	x86 self-check for qwen3 graph_lowering: op-list shape,
 *		WEIGHTS/ACT layout, and header fields, against a tiny config
 *		and a qwen3-0.6b-dims smoke test. Not gtest, mirrors the
 *		test_oplist_header.c self-contained main pattern.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../Applications/CausalLM/hexagon/qwen3_lowering.h"
#include "../../nntrainer/tensor/hexagon/htp/nntr_htp_common.h"

using nntrainer::hexagon::HexLoweredGraph;
using nntrainer::hexagon::HexModelConfig;
using nntrainer::hexagon::lower_qwen3;

/** @brief Print the failing check and exit 1. */
#define FAIL(msg)                                                            \
  do {                                                                       \
    std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);     \
    std::exit(1);                                                            \
  } while (0)

#define CHECK(cond, msg)                                                     \
  do {                                                                       \
    if (!(cond))                                                             \
      FAIL(msg);                                                             \
  } while (0)

namespace {

uint32_t f32_bits(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

uint64_t align128(uint64_t x) {
  return (x + 127ull) & ~static_cast<uint64_t>(127ull);
}

nntr_htp_oplist_header read_header(const HexLoweredGraph &g) {
  nntr_htp_oplist_header h;
  std::memcpy(&h, g.oplist.data(), sizeof(h));
  return h;
}

nntr_htp_op_desc read_op(const HexLoweredGraph &g, uint32_t i) {
  nntr_htp_op_desc d;
  uint64_t off = sizeof(nntr_htp_oplist_header) +
                 static_cast<uint64_t>(i) * sizeof(d);
  std::memcpy(&d, g.oplist.data() + off, sizeof(d));
  return d;
}

/** @brief 16 op kinds of one transformer layer block, plan table order. */
const uint32_t kLayerKinds[16] = {
  NNTR_HTP_OP_RMSNORM,      NNTR_HTP_OP_MATMUL_W8A8,
  NNTR_HTP_OP_MATMUL_W8A8,  NNTR_HTP_OP_MATMUL_W8A8,
  NNTR_HTP_OP_RMSNORM,      NNTR_HTP_OP_RMSNORM,
  NNTR_HTP_OP_ROPE,         NNTR_HTP_OP_ATTN,
  NNTR_HTP_OP_MATMUL_W8A8,  NNTR_HTP_OP_ADD,
  NNTR_HTP_OP_RMSNORM,      NNTR_HTP_OP_MATMUL_W8A8,
  NNTR_HTP_OP_MATMUL_W8A8,  NNTR_HTP_OP_SILU_MUL,
  NNTR_HTP_OP_MATMUL_W8A8,  NNTR_HTP_OP_ADD,
};

/** @brief Check the op sequence kinds/params for one config's graph. */
void check_sequence(const HexLoweredGraph &g, const HexModelConfig &cfg) {
  const uint32_t n_q = cfg.n_heads * cfg.head_dim;
  const uint32_t n_kv = cfg.n_kv_heads * cfg.head_dim;
  const uint32_t eps_bits = f32_bits(cfg.rms_eps);
  const uint32_t n_ops = 1u + 16u * cfg.n_layers + 2u;

  CHECK(read_header(g).n_ops == n_ops, "n_ops mismatch");
  CHECK(g.oplist.size() ==
          sizeof(nntr_htp_oplist_header) +
            (uint64_t)n_ops * sizeof(nntr_htp_op_desc),
        "oplist byte size mismatch");

  nntr_htp_op_desc embed = read_op(g, 0);
  CHECK(embed.kind == NNTR_HTP_OP_EMBED, "op0 must be EMBED");
  CHECK(embed.k == cfg.hidden, "EMBED k != hidden");

  uint32_t idx = 1;
  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    for (uint32_t j = 0; j < 16; ++j, ++idx) {
      nntr_htp_op_desc d = read_op(g, idx);
      CHECK(d.kind == kLayerKinds[j], "layer op kind mismatch");

      switch (j) {
      case 0: // L.1 RMSNORM (attn_norm)
      case 10: // L.11 RMSNORM (ffn_norm)
        CHECK(d.param0 == eps_bits, "RMSNORM param0 != eps bits");
        CHECK(!(d.flags & NNTR_HTP_FLAG_PER_HEAD), "unexpected PER_HEAD");
        break;
      case 4: // L.5 RMSNORM (q_norm)
        CHECK(d.param0 == eps_bits, "RMSNORM param0 != eps bits");
        CHECK(d.flags & NNTR_HTP_FLAG_PER_HEAD, "q_norm missing PER_HEAD");
        CHECK(d.n == n_q, "q_norm n != n_heads*head_dim");
        break;
      case 5: // L.6 RMSNORM (k_norm)
        CHECK(d.param0 == eps_bits, "RMSNORM param0 != eps bits");
        CHECK(d.flags & NNTR_HTP_FLAG_PER_HEAD, "k_norm missing PER_HEAD");
        CHECK(d.n == n_kv, "k_norm n != n_kv_heads*head_dim");
        break;
      case 1: // L.2 MATMUL wq
        CHECK(d.k == cfg.hidden && d.n == n_q, "wq k,n mismatch");
        break;
      case 2: // L.3 MATMUL wk
      case 3: // L.4 MATMUL wv
        CHECK(d.k == cfg.hidden && d.n == n_kv, "wk/wv k,n mismatch");
        break;
      case 7: // L.8 ATTN
        CHECK(d.layer == l, "ATTN layer field mismatch");
        break;
      case 8: // L.9 MATMUL wo
        CHECK(d.k == n_q && d.n == cfg.hidden, "wo k,n mismatch");
        break;
      case 11: // L.12 MATMUL gate
      case 12: // L.13 MATMUL up
        CHECK(d.k == cfg.hidden && d.n == cfg.ffn, "gate/up k,n mismatch");
        break;
      case 14: // L.15 MATMUL down
        CHECK(d.k == cfg.ffn && d.n == cfg.hidden, "down k,n mismatch");
        break;
      default:
        break;
      }
    }
  }

  nntr_htp_op_desc final_norm = read_op(g, idx++);
  CHECK(final_norm.kind == NNTR_HTP_OP_RMSNORM, "final norm kind");
  CHECK(final_norm.param0 == eps_bits, "final norm param0 != eps bits");

  nntr_htp_op_desc logits = read_op(g, idx++);
  CHECK(logits.kind == NNTR_HTP_OP_MATMUL_LOGITS, "last op != LOGITS");
  CHECK(logits.m == 1u, "MATMUL_LOGITS m != 1");
  CHECK(logits.k == cfg.hidden && logits.n == cfg.vocab, "LOGITS k,n");
  CHECK(logits.out.buf == NNTR_HTP_BUF_LOGITS, "LOGITS out buf");
  CHECK(idx == n_ops, "op count walked mismatch");

  // Tied sharing: LOGITS in1/in2 must reuse EMBED's in1/in2 offsets.
  CHECK(logits.in1.offset == embed.in1.offset, "tied embed offset");
  CHECK(logits.in2.offset == embed.in2.offset, "tied embed_scale offset");
}

/** @brief Check ACT slot offsets (read back from op refs) are disjoint. */
void check_act_disjoint(const HexLoweredGraph &g, const HexModelConfig &cfg) {
  const uint64_t mc = cfg.max_chunk;
  const uint64_t n_q = static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim;
  const uint64_t n_kv = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;

  // Slot offsets are read back from the first layer's op refs (op 1..16).
  nntr_htp_op_desc l1 = read_op(g, 1);  // out -> t
  nntr_htp_op_desc l2 = read_op(g, 2);  // out -> q
  nntr_htp_op_desc l3 = read_op(g, 3);  // out -> kb
  nntr_htp_op_desc l4 = read_op(g, 4);  // out -> vb
  nntr_htp_op_desc l8 = read_op(g, 8);  // out -> ao
  nntr_htp_op_desc l9 = read_op(g, 9);  // out -> h2
  nntr_htp_op_desc l12 = read_op(g, 12); // out -> g
  nntr_htp_op_desc l13 = read_op(g, 13); // out -> u
  nntr_htp_op_desc embed = read_op(g, 0); // out -> x

  struct Slot {
    uint64_t off, size;
  };
  Slot slots[9] = {
    {embed.out.offset, mc * cfg.hidden * 2u}, // x
    {l1.out.offset, mc * cfg.hidden * 2u},    // t
    {l2.out.offset, mc * n_q * 2u},           // q
    {l3.out.offset, mc * n_kv * 2u},          // kb
    {l4.out.offset, mc * n_kv * 2u},          // vb
    {l8.out.offset, mc * n_q * 2u},           // ao
    {l9.out.offset, mc * cfg.hidden * 2u},    // h2
    {l12.out.offset, mc * cfg.ffn * 2u},      // g
    {l13.out.offset, mc * cfg.ffn * 2u},      // u
  };

  for (int i = 0; i < 9; ++i) {
    CHECK(slots[i].off + slots[i].size <= g.act_size, "ACT slot exceeds size");
    for (int j = i + 1; j < 9; ++j) {
      bool disjoint = slots[i].off + slots[i].size <= slots[j].off ||
                       slots[j].off + slots[j].size <= slots[i].off;
      CHECK(disjoint, "ACT slots overlap");
    }
  }
}

/** @brief Recompute expected WEIGHTS byte size independently of the impl. */
uint64_t expected_weights_size(const HexModelConfig &cfg) {
  uint64_t cur = 0;
  auto add = [&](uint64_t bytes) { cur = align128(cur) + bytes; };
  const uint64_t n_q = static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim;
  const uint64_t n_kv = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;

  add(static_cast<uint64_t>(cfg.vocab) * cfg.hidden); // embed
  add(static_cast<uint64_t>(cfg.vocab) * 4u);          // embed_scale
  add(static_cast<uint64_t>(cfg.max_seq) * 128u * 2u);  // rope_table
  add(static_cast<uint64_t>(cfg.hidden) * 2u);          // final_norm

  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    add(n_q * cfg.hidden);                    // wq
    add(n_q * 4u);                            // wq_s
    add(n_kv * cfg.hidden);                   // wk
    add(n_kv * 4u);                           // wk_s
    add(n_kv * cfg.hidden);                   // wv
    add(n_kv * 4u);                           // wv_s
    add(static_cast<uint64_t>(cfg.hidden) * n_q); // wo
    add(static_cast<uint64_t>(cfg.hidden) * 4u);  // wo_s
    add(static_cast<uint64_t>(cfg.ffn) * cfg.hidden); // gate
    add(static_cast<uint64_t>(cfg.ffn) * 4u);         // gate_s
    add(static_cast<uint64_t>(cfg.ffn) * cfg.hidden); // up
    add(static_cast<uint64_t>(cfg.ffn) * 4u);         // up_s
    add(static_cast<uint64_t>(cfg.hidden) * cfg.ffn); // down
    add(static_cast<uint64_t>(cfg.hidden) * 4u);      // down_s
    add(static_cast<uint64_t>(cfg.hidden) * 2u); // attn_norm
    add(static_cast<uint64_t>(cfg.hidden) * 2u); // ffn_norm
    add(static_cast<uint64_t>(cfg.head_dim) * 2u); // q_norm
    add(static_cast<uint64_t>(cfg.head_dim) * 2u); // k_norm
  }
  return cur;
}

} // namespace

int main(void) {
  // --- Tiny config (deliberately hidden != n_heads*head_dim). ---
  HexModelConfig cfg{};
  cfg.n_layers = 2;
  cfg.hidden = 256;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.head_dim = 128;
  cfg.ffn = 512;
  cfg.vocab = 512;
  cfg.max_seq = 64;
  cfg.max_chunk = 8;
  cfg.rms_eps = 1e-6f;
  cfg.rope_theta = 1e6f;

  HexLoweredGraph g = lower_qwen3(cfg);

  // 1. validate() must accept the produced op-list against the sizes
  // lower_qwen3 itself reports.
  uint32_t buf_size[NNTR_HTP_BUF_COUNT];
  buf_size[NNTR_HTP_BUF_WEIGHTS] = static_cast<uint32_t>(g.weights_size);
  buf_size[NNTR_HTP_BUF_KV] = static_cast<uint32_t>(g.kv_size);
  buf_size[NNTR_HTP_BUF_ACT] = static_cast<uint32_t>(g.act_size);
  buf_size[NNTR_HTP_BUF_TOKENS] = cfg.max_chunk * 4u;
  buf_size[NNTR_HTP_BUF_LOGITS] = cfg.vocab * 4u;
  CHECK(nntr_htp_oplist_validate(g.oplist.data(),
                                 static_cast<uint32_t>(g.oplist.size()),
                                 buf_size) == 0,
        "validate() failed on tiny config");

  // 2/3/5: op count/sequence, tied sharing, per-op params.
  check_sequence(g, cfg);

  // 4. header fields match cfg.
  nntr_htp_oplist_header h = read_header(g);
  CHECK(h.magic == NNTR_HTP_OPLIST_MAGIC, "header magic");
  CHECK(h.version == NNTR_HTP_ABI_VERSION, "header version");
  CHECK(h.n_layers == cfg.n_layers, "header n_layers");
  CHECK(h.n_heads == cfg.n_heads, "header n_heads");
  CHECK(h.n_kv_heads == cfg.n_kv_heads, "header n_kv_heads");
  CHECK(h.head_dim == cfg.head_dim, "header head_dim");
  CHECK(h.hidden == cfg.hidden, "header hidden");
  CHECK(h.ffn == cfg.ffn, "header ffn");
  CHECK(h.vocab == cfg.vocab, "header vocab");
  CHECK(h.max_seq == cfg.max_seq, "header max_seq");
  CHECK(h.max_chunk == cfg.max_chunk, "header max_chunk");

  // 6. ACT slots pairwise disjoint and within act_size.
  check_act_disjoint(g, cfg);

  CHECK(g.weights_size == expected_weights_size(cfg),
        "tiny weights_size mismatch vs independent layout recompute");

  // 7. Real-dims smoke: qwen3-0.6b.
  HexModelConfig real{};
  real.n_layers = 28;
  real.hidden = 1024;
  real.n_heads = 16;
  real.n_kv_heads = 8;
  real.head_dim = 128;
  real.ffn = 3072;
  real.vocab = 151936;
  real.max_seq = 2048;
  real.max_chunk = 128;
  real.rms_eps = 1e-6f;
  real.rope_theta = 1e6f;

  HexLoweredGraph rg = lower_qwen3(real);
  CHECK(read_header(rg).n_ops == 451u, "qwen3-0.6b n_ops != 451");
  uint64_t exp_kv = 2ull * 28u * 8u * real.max_seq * 128u * 2u;
  CHECK(rg.kv_size == exp_kv, "qwen3-0.6b kv_size mismatch");

  uint32_t rbuf[NNTR_HTP_BUF_COUNT];
  rbuf[NNTR_HTP_BUF_WEIGHTS] = static_cast<uint32_t>(rg.weights_size);
  rbuf[NNTR_HTP_BUF_KV] = static_cast<uint32_t>(rg.kv_size);
  rbuf[NNTR_HTP_BUF_ACT] = static_cast<uint32_t>(rg.act_size);
  rbuf[NNTR_HTP_BUF_TOKENS] = real.max_chunk * 4u;
  rbuf[NNTR_HTP_BUF_LOGITS] = real.vocab * 4u;
  CHECK(nntr_htp_oplist_validate(rg.oplist.data(),
                                 static_cast<uint32_t>(rg.oplist.size()),
                                 rbuf) == 0,
        "validate() failed on qwen3-0.6b dims");

  CHECK(rg.weights_size == expected_weights_size(real),
        "qwen3-0.6b weights_size mismatch vs independent recompute");
  // Sanity band around the actual qwen3-0.6b parameter count (embed
  // 155.6MB + non-embedding 0.44B int8 bytes + small scale/norm/rope
  // overhead): observed ~598.6MB. See task-7-report.md for why this
  // replaces the plan's rough 655-700MB estimate.
  CHECK(rg.weights_size >= 590000000ull && rg.weights_size <= 610000000ull,
        "qwen3-0.6b weights_size out of sane band");

  std::puts("LOWER_TEST PASS");
  return 0;
}
