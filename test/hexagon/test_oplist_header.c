// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_oplist_header.c
 * @date	15 August 2026
 * @brief	x86 self-check for the op-list header validation shared by host and
 * DSP.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "../../nntrainer/tensor/hexagon/htp/nntr_htp_common.h"

/**
 * @brief Build a valid 2-op list: RMSNORM followed by MATMUL_W8A8.
 */
static void build_valid(struct nntr_htp_oplist_header *h,
                        struct nntr_htp_op_desc ops[2],
                        uint32_t buf_size[NNTR_HTP_BUF_COUNT]) {
  memset(h, 0, sizeof(*h));
  h->magic = NNTR_HTP_OPLIST_MAGIC;
  h->version = NNTR_HTP_ABI_VERSION;
  h->n_ops = 2;
  h->n_layers = 1;
  h->n_heads = 2;
  h->n_kv_heads = 2;
  h->head_dim = 128;
  h->hidden = 128;
  h->ffn = 128;
  h->vocab = 32;
  h->max_seq = 8;
  h->max_chunk = 4;
  h->weight_layout = NNTR_HTP_WEIGHT_LAYOUT_TILED32;

  memset(ops, 0, 2 * sizeof(*ops));
  /** op0: RMSNORM, x[ACT@0] * gamma[WEIGHTS@0] -> out[ACT@1024] */
  ops[0].kind = NNTR_HTP_OP_RMSNORM;
  ops[0].m = 0;
  ops[0].n = h->hidden;
  ops[0].in0.buf = NNTR_HTP_BUF_ACT;
  ops[0].in0.offset = 0;
  ops[0].in1.buf = NNTR_HTP_BUF_WEIGHTS;
  ops[0].in1.offset = 0;
  ops[0].out.buf = NNTR_HTP_BUF_ACT;
  ops[0].out.offset = 1024;

  /** op1: MATMUL_W8A8, X[ACT@1024] x W[WEIGHTS@256] -> out[ACT@2048] */
  ops[1].kind = NNTR_HTP_OP_MATMUL_W8A8;
  ops[1].m = 0;
  ops[1].k = h->hidden;
  ops[1].n = 256;
  ops[1].in0.buf = NNTR_HTP_BUF_ACT;
  ops[1].in0.offset = 1024;
  ops[1].in1.buf = NNTR_HTP_BUF_WEIGHTS;
  ops[1].in1.offset = 256;
  ops[1].in2.buf = NNTR_HTP_BUF_WEIGHTS;
  ops[1].in2.offset = 33024;
  ops[1].out.buf = NNTR_HTP_BUF_ACT;
  ops[1].out.offset = 2048;

  buf_size[NNTR_HTP_BUF_WEIGHTS] = 65536;
  buf_size[NNTR_HTP_BUF_KV] = 0;
  buf_size[NNTR_HTP_BUF_ACT] = 8192;
  buf_size[NNTR_HTP_BUF_TOKENS] = h->max_chunk * 4;
  buf_size[NNTR_HTP_BUF_LOGITS] = h->vocab * 4;
}

int main(void) {
  struct nntr_htp_oplist_header h = {NNTR_HTP_OPLIST_MAGIC,
                                     NNTR_HTP_ABI_VERSION, 0, 0};

  /* v4: reserved2[0] became weight_layout; the record stays 64 bytes. */
  assert(NNTR_HTP_ABI_VERSION == 4u);
  assert(sizeof(struct nntr_htp_oplist_header) == 64u);
  assert(offsetof(struct nntr_htp_oplist_header, weight_layout) == 52u);

  /* tile_off by example (spec P2 "tile definition"): K = 256 -> k_tiles = 2. */
  assert(nntr_htp_tile_off(0u, 0u, 256u) == 0u);
  assert(nntr_htp_tile_off(0u, 1u, 256u) == 1u);   /* k%4 is the byte */
  assert(nntr_htp_tile_off(1u, 0u, 256u) == 4u);   /* row r -> lane 4r */
  assert(nntr_htp_tile_off(0u, 4u, 256u) == 128u); /* 4k group g -> vector g */
  assert(nntr_htp_tile_off(31u, 127u, 256u) ==
         4095u); /* last byte of tile(0,0) */
  assert(nntr_htp_tile_off(0u, 128u, 256u) == 4096u); /* tile(0,1) */
  assert(nntr_htp_tile_off(32u, 0u, 256u) ==
         8192u); /* tile(1,0) = k_tiles*4096 */

  /* repack is a bijection onto [0, N*K) and tile_off is its inverse. */
  {
    enum { TN = 64, TK = 256 };
    static uint8_t src[TN * TK], dst[TN * TK], seen[TN * TK];
    uint32_t n, k;
    for (n = 0; n < (uint32_t)(TN * TK); ++n)
      src[n] = (uint8_t)(n * 31u + 7u);
    memset(seen, 0, sizeof(seen));
    nntr_htp_repack_tiled32(dst, src, TN, TK);
    for (n = 0; n < TN; ++n)
      for (k = 0; k < TK; ++k) {
        uint32_t o = nntr_htp_tile_off(n, k, TK);
        assert(o < (uint32_t)(TN * TK) && !seen[o]);
        seen[o] = 1;
        assert(dst[o] == src[n * TK + k]);
      }
  }

  assert(nntr_htp_oplist_check(&h, sizeof(h)) == 0);

  h.version = 999u;
  assert(nntr_htp_oplist_check(&h, sizeof(h)) == 3);
  h.version = NNTR_HTP_ABI_VERSION;

  h.magic = 0;
  assert(nntr_htp_oplist_check(&h, sizeof(h)) == 2);
  h.magic = NNTR_HTP_OPLIST_MAGIC;

  assert(nntr_htp_oplist_check(&h, 3) == 1);
  assert(nntr_htp_oplist_check(0, sizeof(h)) == 1);

  {
    struct {
      struct nntr_htp_oplist_header h;
      struct nntr_htp_op_desc ops[2];
    } wire;
    uint32_t buf_size[NNTR_HTP_BUF_COUNT];

    /* valid list -> 0 */
    build_valid(&wire.h, wire.ops, buf_size);
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 0);

    /* length mismatch -> 1 */
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire) - 1, buf_size) == 1);

    /* version != 2 -> 3 */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.h.version = 1u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 3);

    /* head_dim != 128 -> 4 */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.h.head_dim = 64u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 4);

    /* unknown kind -> 5 */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[0].kind = 99u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /* unaligned offset -> 5 */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[0].in0.offset = 4u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /* ACT bounds overflow -> 5 */
    build_valid(&wire.h, wire.ops, buf_size);
    buf_size[NNTR_HTP_BUF_ACT] = 10u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /* v4: unknown weight_layout -> 4 (a v3-era list has 0 here) */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.h.weight_layout = 0u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 4);

    /* tiled kinds need n % 32 == 0 -> 5; W8A16 (down, row-major) is exempt */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[1].n = 250u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
    wire.ops[1].kind = NNTR_HTP_OP_MATMUL_W8A16;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 0);

    /** EMBED needs k % 128 == 0 -> 5; k=4u (a multiple of 4, not 128) keeps
     * the vocab*k bounds check from tripping first. */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[1].kind = NNTR_HTP_OP_EMBED;
    wire.ops[1].k = 4u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);

    /* W8A16 int32 accumulation is only exact for k <= 16384 -> 5 above it */
    build_valid(&wire.h, wire.ops, buf_size);
    wire.ops[1].kind = NNTR_HTP_OP_MATMUL_W8A16;
    wire.ops[1].k = 16512u;
    assert(nntr_htp_oplist_validate(&wire, sizeof(wire), buf_size) == 5);
  }

  puts("oplist header check: PASS");
  return 0;
}
