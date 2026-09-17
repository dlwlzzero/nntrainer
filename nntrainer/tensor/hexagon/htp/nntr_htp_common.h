// SPDX-License-Identifier: Apache-2.0
/**
 * @file	nntr_htp_common.h
 * @date	15 August 2026
 * @brief	Op-list wire format (ABI v4: op descriptors) shared by host (arm64)
 *		and DSP (hexagon-clang). Plain C - compiled by both toolchains.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef NNTR_HTP_COMMON_H
#define NNTR_HTP_COMMON_H

#include <stdint.h>
#include <string.h>

#define NNTR_HTP_OPLIST_MAGIC 0x5054484Eu /* "NHTP" little-endian */
#define NNTR_HTP_ABI_VERSION                                                   \
  4u /* v4: tiled32 int8 projections, header weight_layout */

/** WEIGHTS layout id carried in the op-list header (v4). The only value the
 * kernels understand; anything else is rejected by the validator. */
#define NNTR_HTP_WEIGHT_LAYOUT_TILED32 1u

/** tiled32: an int8 [N][K] projection is stored as 4 KB tiles of 32 rows x
 * 128 k, n-tile outer, k-tile inner. Inside a tile, vector g (128 B) holds
 * bytes [4r, 4r+3] = w[nt*32 + r][kt*128 + 4g .. +3], so one vrmpy of
 * vector g against a 4-byte activation splat accumulates a 4-MAC partial
 * sum for 32 output rows at once (no horizontal reduction). Requires
 * N % 32 == 0 and K % 128 == 0. down_proj (MATMUL_W8A16) is NOT tiled. */
#define NNTR_HTP_TILE_ROWS 32u
#define NNTR_HTP_TILE_K 128u
#define NNTR_HTP_TILE_BYTES 4096u

enum nntr_htp_buf_id {
  NNTR_HTP_BUF_WEIGHTS = 0,
  NNTR_HTP_BUF_KV = 1,
  NNTR_HTP_BUF_ACT = 2,
  NNTR_HTP_BUF_TOKENS =
    3, /* forward() token_ids argument, updated every call */
  NNTR_HTP_BUF_LOGITS = 4, /* forward() logits argument, updated every call */
  NNTR_HTP_BUF_COUNT = 5
};

enum nntr_htp_op_kind {
  NNTR_HTP_OP_EMBED = 0,
  NNTR_HTP_OP_RMSNORM = 1,
  NNTR_HTP_OP_MATMUL_W8A8 = 2,
  NNTR_HTP_OP_ROPE = 3,
  NNTR_HTP_OP_ATTN = 4,
  NNTR_HTP_OP_SILU_MUL = 5,
  NNTR_HTP_OP_ADD = 6,
  NNTR_HTP_OP_MATMUL_LOGITS = 7,
  NNTR_HTP_OP_MATMUL_W8A16 =
    8, /* per-token int16 x . int8 w (row-major), fp16 y; down_proj */
  NNTR_HTP_OP_KIND_COUNT = 9
};

#define NNTR_HTP_FLAG_PER_HEAD 0x1u /* RMSNORM: per head_dim QK-Norm */

/**
 * @brief One tensor reference: which buffer plus a 128B-aligned byte offset.
 */
struct nntr_htp_tensor_ref {
  uint32_t buf;    /**< enum nntr_htp_buf_id */
  uint32_t offset; /**< bytes, 128B aligned */
};                 /* 8B */

/**
 * @brief One op entry of the op-list. Fixed 64B wire record.
 */
struct nntr_htp_op_desc {
  uint32_t kind, flags, layer;
  uint32_t m, k, n; /* m==0 -> substituted with n_tokens at execution time */
  struct nntr_htp_tensor_ref in0, in1, in2, out;
  uint32_t param0, param1; /* op specific (e.g. fp32 bit pattern) */
};                         /* 64B, size check required */

/**
 * @brief Leading header of every op-list buffer sent from host to DSP.
 */
struct nntr_htp_oplist_header {
  uint32_t magic, version, n_ops, reserved; /* v1 prefix kept for compat */
  uint32_t n_layers, n_heads, n_kv_heads, head_dim;
  uint32_t hidden, ffn, vocab, max_seq;
  uint32_t max_chunk;
  uint32_t weight_layout; /* v4: NNTR_HTP_WEIGHT_LAYOUT_*; was reserved2[0] */
  uint32_t reserved2[2];
}; /* 64B, size check required */

/* The struct is the wire ABI: all three toolchains must agree on 64 bytes. */
typedef char nntr_htp_oplist_header_size_check
  [(sizeof(struct nntr_htp_oplist_header) == 64) ? 1 : -1];
typedef char
  nntr_htp_op_desc_size_check[(sizeof(struct nntr_htp_op_desc) == 64) ? 1 : -1];

/**
 * @brief Byte offset of element w[n][k] inside a tiled32 [N][K] int8
 *        tensor (the inverse index of the layout above). Shared by the host
 *        packer, the DSP kernels and the scalar references so no two of
 *        them can disagree.
 */
static inline uint32_t nntr_htp_tile_off(uint32_t n, uint32_t k, uint32_t K) {
  const uint32_t k_tiles = K / NNTR_HTP_TILE_K;
  return ((n / NNTR_HTP_TILE_ROWS) * k_tiles + k / NNTR_HTP_TILE_K) *
           NNTR_HTP_TILE_BYTES +
         ((k % NNTR_HTP_TILE_K) / 4u) * 128u + (n % NNTR_HTP_TILE_ROWS) * 4u +
         (k % 4u);
}

/**
 * @brief Repack a row-major int8 [N][K] tensor into the tiled32 layout.
 *        dst and src are N*K bytes each and must not overlap. Rows are
 *        moved in 4-byte chunks (the smallest contiguous unit of a tile).
 */
static inline void nntr_htp_repack_tiled32(uint8_t *dst, const uint8_t *src,
                                           uint32_t N, uint32_t K) {
  uint32_t n, k;
  for (n = 0; n < N; ++n)
    for (k = 0; k < K; k += 4u)
      memcpy(dst + nntr_htp_tile_off(n, k, K), src + (uint64_t)n * K + k, 4u);
}

/**
 * @brief Validate an op-list buffer header.
 * @return 0 ok, 1 bad pointer/size, 2 bad magic, 3 version mismatch
 */
static inline int nntr_htp_oplist_check(const void *buf, uint32_t len) {
  struct nntr_htp_oplist_header h;
  if (buf == 0 || len < (uint32_t)sizeof(h))
    return 1;
  memcpy(&h, buf, sizeof(h));
  if (h.magic != NNTR_HTP_OPLIST_MAGIC)
    return 2;
  if (h.version != NNTR_HTP_ABI_VERSION)
    return 3;
  return 0;
}

/**
 * @brief Bounds-check one tensor reference against its target buffer.
 * @return 0 ok, 1 out of bounds
 */
static inline int
nntr_htp_check_ref(uint32_t buf_id, uint32_t offset, uint64_t bytes,
                   const uint32_t buf_size[NNTR_HTP_BUF_COUNT]) {
  if ((uint64_t)offset + bytes > (uint64_t)buf_size[buf_id])
    return 1;
  return 0;
}

/**
 * @brief The per-call token-id gate shared by the DSP executor and the
 *        host pre-check: every id must index the vocab-row tables. One
 *        unsigned compare also rejects negative ids (they wrap past 2^31).
 * @return 1 if all n ids are < vocab, 0 otherwise
 */
static inline int nntr_htp_token_ids_ok(const int32_t *ids, uint32_t n,
                                        uint32_t vocab) {
  uint32_t i;
  for (i = 0; i < n; ++i)
    if ((uint32_t)ids[i] >= vocab)
      return 0;
  return 1;
}

/**
 * @brief Op row count: d->m if set, else the current call's n_tokens. The
 *        one rule every executor (DSP kernels, x86 reference, simulator
 *        reference) applies to a descriptor.
 */
static inline uint32_t nntr_htp_op_rows(const struct nntr_htp_op_desc *d,
                                        uint32_t n_tokens) {
  return d->m ? d->m : n_tokens;
}

/** @brief KV cache bytes: K and V, fp16,
 * [n_layers][n_kv_heads][max_seq][head_dim]. */
static inline uint64_t nntr_htp_kv_bytes(uint32_t n_layers, uint32_t n_kv_heads,
                                         uint32_t max_seq, uint32_t head_dim) {
  return 2ull * n_layers * n_kv_heads * max_seq * head_dim * 2u;
}

/** @brief Wire size of an op-list: header + n_ops descriptors. */
static inline uint64_t nntr_htp_oplist_bytes(uint32_t n_ops) {
  return (uint64_t)sizeof(struct nntr_htp_oplist_header) +
         (uint64_t)n_ops * sizeof(struct nntr_htp_op_desc);
}

/**
 * @brief Byte extent of each tensor ref of one op for `rows` token rows.
 */
struct nntr_htp_op_extent {
  uint64_t in0, in1, in2, out; /**< bytes each ref must cover for `rows` */
  uint32_t used;               /**< bit i set: ref i (in0,in1,in2,out) is read
                                  or written by the kind */
  uint32_t out_alias_in0;      /**< ROPE: out is in place on in0 and is not
                                  a separate extent */
};

/**
 * @brief The per-kind operand shape table: how many bytes of every ref an
 *        op of kind d->kind touches for `rows` token rows, with k/n from the
 *        descriptor and the model dims from the header. The validator passes
 *        h->max_chunk (the widest count forward() can pass); executors and
 *        dump tools pass n_tokens. d->m is ignored: the validator forces it
 *        to 0 on every kind but LOGITS, whose kernel ignores it.
 * @return 0 ok, 1 unknown kind
 */
static inline int nntr_htp_op_extent(const struct nntr_htp_oplist_header *h,
                                     const struct nntr_htp_op_desc *d,
                                     uint32_t rows,
                                     struct nntr_htp_op_extent *e) {
  const uint64_t r = rows;
  const uint64_t q_row = (uint64_t)h->n_heads * 128u * 2u;
  const uint64_t kv_row = (uint64_t)h->n_kv_heads * 128u * 2u;

  memset(e, 0, sizeof(*e));
  switch (d->kind) {
  case NNTR_HTP_OP_EMBED: /* TOKENS ids, tiled int8 table, fp32 scales */
    e->in0 = r * 4u;
    e->in1 = (uint64_t)h->vocab * d->k;
    e->in2 = (uint64_t)h->vocab * 4u;
    e->out = r * d->k * 2u;
    e->used = 0xFu;
    break;
  case NNTR_HTP_OP_RMSNORM: /* gamma is head_dim long under PER_HEAD */
    e->in0 = r * d->n * 2u;
    e->in1 = (d->flags & NNTR_HTP_FLAG_PER_HEAD) ? (uint64_t)h->head_dim * 2u
                                                 : (uint64_t)d->n * 2u;
    e->out = r * d->n * 2u;
    e->used = 0xBu;
    break;
  case NNTR_HTP_OP_MATMUL_W8A8:
  case NNTR_HTP_OP_MATMUL_W8A16: /* same operand layout */
    e->in0 = r * d->k * 2u;
    e->in1 = (uint64_t)d->n * d->k;
    e->in2 = (uint64_t)d->n * 4u;
    e->out = r * d->n * 2u;
    e->used = 0xFu;
    break;
  case NNTR_HTP_OP_ROPE: /* q in place, k, cos/sin table of max_seq rows */
    e->in0 = r * q_row;
    e->in1 = r * kv_row;
    e->in2 = (uint64_t)h->max_seq * 128u * 2u;
    e->out = e->in0;
    e->used = 0x7u;
    e->out_alias_in0 = 1u;
    break;
  case NNTR_HTP_OP_ATTN: /* in1/in2: this chunk's fresh K/V rows; the cache
                          * itself is nntr_htp_kv_bytes, not a ref */
    e->in0 = r * q_row;
    e->in1 = r * kv_row;
    e->in2 = r * kv_row;
    e->out = r * q_row;
    e->used = 0xFu;
    break;
  case NNTR_HTP_OP_SILU_MUL:
  case NNTR_HTP_OP_ADD:
    e->in0 = r * d->n * 2u;
    e->in1 = r * d->n * 2u;
    e->out = r * d->n * 2u;
    e->used = 0xBu;
    break;
  case NNTR_HTP_OP_MATMUL_LOGITS: /* reads row rows-1 of in0, one fp32 row out
                                   */
    e->in0 = r * d->k * 2u;
    e->in1 = (uint64_t)d->n * d->k;
    e->in2 = (uint64_t)d->n * 4u;
    e->out = (uint64_t)d->n * 4u;
    e->used = 0xFu;
    break;
  default:
    return 1;
  }
  return 0;
}

/**
 * @brief Validate an op-list buffer: header fields, op kinds/refs, and
 * per-op tensor bounds against the caller-provided buffer sizes.
 *
 * rc 5 covers every per-op rule: unknown kind, buf id or unaligned offset,
 * m != 0 on any kind but LOGITS, k == 0 or k % 128 on the four int8 kinds,
 * k > 16384 on W8A16, n % 32 on the tiled kinds, n % 64 on
 * RMSNORM/ADD/SILU_MUL, PER_HEAD n % head_dim, ATTN.layer >= n_layers, a
 * LOGITS in0 that cannot hold max_chunk rows, and any tensor ref past its
 * buffer (ROPE's out is in-place, == in0, and is not checked separately).
 * The byte extent of every ref comes from nntr_htp_op_extent at
 * rows = max_chunk.
 * @return 0 ok, 1 bad pointer/size, 2 bad magic, 3 version mismatch,
 * 4 bad header field (incl. unknown weight_layout), 5 bad op
 */
static inline int
nntr_htp_oplist_validate(const void *buf, uint32_t len,
                         const uint32_t buf_size[NNTR_HTP_BUF_COUNT]) {
  struct nntr_htp_oplist_header h;
  uint32_t i;

  if (buf == 0 || len < (uint32_t)sizeof(h))
    return 1;
  memcpy(&h, buf, sizeof(h));
  if (h.magic != NNTR_HTP_OPLIST_MAGIC)
    return 2;
  if (h.version != NNTR_HTP_ABI_VERSION)
    return 3;
  if ((uint64_t)len != nntr_htp_oplist_bytes(h.n_ops))
    return 1;
  if (h.head_dim != 128u || h.hidden % 64u != 0u || h.ffn % 64u != 0u ||
      h.n_kv_heads == 0u || h.n_heads % h.n_kv_heads != 0u ||
      h.max_chunk < 1u || h.weight_layout != NNTR_HTP_WEIGHT_LAYOUT_TILED32)
    return 4;

  for (i = 0; i < h.n_ops; ++i) {
    struct nntr_htp_op_desc d;
    struct nntr_htp_op_extent e;
    const struct nntr_htp_tensor_ref *ref[4];
    uint64_t bytes[4];
    uint32_t j;
    memcpy(&d, (const uint8_t *)buf + sizeof(h) + (uint64_t)i * sizeof(d),
           sizeof(d));

    if (d.kind >= (uint32_t)NNTR_HTP_OP_KIND_COUNT)
      return 5;
    if (d.in0.buf >= (uint32_t)NNTR_HTP_BUF_COUNT ||
        d.in1.buf >= (uint32_t)NNTR_HTP_BUF_COUNT ||
        d.in2.buf >= (uint32_t)NNTR_HTP_BUF_COUNT ||
        d.out.buf >= (uint32_t)NNTR_HTP_BUF_COUNT)
      return 5;
    if (d.in0.offset % 128u != 0u || d.in1.offset % 128u != 0u ||
        d.in2.offset % 128u != 0u || d.out.offset % 128u != 0u)
      return 5;
    if ((d.kind == (uint32_t)NNTR_HTP_OP_MATMUL_W8A8 ||
         d.kind == (uint32_t)NNTR_HTP_OP_MATMUL_W8A16 ||
         d.kind == (uint32_t)NNTR_HTP_OP_MATMUL_LOGITS ||
         d.kind == (uint32_t)NNTR_HTP_OP_EMBED) &&
        (d.k == 0u || d.k % 128u != 0u))
      return 5;
    /* W8A16 accumulates int16 x int8 in int32 lanes: exact for k <= 16384. */
    if (d.kind == (uint32_t)NNTR_HTP_OP_MATMUL_W8A16 && d.k > 16384u)
      return 5;
    /** tiled32 projections are whole 32-row tiles; down_proj (W8A16) stays
     * row-major and is exempt. EMBED's table has vocab rows. */
    if ((d.kind == (uint32_t)NNTR_HTP_OP_MATMUL_W8A8 ||
         d.kind == (uint32_t)NNTR_HTP_OP_MATMUL_LOGITS) &&
        d.n % NNTR_HTP_TILE_ROWS != 0u)
      return 5;
    if (d.kind == (uint32_t)NNTR_HTP_OP_EMBED &&
        h.vocab % NNTR_HTP_TILE_ROWS != 0u)
      return 5;

    /** m is the per-call token count on every kind: forward() validates
     * n_tokens, pos and the ids for exactly that many rows (TOKENS holds
     * n_tokens ids on the wire, KV/rope rows run [pos, pos + m), the quant
     * scratch has max_chunk rows), so a descriptor-fixed m would bypass
     * all of it. MATMUL_LOGITS ignores m (it reads row n_tokens - 1) and
     * the lowering writes 1 there. */
    if (d.m != 0u && d.kind != (uint32_t)NNTR_HTP_OP_MATMUL_LOGITS)
      return 5;

    switch (d.kind) {
    case NNTR_HTP_OP_RMSNORM:
      /** Rows are consumed as whole 64-half vectors; PER_HEAD additionally
       * splits every row into head_dim chunks. */
      if (d.n % 64u != 0u)
        return 5;
      if ((d.flags & NNTR_HTP_FLAG_PER_HEAD) && d.n % h.head_dim != 0u)
        return 5;
      break;
    case NNTR_HTP_OP_ATTN:
      /* d.layer selects the KV cache slice; the cache must hold n_layers. */
      if (d.layer >= h.n_layers)
        return 5;
      if (nntr_htp_kv_bytes(h.n_layers, h.n_kv_heads, h.max_seq, h.head_dim) >
          (uint64_t)buf_size[NNTR_HTP_BUF_KV])
        return 5;
      if (h.max_seq % 64u) /* K^T rows are streamed 64 positions/vector */
        return 5;
      break;
    case NNTR_HTP_OP_SILU_MUL:
    case NNTR_HTP_OP_ADD:
      if (d.n % 64u != 0u) /* whole 64-half vectors per row */
        return 5;
      break;
    default:
      break;
    }

    /** Every ref the kind touches must fit its buffer for max_chunk rows,
     * the widest count forward() can pass (LOGITS reads row n_tokens - 1 of
     * in0 whatever d.m says). */
    if (nntr_htp_op_extent(&h, &d, h.max_chunk, &e))
      return 5;
    ref[0] = &d.in0;
    ref[1] = &d.in1;
    ref[2] = &d.in2;
    ref[3] = &d.out;
    bytes[0] = e.in0;
    bytes[1] = e.in1;
    bytes[2] = e.in2;
    bytes[3] = e.out;
    for (j = 0; j < 4u; ++j)
      if (((e.used >> j) & 1u) &&
          nntr_htp_check_ref(ref[j]->buf, ref[j]->offset, bytes[j], buf_size))
        return 5;
  }
  return 0;
}

#endif /* NNTR_HTP_COMMON_H */
