// SPDX-License-Identifier: Apache-2.0
/**
 * @file	graph_lowering.cpp
 * @date	19 August 2026
 * @brief	pack_weights(): copies/converts model-agnostic source weights
 *		into the WEIGHTS byte image at the offsets a lowering recipe
 *		(e.g. lower_qwen3()) already computed, plus the precomputed
 *		RoPE cos/sin table. No shape knowledge beyond HexModelConfig
 *		and HexWeightOffsets; walks whatever a lowering produced.
 *		int8 projections other than down_proj are written tiled32
 *		(nntr_htp_tile_off).
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "graph_lowering.h"
#include "nntr_htp_common.h"

#include <cmath>
#include <cstring>

namespace nntrainer::hexagon {

namespace {

/** @brief Convert n fp32 values to fp16 and write them at dst+off. */
void write_f16_vec(uint8_t *dst, uint32_t off, const float *src, uint64_t n) {
  uint16_t *out = reinterpret_cast<uint16_t *>(dst + off);
  for (uint64_t i = 0; i < n; ++i)
    out[i] = f32_to_f16_bits(src[i]);
}

/**
 * @brief Fill the RoPE table at dst+off: max_seq rows, each row
 *        [cos64||sin64] fp16, angle = p * theta^(-2*i/128).
 */
void write_rope_table(uint8_t *dst, uint32_t off, uint32_t max_seq,
                      float theta) {
  uint16_t *out = reinterpret_cast<uint16_t *>(dst + off);
  for (uint32_t p = 0; p < max_seq; ++p) {
    uint16_t *row = out + static_cast<uint64_t>(p) * 128u;
    for (uint32_t i = 0; i < 64u; ++i) {
      float exponent = -2.0f * static_cast<float>(i) / 128.0f;
      float angle = static_cast<float>(p) * powf(theta, exponent);
      row[i] = f32_to_f16_bits(cosf(angle));
      row[64u + i] = f32_to_f16_bits(sinf(angle));
    }
  }
}

/**
 * @brief Write an int8 [n][k] projection in the tiled32 WEIGHTS layout
 *        (nntr_htp_tile_off): same byte count as the row-major source,
 *        only the order changes. down_proj is the one projection that
 *        stays row-major (MATMUL_W8A16 reads rows).
 */
void write_tiled(uint8_t *dst, uint32_t off, const int8_t *src, uint32_t n,
                 uint32_t k) {
  nntr_htp_repack_tiled32(dst + off, reinterpret_cast<const uint8_t *>(src), n,
                          k);
}

} // namespace

void pack_weights(const HexLoweredGraph &g, const HexModelConfig &cfg,
                  const HexModelWeights &w, uint8_t *dst) {
  const uint64_t n_q = static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim;
  const uint64_t n_kv = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;

  write_tiled(dst, g.woff.embed, w.embed, cfg.vocab, cfg.hidden);
  std::memcpy(dst + g.woff.embed_scale, w.embed_s,
              static_cast<uint64_t>(cfg.vocab) * 4u);
  write_rope_table(dst, g.woff.rope_table, cfg.max_seq, cfg.rope_theta);
  write_f16_vec(dst, g.woff.final_norm, w.final_norm, cfg.hidden);

  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    const HexWeightOffsets::PerLayer &pl = g.woff.layers[l];
    const HexLayerWeights &lw = w.layers[l];

    const uint32_t n_q32 = static_cast<uint32_t>(n_q);
    const uint32_t n_kv32 = static_cast<uint32_t>(n_kv);

    write_tiled(dst, pl.wq, lw.wq, n_q32, cfg.hidden);
    std::memcpy(dst + pl.wq_s, lw.wq_s, n_q * 4u);
    write_tiled(dst, pl.wk, lw.wk, n_kv32, cfg.hidden);
    std::memcpy(dst + pl.wk_s, lw.wk_s, n_kv * 4u);
    write_tiled(dst, pl.wv, lw.wv, n_kv32, cfg.hidden);
    std::memcpy(dst + pl.wv_s, lw.wv_s, n_kv * 4u);
    write_tiled(dst, pl.wo, lw.wo, cfg.hidden, n_q32);
    std::memcpy(dst + pl.wo_s, lw.wo_s, static_cast<uint64_t>(cfg.hidden) * 4u);
    write_tiled(dst, pl.gate, lw.w_gate, cfg.ffn, cfg.hidden);
    std::memcpy(dst + pl.gate_s, lw.w_gate_s,
                static_cast<uint64_t>(cfg.ffn) * 4u);
    write_tiled(dst, pl.up, lw.w_up, cfg.ffn, cfg.hidden);
    std::memcpy(dst + pl.up_s, lw.w_up_s, static_cast<uint64_t>(cfg.ffn) * 4u);
    std::memcpy(dst + pl.down, lw.w_down,
                static_cast<uint64_t>(cfg.hidden) * cfg.ffn); /* row-major */
    std::memcpy(dst + pl.down_s, lw.w_down_s,
                static_cast<uint64_t>(cfg.hidden) * 4u);

    write_f16_vec(dst, pl.attn_norm, lw.attn_norm, cfg.hidden);
    write_f16_vec(dst, pl.ffn_norm, lw.ffn_norm, cfg.hidden);
    write_f16_vec(dst, pl.q_norm, lw.q_norm, cfg.head_dim);
    write_f16_vec(dst, pl.k_norm, lw.k_norm, cfg.head_dim);
  }
}

} // namespace nntrainer::hexagon
