// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_htp_common.cpp
 * @date	23 April 2026
 * @brief	Shared implementations for HTP unit tests.
 *              Also defines main() with DSP session lifecycle so each HTP
 *              test binary links a single source of truth.
 * @see		https://github.com/nnstreamer/nntrainer
 * @bug		No known bugs except for NYI items
 */

#include "unittest_htp_common.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

#include <gtest/gtest.h>

#if defined(ENABLE_HTP)

#include <fp16.h>
#include <htp_interface.h>

using namespace nntrainer;

void permute_weight_to_fp16_tiles(const float *weight_f32,
                                  uint16_t *weight_fp16, int k, int n) {
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      int i0 = i / 32, i1 = i % 32;
      int j0 = j / 32, j1 = j % 32;
      int tile_idx = j0 * (k / 32) + i0;
      uint16_t *tile = weight_fp16 + tile_idx * 1024;
      tile[(i1 & ~1) * 32 + j1 * 2 + (i1 & 1)] =
        compute_fp32_to_fp16(weight_f32[i * n + j]);
    }
  }
}

void quantize_row_q4_0_ref(const float *src, block_q4_0 *dst, int k) {
  assert(k % 32 == 0);
  const int nb = k / 32;
  for (int b = 0; b < nb; ++b) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
      float v = std::fabs(src[b * 32 + j]);
      if (v > amax) amax = v;
    }
    float d = amax / 8.0f;
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    dst[b].d = compute_fp32_to_fp16(d);

    for (int j = 0; j < 16; ++j) {
      float x0 = src[b * 32 + j] * id;
      float x1 = src[b * 32 + 16 + j] * id;
      int q0 = (int)(x0 + 8.5f);
      int q1 = (int)(x1 + 8.5f);
      if (q0 < 0) q0 = 0;
      if (q0 > 15) q0 = 15;
      if (q1 < 0) q1 = 0;
      if (q1 > 15) q1 = 15;
      dst[b].qs[j] = (uint8_t)(q0 | (q1 << 4));
    }
  }
}

void dequantize_block_q4_0_ref(const block_q4_0 *blk, float *dst) {
  float d = compute_fp16_to_fp32(blk->d);
  for (int j = 0; j < 16; ++j) {
    int q0 = (blk->qs[j] & 0x0F);
    int q1 = (blk->qs[j] >> 4);
    dst[j] = ((float)q0 - 8.0f) * d;
    dst[16 + j] = ((float)q1 - 8.0f) * d;
  }
}

size_t repack_q4_0_to_x4x2(const block_q4_0 *src_q4_0, uint8_t *dst, int N,
                           int K) {
  assert(K % 256 == 0);
  const int groups_per_row = K / 32;
  const int quants_per_row = K / 2;
  const int n_superblocks = K / 256;
  const int scales_per_row = n_superblocks * 16;
  const size_t row_stride = (size_t)quants_per_row + scales_per_row;

  for (int row = 0; row < N; ++row) {
    uint8_t *dst_row = dst + row * row_stride;
    uint8_t *dst_q = dst_row;
    uint8_t *dst_s = dst_row + quants_per_row;
    const block_q4_0 *src_row = src_q4_0 + row * groups_per_row;

    for (int sb = 0; sb < n_superblocks; ++sb) {
      uint8_t *sb_q = dst_q + sb * 128;
      uint16_t *sb_s = (uint16_t *)(dst_s + sb * 16);

      for (int g = 0; g < 8; ++g) {
        const block_q4_0 *group = &src_row[sb * 8 + g];
        int sub_blk = g / 4;
        int sub_idx = g % 4;
        memcpy(sb_q + sub_blk * 64 + sub_idx * 16, group->qs, 16);
        sb_s[g] = group->d;
      }
    }
  }
  return row_stride;
}

void pack_q4_0_to_q4_0x4(const block_q4_0 *src, block_q4_0x4 *dst, int N,
                         int K) {
  const int nblocks = K / QK4_0;
  const uint64_t xor_mask = 0x8888888888888888ULL;

  for (int b = 0; b < N; b += 4) {
    for (int x = 0; x < nblocks; ++x) {
      block_q4_0 tmp[4];
      for (int i = 0; i < 4; ++i)
        tmp[i] = src[(b + i) * nblocks + x];

      for (int i = 0; i < 4; ++i)
        dst->d[i] = tmp[i].d;
      // blck_size_interleave=8: end = 32 * 2 / 8 = 8
      for (int i = 0; i < 8; ++i) {
        int src_id = i % 4;
        int src_off = (i / 4) * 8;
        int dst_off = i * 8;
        uint64_t elems;
        memcpy(&elems, &tmp[src_id].qs[src_off], sizeof(uint64_t));
        elems ^= xor_mask;
        memcpy(&dst->qs[dst_off], &elems, sizeof(uint64_t));
      }
      dst++;
    }
  }
}

ChanCtx &get_chan_ctx() {
  static ChanCtx ctx;
  if (ctx.chan != nullptr)
    return ctx;

  auto &htp = htp::HtpInterface::instance();
  if (!htp.alloc_shared_mem_buf || !htp.create_htp_message_channel)
    return ctx;

  ctx.size = 4096;
  int err = htp.alloc_shared_mem_buf(&ctx.chan, &ctx.chan_fd, ctx.size);
  if (err != 0) {
    ctx.chan = nullptr;
    return ctx;
  }

  err = htp.create_htp_message_channel(ctx.chan_fd, (unsigned int)ctx.size);
  if (err != 0) {
    htp.free_shared_mem_buf(ctx.chan, ctx.chan_fd, ctx.size);
    ctx.chan = nullptr;
    ctx.chan_fd = -1;
    ctx.size = 0;
  }
  return ctx;
}

#endif // ENABLE_HTP

GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

#if defined(ENABLE_HTP) && ENABLE_HTP == 1
  auto &htp = nntrainer::htp::HtpInterface::instance();
  if (htp.open_dsp_session) {
    int err = htp.open_dsp_session(CDSP_DOMAIN_ID, 1);
    if (err != 0) {
      std::cerr << "Open DSP session failed" << std::endl;
      return 1;
    }
    htp.init_htp_backend();
  }
#endif

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

#if defined(ENABLE_HTP) && ENABLE_HTP == 1
  if (htp.close_dsp_session) {
    htp.close_dsp_session();
  }
#endif

  return result;
}
