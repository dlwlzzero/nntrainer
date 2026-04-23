// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_htp_common.h
 * @date	23 April 2026
 * @brief	Shared utilities for HTP (Hexagon Tensor Processor) unit tests:
 *              session lifecycle main(), fp16 tile permutation, Q4_0
 *              quantization helpers, and chan (shared-memory message channel)
 *              context used by both FastRPC and chan-based tests.
 * @see		https://github.com/nnstreamer/nntrainer
 * @bug		No known bugs except for NYI items
 */

#pragma once

#if defined(ENABLE_HTP)

#include <cstddef>
#include <cstdint>
#include <ctime>

#include <q4_0_utils.h>

#define CDSP_DOMAIN_ID 3

static inline int64_t htp_now_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

/**
 * @brief Permute fp16 weights into the HMX tile layout expected by
 *        hmx_mat_mul_af32_pwf16_of32.
 *
 * The layout groups weights into 32x32 tiles. Within each tile the
 * element at row i, column j is stored at:
 *   tile[(i & ~1) * 32 + j * 2 + (i & 1)]
 */
void permute_weight_to_fp16_tiles(const float *weight_f32,
                                  uint16_t *weight_fp16, int k, int n);

/**
 * @brief Quantize a row of K floats into K/32 block_q4_0 blocks.
 */
void quantize_row_q4_0_ref(const float *src, block_q4_0 *dst, int k);

/**
 * @brief Dequantize one block_q4_0 (32 elements) back to float.
 */
void dequantize_block_q4_0_ref(const block_q4_0 *blk, float *dst);

/**
 * @brief Repack block_q4_0 data into x4x2 row-strided format for HMX.
 * Returns the row stride in bytes.
 */
size_t repack_q4_0_to_x4x2(const block_q4_0 *src_q4_0, uint8_t *dst, int N,
                           int K);

/**
 * @brief Simulate the ARM quantizer's packing: block_q4_0 -> block_q4_0x4
 * (with XOR, blck_size_interleave=8, matching nntr_make_block_q4_0x4 in NEON).
 */
void pack_q4_0_to_q4_0x4(const block_q4_0 *src, block_q4_0x4 *dst, int N,
                         int K);

/**
 * @brief Single chan (shared-memory message channel) context shared across
 *        all chan-based HTP tests.
 *
 * The DSP side supports only one active channel at a time (commu.c:
 * htp_ops_create_channel returns AEE_EALREADY on second call), so tests
 * reuse this one. No destroy path: close_dsp_session() in main() tears
 * down the DSP-side receiver thread at process exit.
 */
struct ChanCtx {
  void  *chan    = nullptr;
  int    chan_fd = -1;
  size_t size    = 0;
};

/**
 * @brief Lazily create the process-wide HTP message channel and return it.
 *        Returns a ChanCtx with chan==nullptr if creation failed, allowing
 *        callers to GTEST_SKIP().
 */
ChanCtx &get_chan_ctx();

#endif // ENABLE_HTP
