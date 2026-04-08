// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_0_utils.h
 * @date	15 October 2025
 * @brief	This is Q4_0Utils class for utils for Q4_0 quantization format.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Maciej Nalewaj <m.nalewaj@samsung.com>
 * @bug		No known bugs
 */

#ifndef __NNTRAINER_Q4_0_UTILS_H__
#define __NNTRAINER_Q4_0_UTILS_H__

#include <algorithm>
#include <cstdint>
#include <vector>

#define QK4_0 32

template <int K> constexpr int QK_0() {
  if constexpr (K == 4) {
    return 32;
  }
  if constexpr (K == 8) {
    return 32;
  }
  return -1;
}

/**
 * @brief block_q4_0xN
 */
template <int K, int N> struct block {
  uint16_t d[N];                      // deltas for N qK_0 blocks
  int8_t qs[(QK_0<K>() * N * K) / 8]; // quants for N qK_0 blocks
};

using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;

/**
 * @brief block_q4_0
 */
typedef struct {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

namespace nntrainer {

/**
 * @class Q4_0Utils class
 * @brief Q4_0Utils class with helpers for Q4_0 format calculation, quantization
 * and dequantization methods.
 */
class Q4_0Utils {
public:
  /**
   * @brief     Unpack one Q4_0x4 block to 4 Q4_0 blocks
   * @param[in] in block_q4_0x4* input Q4_0x4 block
   * @param[out] dst block_q4_0* output vector of 4 Q4_0 blocks
   */
  static void unpackOneBlockQ4_0x4(const block_q4_0x4 *in, block_q4_0 *dst);

  /**
   * @brief     Unpack Q4_0x4 blocks data to Q4_0 format
   * @param[in] src block_q4_0x4* input data in Q4_0x4 blocks format
   * @param[in] data_size number of Q4_0x4 blocks * sizeof(block_q4_0x4)
   * @param[in] nrow number of rows
   * @param[in] K number of columns
   * @param[out] dst block_q4_0* output data in Q4_0 blocks format
   */
  static void unpackBlocksQ4_0x4(const block_q4_0x4 *__restrict src,
                                 size_t data_size, size_t nrow, size_t K,
                                 block_q4_0 *__restrict dst);

  /**
   * @brief     Dequantize weights in block_q4_0x4 format to matrix of floats
   * @param[in] q4_weight_repacked void * input data in format block_q4_0x4
   * @param[in] N number of rows
   * @param[in] K number of columns
   * @param[out] dequantized_weights float * dequantized weights matrix
   */
  static void dequantizeQ4_0x4(const void *q4_weight_repacked, int N, int K,
                               float *dequantized_weights);

  /**
   * @brief     Unpack one Q4_0x8 block to 8 Q4_0 blocks
   * @param[in] in block_q4_0x8* input Q4_0x8 block
   * @param[out] dst block_q4_0* output vector of 8 Q4_0 blocks
   */
  static void unpackOneBlockQ4_0x8(const block_q4_0x8 *in, block_q4_0 *dst);

  /**
   * @brief     Unpack Q4_0x8 blocks data to Q4_0 format
   * @param[in] src block_q4_0x8* input data in Q4_0x8 blocks format
   * @param[in] data_size number of Q4_0x8 blocks * sizeof(block_q4_0x8)
   * @param[in] nrow number of rows
   * @param[in] K number of columns
   * @param[out] dst block_q4_0* output data in Q4_0 blocks format
   */
  static void unpackBlocksQ4_0x8(const block_q4_0x8 *__restrict src,
                                 size_t data_size, size_t nrow, size_t K,
                                 block_q4_0 *__restrict dst);

  /**
   * @brief     Dequantize weights in block_q4_0x8 format to matrix of floats
   * @param[in] q4_weight_repacked void * input data in format block_q4_0x8
   * @param[in] N number of rows
   * @param[in] K number of columns
   * @param[out] dequantized_weights float * dequantized weights matrix
   */
  static void dequantizeQ4_0x8(const void *q4_weight_repacked, int N, int K,
                               float *dequantized_weights);

  /**
   * @brief Transforms data from in-memory layout osv32_isv2 to block_q4_0x8 or
   * block_q4_0x4 in-memory layout.
   * @param N number of rows
   * @param K number of columns
   * @param osv32_weights uint8_t* data of weights in osv32_isv2 layout
   * @param osv32_scales fp16* scales
   * @param scale_group_size group size (32 or 64 or 128)
   * @param q4_0x_block_size output q4_0x block size - number of rows (4 or 8)
   * @param dst_q4_0x void * output data in block_q4_0x8 or block_q4_0x4 layout
   * depending on q4_0x_block_size
   */
  static void transformQ4_0x_FromInt4(size_t N, size_t K,
                                      const uint8_t *osv32_weights,
                                      const uint16_t *osv32_scales,
                                      size_t scale_group_size,
                                      int q4_0x_block_size, void *dst_q4_0x);

  /**
   * @brief     Print the Q4_0 block data
   * @param[in] block Pointer to the Q4_0 block
   */
  static void printBlockQ4_0(const block_q4_0 *block);

  /**
   * @brief Repack Q4_0 weights from block_q4_0 format to x4x2 row-strided
   * format. x4x2 row layout: [packed_quants (K/2 bytes) | scale_blocks]
   * Each row represents one output neuron (N dimension). Scales are grouped
   * in blocks of 8 FP16 values (HMX_X4X2_DBLK_SIZE = 16 bytes).
   * @param[in] src_q4_0 source data in block_q4_0 format (N * K/32 blocks)
   * @param[in] N number of output rows
   * @param[in] K number of input columns (must be divisible by 256)
   * @param[out] dst_x4x2 output buffer in x4x2 row-strided format
   * @param[out] out_row_stride bytes per row in the output
   */
  static void repackToX4x2_Q4_0(const block_q4_0 *src_q4_0, uint8_t *dst_x4x2,
                                 size_t N, size_t K, size_t *out_row_stride);

  /**
   * @brief Repack Q4_0 weights from block_q4_0x4 interleaved format to x4x2
   * row-strided format. On ARM, the quantizer packs Q4_0 data into block_q4_0x4
   * (4 rows interleaved with XOR mask). This directly converts to x4x2.
   * @param[in] src_x4 source data in block_q4_0x4 format
   * @param[in] N number of output rows (must be divisible by 4)
   * @param[in] K number of input columns (must be divisible by 256)
   * @param[out] dst_x4x2 output buffer in x4x2 row-strided format
   * @param[out] out_row_stride bytes per row in the output
   */
  static void repackToX4x2_Q4_0x4(const block_q4_0x4 *src_x4,
                                   uint8_t *dst_x4x2, size_t N, size_t K,
                                   size_t *out_row_stride);

  /**
   * @brief Repack Q4_0 weights from block_q4_0x8 interleaved format to x4x2
   * row-strided format. On x86, the quantizer packs Q4_0 data into block_q4_0x8
   * (8 rows interleaved with XOR mask). This directly converts to x4x2.
   * @param[in] src_x8 source data in block_q4_0x8 format
   * @param[in] N number of output rows (must be divisible by 8)
   * @param[in] K number of input columns (must be divisible by 256)
   * @param[out] dst_x4x2 output buffer in x4x2 row-strided format
   * @param[out] out_row_stride bytes per row in the output
   */
  static void repackToX4x2_Q4_0x8(const block_q4_0x8 *src_x8,
                                   uint8_t *dst_x4x2, size_t N, size_t K,
                                   size_t *out_row_stride);

  /**
   * @brief Repack Q8_0 weights from block_q8_0-like format to x4x2
   * row-strided format. Similar to Q4_0 but quants are 8-bit.
   * @param[in] src_q8_0 source data in int8 quantized format
   * @param[in] scales source FP16 scales
   * @param[in] N number of output rows
   * @param[in] K number of input columns (must be divisible by 256)
   * @param[out] dst_x4x2 output buffer
   * @param[out] out_row_stride bytes per row
   */
  static void repackToX4x2_Q8_0(const int8_t *src_q8_0,
                                 const uint16_t *scales, uint8_t *dst_x4x2,
                                 size_t N, size_t K, size_t *out_row_stride);
};
} // namespace nntrainer

#endif // __NNTRAINER_INT4_UTILS_H__
