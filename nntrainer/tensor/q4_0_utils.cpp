// SPDX-License-Identifier: Apache-2.0
/**
 * @file	q4_0_utils.cpp
 * @date	15 October 2025
 * @brief	This is Q4_0Utils class for utils for Q4_0 quantization format.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Maciej Nalewaj <m.nalewaj@samsung.com>
 * @bug		No known bugs
 */

#include <cassert>
#include <cmath>
#include <cstring>

#include "cpu_backend.h"
#include "fp16.h"
#include "int4_utils.h"
#include "nntrainer_error.h"
#include "q4_0_utils.h"
#include "util_func.h"

namespace nntrainer {

void Q4_0Utils::unpackOneBlockQ4_0x4(const block_q4_0x4 *in, block_q4_0 *dst) {
  unsigned int blck_size_interleave = 8;

  for (int i = 0; i < 4; i++) {
    dst[i].d = in->d[i];
  }

  const int end = QK4_0 * 2 / blck_size_interleave;
  const uint64_t xor_mask = 0x8888888888888888ULL;

  for (int i = 0; i < end; ++i) {
    int dst_id = i % 4;
    int dst_offset = (i / 4) * blck_size_interleave;
    int src_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in->qs[src_offset], sizeof(uint64_t));
    elems ^= xor_mask;
    memcpy(&dst[dst_id].qs[dst_offset], &elems, sizeof(uint64_t));
  }
}

void Q4_0Utils::unpackBlocksQ4_0x4(const block_q4_0x4 *__restrict src,
                                   size_t data_size, size_t nrow, size_t K,
                                   block_q4_0 *__restrict dst) {
  int interleave_block = 4;

  const block_q4_0x4 *src_ = src;
  block_q4_0 *dst_ = (block_q4_0 *)dst;
  block_q4_0 dst_tmp[4];
  int nblocks = K / QK4_0;

  assert(data_size == (nrow / 4) * nblocks * sizeof(block_q4_0x4));

  for (size_t b = 0; b < nrow; b += interleave_block) {
    for (int64_t x = 0; x < nblocks; x++) {
      unpackOneBlockQ4_0x4(src_++, dst_tmp);

      for (size_t i = 0; i < interleave_block; i++) {
        dst_[x + i * nblocks] = dst_tmp[i];
      }
    }
    dst_ += interleave_block * nblocks;
  }
}

void Q4_0Utils::dequantizeQ4_0x4(const void *q4_weight_repacked, int N, int K,
                                 float *dequantized_weights) {
  assert(K % QK4_0 == 0);
  assert(N % 4 == 0);
  size_t data_size = (K / QK4_0) * (N / 4) * sizeof(block_q4_0x4);
  std::vector<uint8_t> q4_weight_out(data_size);
  unpackBlocksQ4_0x4((block_q4_0x4 *)q4_weight_repacked, data_size, N, K,
                     (block_q4_0 *)q4_weight_out.data());

  nntrainer::dequantize_row_q4_0((const void *)q4_weight_out.data(),
                                 dequantized_weights, K * N);
}

void Q4_0Utils::unpackOneBlockQ4_0x8(const block_q4_0x8 *in, block_q4_0 *dst) {
  unsigned int blck_size_interleave = 8;

  for (int i = 0; i < 8; i++) {
    dst[i].d = in->d[i];
  }

  const int end = QK4_0 * 4 / blck_size_interleave;
  const uint64_t xor_mask = 0x8888888888888888ULL;

  for (int i = 0; i < end; ++i) {
    int dst_id = i % 8;
    int dst_offset = (i / 8) * blck_size_interleave;
    int src_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in->qs[src_offset], sizeof(uint64_t));
    elems ^= xor_mask;
    memcpy(&dst[dst_id].qs[dst_offset], &elems, sizeof(uint64_t));
  }
}

void Q4_0Utils::unpackBlocksQ4_0x8(const block_q4_0x8 *__restrict src,
                                   size_t data_size, size_t nrow, size_t K,
                                   block_q4_0 *__restrict dst) {
  int interleave_block = 8;

  const block_q4_0x8 *src_ = src;
  block_q4_0 *dst_ = (block_q4_0 *)dst;
  block_q4_0 dst_tmp[8];
  int nblocks = K / QK4_0;

  assert(data_size == (nrow / 8) * nblocks * sizeof(block_q4_0x8));

  for (size_t b = 0; b < nrow; b += interleave_block) {
    for (int64_t x = 0; x < nblocks; x++) {
      unpackOneBlockQ4_0x8(src_++, dst_tmp);

      for (size_t i = 0; i < interleave_block; i++) {
        dst_[x + i * nblocks] = dst_tmp[i];
      }
    }
    dst_ += interleave_block * nblocks;
  }
}

void Q4_0Utils::dequantizeQ4_0x8(const void *q4_weight_repacked, int N, int K,
                                 float *dequantized_weights) {
  assert(K % QK4_0 == 0);
  assert(N % 8 == 0);
  size_t data_size = (K / QK4_0) * (N / 8) * sizeof(block_q4_0x8);
  std::vector<uint8_t> q4_weight_out(data_size);
  unpackBlocksQ4_0x8((block_q4_0x8 *)q4_weight_repacked, data_size, N, K,
                     (block_q4_0 *)q4_weight_out.data());

  nntrainer::dequantize_row_q4_0((const void *)q4_weight_out.data(),
                                 dequantized_weights, K * N);
}

inline static void nntr_make_block_q4_0x4(const block_q4_0 *in,
                                          block_q4_0x4 *out) {
  constexpr size_t IN_CNT = 4;
  constexpr size_t HALF_SIZE = 8;

  for (int i = 0; i < IN_CNT; ++i) {
    out->d[i] = in[i].d;
  }

  for (int i = 0; i < IN_CNT; ++i) {
    memcpy(&out->qs[i * HALF_SIZE], &in[i].qs[0], HALF_SIZE);
  }
  for (int i = 0; i < IN_CNT; ++i) {
    memcpy(&out->qs[IN_CNT * HALF_SIZE + i * HALF_SIZE], &in[i].qs[8],
           HALF_SIZE);
  }
}

inline static void nntr_make_block_q4_0x8(const block_q4_0 *in,
                                          block_q4_0x8 *out) {
  constexpr size_t IN_CNT = 8;
  constexpr size_t HALF_SIZE = 8;

  for (int i = 0; i < IN_CNT; ++i) {
    out->d[i] = in[i].d;
  }

  for (int i = 0; i < IN_CNT; ++i) {
    memcpy(&out->qs[i * HALF_SIZE], &in[i].qs[0], HALF_SIZE);
  }
  for (int i = 0; i < IN_CNT; ++i) {
    memcpy(&out->qs[IN_CNT * HALF_SIZE + i * HALF_SIZE], &in[i].qs[8],
           HALF_SIZE);
  }
}

void Q4_0Utils::transformQ4_0x_FromInt4(size_t N, size_t K,
                                        const uint8_t *osv32_weights,
                                        const uint16_t *osv32_scales,
                                        size_t scale_group_size,
                                        int q4_0x_block_size, void *dst_q4_0x) {

  NNTR_THROW_IF((!(scale_group_size == 32 || scale_group_size == 64 ||
                   scale_group_size == 128)),
                std::invalid_argument)
    << "Scale group size must be 32/64/128";
  NNTR_THROW_IF(K % QK4_0 != 0, std::invalid_argument)
    << "K size must be divisable by QK4_0 (32)";
  NNTR_THROW_IF(N % 8 != 0, std::invalid_argument)
    << "N size must be divisable by 8";
  NNTR_THROW_IF((!(q4_0x_block_size == 4 || q4_0x_block_size == 8)),
                std::invalid_argument)
    << "q4_0x_block_size must be 4 or 8";

  static constexpr const size_t ROW_BLOCK_SIZE = 32;
  static constexpr const size_t COLUMN_BLOCK_SIZE = 2;

  uint8_t int4_weight[16];
  uint16_t scale;
  block_q4_0 dst_tmp[8];
  uint8_t *dst_ = reinterpret_cast<uint8_t *>(dst_q4_0x);

  // --- Layout ---
  const size_t rows_count_pad = align(N, ROW_BLOCK_SIZE);
  const size_t columns_count_pad = align(K, ROW_BLOCK_SIZE);
  const size_t column_blocks_count =
    columns_count_pad / COLUMN_BLOCK_SIZE; // COLUMN_BLOCK_SIZE == 2
  const size_t bytes_per_row_block_span = column_blocks_count * ROW_BLOCK_SIZE;

  for (size_t row_id = 0; row_id < N; row_id += q4_0x_block_size) {
    const size_t row_block_id = row_id / ROW_BLOCK_SIZE;
    size_t i_in_block = row_id % ROW_BLOCK_SIZE;
    for (int64_t column_idx = 0; column_idx < K; column_idx += QK4_0) {
      for (size_t i = 0; i < q4_0x_block_size; i++) {
        int row_idx = row_id + i;
        // Address the bytes for this row
        const size_t row_block_base =
          row_block_id * bytes_per_row_block_span + i_in_block + i;
        int index0 = row_block_base + (column_idx / 2) * ROW_BLOCK_SIZE;

        for (size_t column_block_id = 0; column_block_id < 16;
             ++column_block_id) {
          int4_weight[column_block_id] =
            osv32_weights[index0 + column_block_id * ROW_BLOCK_SIZE];
        }
        scale = osv32_scales[row_idx +
                             (column_idx / scale_group_size) * rows_count_pad];

        create_q4_0_weights(int4_weight, dst_tmp[i].qs);
        dst_tmp[i].d = scale;
      }
      // Repack Q4_0 data
      if (q4_0x_block_size == 4) {
        nntr_make_block_q4_0x4(dst_tmp, (block_q4_0x4 *)dst_);
      } else {
        nntr_make_block_q4_0x8(dst_tmp, (block_q4_0x8 *)dst_);
      }
      dst_ += q4_0x_block_size * sizeof(block_q4_0);
    }
  }
}

void Q4_0Utils::printBlockQ4_0(const block_q4_0 *block) {
  printf("Q4_0: ");
  for (int i = 0; i < 16; i++) {
    printf("%i %i ", block->qs[i] & 0x0F, (block->qs[i] >> 4) & 0x0F);
  }
  printf("| scale:%f\n", compute_fp16_to_fp32(block->d));
}

void Q4_0Utils::repackToX4x2_Q4_0(const block_q4_0 *src_q4_0,
                                   uint8_t *dst_x4x2, size_t N, size_t K,
                                   size_t *out_row_stride) {
  // x4x2 format: each row (one output neuron) stores K elements as:
  //   [packed_quants: K/2 bytes] [scale_blocks: (K/256) * 16 bytes]
  // Within packed_quants: 4 groups of 32 elements interleaved as x4x2 blocks
  //   Each 256-element super-block has 128 bytes of packed 4-bit values
  //   arranged as 4 groups x 2 sub-blocks (lo/hi nibbles)
  // Scale blocks: 8 FP16 scales per 256-element block

  assert(K % 256 == 0);
  const size_t n_groups_per_row = K / QK4_0;  // groups of 32 elements per row
  const size_t quants_per_row = K / 2;         // bytes of packed 4-bit quants
  const size_t n_superblocks_per_row = K / 256;
  const size_t scales_per_row = n_superblocks_per_row * 16;  // 16 bytes per scale block
  const size_t row_stride = quants_per_row + scales_per_row;

  *out_row_stride = row_stride;

  for (size_t row = 0; row < N; ++row) {
    uint8_t *dst_row = dst_x4x2 + row * row_stride;
    uint8_t *dst_quants = dst_row;
    uint8_t *dst_scales = dst_row + quants_per_row;

    // Source: each row has n_groups_per_row block_q4_0 blocks
    const block_q4_0 *src_row = src_q4_0 + row * n_groups_per_row;

    // Pack quants: interleave 4 groups into x4x2 sub-blocks (128 bytes per 256 elements)
    for (size_t sb = 0; sb < n_superblocks_per_row; ++sb) {
      // Each super-block has 8 groups of 32 elements
      // x4x2 layout: groups 0-3 share lo nibbles, groups 4-7 share hi nibbles
      // Pack all 8 groups' quants into 128 bytes
      uint8_t *sb_quants = dst_quants + sb * 128;
      uint16_t *sb_scales = (uint16_t *)(dst_scales + sb * 16);

      for (int g = 0; g < 8; ++g) {
        const block_q4_0 *group = &src_row[sb * 8 + g];

        // Copy 16 bytes of packed quants for this group
        // In x4x2, groups 0-3 go to sub-block 0, groups 4-7 to sub-block 1
        // Each sub-block has 4 groups x 16 bytes = 64 bytes
        int sub_blk = g / 4;
        int sub_idx = g % 4;
        std::memcpy(sb_quants + sub_blk * 64 + sub_idx * 16, group->qs,
                    QK4_0 / 2);

        // Copy scale (FP16, stored as uint16_t)
        sb_scales[g] = group->d;
      }
    }
  }
}

void Q4_0Utils::repackToX4x2_Q4_0x8(const block_q4_0x8 *src_x8,
                                     uint8_t *dst_x4x2, size_t N, size_t K,
                                     size_t *out_row_stride) {
  assert(K % 256 == 0);
  assert(N % 8 == 0);

  const size_t nblocks = K / QK4_0; // blocks per row
  const size_t quants_per_row = K / 2;
  const size_t n_superblocks_per_row = K / 256;
  const size_t scales_per_row = n_superblocks_per_row * 16;
  const size_t row_stride = quants_per_row + scales_per_row;

  *out_row_stride = row_stride;

  const uint64_t xor_mask = 0x8888888888888888ULL;

  // block_q4_0x8 data is stored as groups of 8 rows.
  // Within each group, blocks are stored sequentially:
  //   group0: [blk_x8(row0-7, col_blk0), blk_x8(row0-7, col_blk1), ...]
  //   group1: [blk_x8(row8-15, col_blk0), ...]
  const block_q4_0x8 *src = src_x8;

  for (size_t row_base = 0; row_base < N; row_base += 8) {
    for (size_t x = 0; x < nblocks; ++x) {
      const block_q4_0x8 *blk = src++;

      // Unpack this block_q4_0x8 into 8 rows' x4x2 data
      // block_q4_0x8 layout (with XOR applied by quantizer):
      //   d[8]: scales for 8 rows
      //   qs[0..63]:   first 8 bytes of each row's block (rows 0-7), XORed
      //   qs[64..127]: second 8 bytes of each row's block (rows 0-7), XORed

      size_t sb_idx = x / 8;
      size_t grp_in_sb = x % 8;
      size_t sub_blk = grp_in_sb / 4;
      size_t grp_in_sub = grp_in_sb % 4;
      size_t q_off = sb_idx * 128 + sub_blk * 64 + grp_in_sub * 16;
      size_t s_off = quants_per_row + sb_idx * 16 + grp_in_sb * 2;

      for (int r = 0; r < 8; ++r) {
        size_t row = row_base + r;
        if (row >= N) break;
        uint8_t *dst_row = dst_x4x2 + row * row_stride;

        // Reverse XOR and copy 16 quant bytes (two 8-byte halves)
        uint64_t lo, hi;
        std::memcpy(&lo, &blk->qs[r * 8], sizeof(uint64_t));
        std::memcpy(&hi, &blk->qs[64 + r * 8], sizeof(uint64_t));
        lo ^= xor_mask;
        hi ^= xor_mask;
        std::memcpy(dst_row + q_off, &lo, sizeof(uint64_t));
        std::memcpy(dst_row + q_off + 8, &hi, sizeof(uint64_t));

        // Copy scale (FP16)
        std::memcpy(dst_row + s_off, &blk->d[r], sizeof(uint16_t));
      }
    }
  }
}

void Q4_0Utils::repackToX4x2_Q8_0(const int8_t *src_q8_0,
                                    const uint16_t *scales,
                                    uint8_t *dst_x4x2, size_t N, size_t K,
                                    size_t *out_row_stride) {
  assert(K % 256 == 0);
  const size_t n_groups_per_row = K / 32;
  const size_t quants_per_row = K;  // 1 byte per element for Q8_0
  const size_t n_superblocks_per_row = K / 256;
  const size_t scales_per_row = n_superblocks_per_row * 16;
  const size_t row_stride = quants_per_row + scales_per_row;

  *out_row_stride = row_stride;

  for (size_t row = 0; row < N; ++row) {
    uint8_t *dst_row = dst_x4x2 + row * row_stride;
    int8_t *dst_quants = (int8_t *)dst_row;
    uint8_t *dst_scales_region = dst_row + quants_per_row;

    const int8_t *src_row_quants = src_q8_0 + row * K;
    const uint16_t *src_row_scales = scales + row * n_groups_per_row;

    // Copy quants directly (Q8_0 quants are stored as-is in x4x2)
    std::memcpy(dst_quants, src_row_quants, K);

    // Pack scales into x4x2 scale blocks (8 scales per 256-element block)
    for (size_t sb = 0; sb < n_superblocks_per_row; ++sb) {
      uint16_t *sb_scales = (uint16_t *)(dst_scales_region + sb * 16);
      for (int g = 0; g < 8; ++g) {
        sb_scales[g] = src_row_scales[sb * 8 + g];
      }
    }
  }
}

} // namespace nntrainer
