#pragma once

#include <stdint.h>

#define QK_K 256  // super-block size

#define QK_0 32 // NOTE: This may be subject to change

#define QK4_0 32

typedef struct {
  __fp16  scale;
  uint8_t quants[QK4_0 / 2];
} __attribute__((packed)) block_q4_0;

typedef struct {
  __fp16  scales[8];
  uint8_t quants[8 * QK4_0 / 2];
} __attribute__((packed)) my_block_q4_0;

#define QK8_0 32

typedef struct {
  __fp16  scale;
  uint8_t quants[QK8_0];
} __attribute__((packed)) block_q8_0;

typedef struct {
  __fp16 scales[8];
  int8_t quants[8 * QK8_0];
} __attribute__((packed)) my_block_q8_0;

// x4x2 format constants: 4 groups x 2 sub-blocks (lo/hi nibbles)
// Row layout: [packed_quants | scale_blocks]
#define QK_Q4_0x4x2          256  // 4 groups × 2 sub-blocks × 32 elements
#define QK_Q8_0x4x2          256
#define HMX_X4X2_SCALES_PER_BLK  8
#define HMX_X4X2_DBLK_SIZE       16  // 8 × sizeof(__fp16) = 16 bytes per scale block

// Compute the row stride for x4x2 weight format (bytes per weight row of K elements)
// Q4_0/IQ4_NL: quants = K/2 bytes, scales = (K/QK_Q4_0x4x2) * HMX_X4X2_DBLK_SIZE bytes
// Q8_0:        quants = K bytes,   scales = (K/QK_Q8_0x4x2) * HMX_X4X2_DBLK_SIZE bytes
static inline size_t compute_x4x2_row_stride_q4_0(int k) {
  return (size_t)(k / 2) + (size_t)(k / QK_Q4_0x4x2) * HMX_X4X2_DBLK_SIZE;
}

static inline size_t compute_x4x2_row_stride_q8_0(int k) {
  return (size_t)k + (size_t)(k / QK_Q8_0x4x2) * HMX_X4X2_DBLK_SIZE;
}

enum ggml_type {
  GGML_TYPE_F32     = 0,
  GGML_TYPE_F16     = 1,
  GGML_TYPE_Q4_0    = 2,
  GGML_TYPE_Q4_1    = 3,
  // GGML_TYPE_Q4_2 = 4, support has been removed
  // GGML_TYPE_Q4_3 = 5, support has been removed
  GGML_TYPE_Q5_0    = 6,
  GGML_TYPE_Q5_1    = 7,
  GGML_TYPE_Q8_0    = 8,
  GGML_TYPE_Q8_1    = 9,
  GGML_TYPE_Q2_K    = 10,
  GGML_TYPE_Q3_K    = 11,
  GGML_TYPE_Q4_K    = 12,
  GGML_TYPE_Q5_K    = 13,
  GGML_TYPE_Q6_K    = 14,
  GGML_TYPE_Q8_K    = 15,
  GGML_TYPE_IQ2_XXS = 16,
  GGML_TYPE_IQ2_XS  = 17,
  GGML_TYPE_IQ3_XXS = 18,
  GGML_TYPE_IQ1_S   = 19,
  GGML_TYPE_IQ4_NL  = 20,
  GGML_TYPE_IQ3_S   = 21,
  GGML_TYPE_IQ2_S   = 22,
  GGML_TYPE_IQ4_XS  = 23,
  GGML_TYPE_I8      = 24,
  GGML_TYPE_I16     = 25,
  GGML_TYPE_I32     = 26,
  GGML_TYPE_I64     = 27,
  GGML_TYPE_F64     = 28,
  GGML_TYPE_IQ1_M   = 29,
  GGML_TYPE_BF16    = 30,
  // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
  // GGML_TYPE_Q4_0_4_8 = 32,
  // GGML_TYPE_Q4_0_8_8 = 33,
  GGML_TYPE_TQ1_0   = 34,
  GGML_TYPE_TQ2_0   = 35,
  // GGML_TYPE_IQ4_NL_4_4 = 36,
  // GGML_TYPE_IQ4_NL_4_8 = 37,
  // GGML_TYPE_IQ4_NL_8_8 = 38,
  GGML_TYPE_COUNT   = 39,
};

// Unified helper to compute x4x2 row stride by weight type
static inline size_t compute_x4x2_row_stride(int k, enum ggml_type weight_type) {
  switch (weight_type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_IQ4_NL:
      return compute_x4x2_row_stride_q4_0(k);
    case GGML_TYPE_Q8_0:
      return compute_x4x2_row_stride_q8_0(k);
    default:
      return 0;
  }
}
