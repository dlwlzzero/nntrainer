// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_htp_kernels.cpp
 * @date	27 February 2026
 * @brief	Unit tests for HTP (Hexagon Tensor Processor) kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @bug		No known bugs except for NYI items
 */

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

#if defined(ENABLE_HTP)

#include <fp16.h>
#include <htp_interface.h>
#include <nntrainer_test_util.h>
#include <q4_0_utils.h>

using namespace nntrainer;

#define CDSP_DOMAIN_ID 3

/**
 * @brief Permute fp16 weights into the HMX tile layout expected by
 *        hmx_mat_mul_af32_pwf16_of32.
 *
 * The layout groups weights into 32x32 tiles. Within each tile the
 * element at row i, column j is stored at:
 *   tile[(i & ~1) * 32 + j * 2 + (i & 1)]
 *
 * @param weight_f32  Original weights in row-major [K x N] layout (float)
 * @param weight_fp16 Output buffer in permuted fp16 tile layout (uint16_t)
 * @param k           Number of rows (reduction dimension)
 * @param n           Number of columns (output dimension)
 */
static void permute_weight_to_fp16_tiles(const float *weight_f32,
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

/**
 * @brief Run a single w16a32 matmul test with the given dimensions.
 *
 * The test:
 *   1. Generates random activation [M x K] (float) and weight [K x N] (float)
 *   2. Computes a mixed-precision CPU reference (fp32 activation x fp16 weight)
 *   3. Permutes the weights into the HMX fp16 tile layout
 *   4. Calls htp_ops_rpc_mat_mul_permuted_w16a32 on the DSP
 *   5. Compares the DSP result against the CPU reference using MSE
 *
 * @param M  Number of output rows (batch dimension)
 * @param K  Reduction dimension
 * @param N  Number of output columns
 */
static void run_mat_mul_af32_pwf16_of32_test(const uint32_t M, const uint32_t K,
                            const uint32_t N) {
  auto &htp = htp::HtpInterface::instance();

  ASSERT_NE(htp.htp_ops_mat_mul_af32_pwf16_of32, nullptr)
    << "HTP library not loaded";

  auto handle = htp.get_global_handle();
  ASSERT_NE(handle, (uint64_t)0)
    << "DSP session not opened (handle == 0)";

  // Generate random test data
  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  std::vector<float> weight =
    generate_random_vector<float, false>(N * K, -0.1f, 0.1f);

  // Compute mixed-precision reference: fp32 activation x fp16 weight
  // This matches what the DSP does (fp16 weight, fp32 activation, fp32 output)
  std::vector<float> ref_dst(M * N, 0.0f);
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (uint32_t l = 0; l < K; ++l) {
        float a = activation[i * K + l];
        float w = compute_fp16_to_fp32(compute_fp32_to_fp16(weight[l * N + j]));
        sum += a * w;
      }
      ref_dst[i * N + j] = sum;
    }
  }

  // Allocate shared memory for DSP
  float *output_ptr = nullptr;
  float *activation_ptr = nullptr;
  uint16_t *weight_ptr = nullptr;
  int output_fd, activation_fd, weight_fd;

  int err = htp.alloc_shared_mem_buf((void **)&output_ptr, &output_fd,
                                     M * N * sizeof(float));
  ASSERT_EQ(err, 0) << "Failed to allocate output buffer";

  err = htp.alloc_shared_mem_buf((void **)&activation_ptr, &activation_fd,
                                 M * K * sizeof(float));
  ASSERT_EQ(err, 0) << "Failed to allocate activation buffer";

  err = htp.alloc_shared_mem_buf((void **)&weight_ptr, &weight_fd,
                                 K * N * sizeof(uint16_t));
  ASSERT_EQ(err, 0) << "Failed to allocate weight buffer";

  // Copy activation data and permute weights
  memcpy(activation_ptr, activation.data(), M * K * sizeof(float));
  memset(weight_ptr, 0, K * N * sizeof(uint16_t));
  permute_weight_to_fp16_tiles(weight.data(), weight_ptr, K, N);

  // Run on DSP
  err = htp.htp_ops_mat_mul_af32_pwf16_of32(handle, output_fd, 0, activation_fd,
                                            0, weight_fd, 0, M, K, N);
  ASSERT_EQ(err, 0) << "htp_ops_mat_mul_permuted_w16a32 failed";

  // Compare results
  std::vector<float> hmx_dst(M * N);
  memcpy(hmx_dst.data(), output_ptr, M * N * sizeof(float));

  float mse_err = mse<float>(hmx_dst.data(), ref_dst.data(), M * N);
  std::cout << "W16A32 GEMM: " << M << " x " << K << " x " << N << std::endl;
  std::cout << " - MSE (vs mixed-precision ref): " << mse_err << std::endl;

  EXPECT_IN_RANGE(mse_err, 0.0f, 0.01f);

  // Cleanup
  htp.free_shared_mem_buf(output_ptr, output_fd, M * N * sizeof(float));
  htp.free_shared_mem_buf(activation_ptr, activation_fd,
                          M * K * sizeof(float));
  htp.free_shared_mem_buf(weight_ptr, weight_fd, K * N * sizeof(uint16_t));
}

#define DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(M, K, N)                                     \
  TEST(nntrainer_htp_kernels, mat_mul_af32_pwf16_of32_##M##_##K##_##N) {                 \
    run_mat_mul_af32_pwf16_of32_test(M, K, N);                                                  \
  }

// Test square GEMM dimensions (K == N, M > 1)
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 32, 32);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 256, 256);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 512, 512);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 1024, 1024);

// Test rectangular GEMM dimensions (K != N, M > 1)
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 256, 512);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 512, 256);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 1024, 256);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 256, 1024);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 64, 512);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 512, 64);

// Test GEMV case (M = 1, K == N)
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(1, 32, 32);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(1, 256, 256);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(1, 512, 512);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(1, 1024, 1024);

// Test GEMV case with rectangular dimensions (M = 1, K != N)
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(1, 256, 512);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(1, 512, 256);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(1, 1024, 64);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(1, 64, 1024);

// Test non-power-of-2 M dimensions
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(28, 256, 256);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(68, 256, 256);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(28, 512, 256);
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(68, 256, 512);

/**
 * @brief Run a single wf16a32 matmul test with the given dimensions.
 *
 * Unlike the w16a32 (permuted) test, weights are stored in standard
 * row-major fp16 layout without HMX tile permutation.
 *
 * The test:
 *   1. Generates random activation [M x K] (float) and weight [K x N] (float)
 *   2. Computes a mixed-precision CPU reference (fp32 activation x fp16 weight)
 *   3. Converts weights to row-major fp16 (no permutation)
 *   4. Calls htp_ops_mat_mul_af32_wf16_of32 on the DSP
 *   5. Compares the DSP result against the CPU reference using MSE
 *
 * @param M  Number of output rows (batch dimension)
 * @param K  Reduction dimension
 * @param N  Number of output columns
 */
static void run_mat_mul_af32_wf16_of32_test(const uint32_t M, const uint32_t K,
                             const uint32_t N) {
  auto &htp = htp::HtpInterface::instance();

  ASSERT_NE(htp.htp_ops_mat_mul_af32_wf16_of32, nullptr)
    << "HTP library not loaded";

  auto handle = htp.get_global_handle();
  ASSERT_NE(handle, (uint64_t)0) << "DSP session not opened (handle == 0)";

  // Generate random test data
  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  // weight is stored in [N x K] layout (row: output dim, col: reduction dim)
  // as expected by hmx_mat_mul_af32_wf16_of32
  std::vector<float> weight =
    generate_random_vector<float, false>(N * K, -0.1f, 0.1f);

  // Compute mixed-precision reference: fp32 activation x fp16 weight
  // C[i,j] = sum_l A[i,l] * W[j,l]  (W in [N x K]: weight[j * K + l])
  std::vector<float> ref_dst(M * N, 0.0f);
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (uint32_t l = 0; l < K; ++l) {
        float a = activation[i * K + l];
        float w = compute_fp16_to_fp32(compute_fp32_to_fp16(weight[j * K + l]));
        sum += a * w;
      }
      ref_dst[i * N + j] = sum;
    }
  }

  // Allocate shared memory for DSP
  float *output_ptr = nullptr;
  float *activation_ptr = nullptr;
  uint16_t *weight_ptr = nullptr;
  int output_fd, activation_fd, weight_fd;

  int err = htp.alloc_shared_mem_buf((void **)&output_ptr, &output_fd,
                                     M * N * sizeof(float));
  ASSERT_EQ(err, 0) << "Failed to allocate output buffer";

  err = htp.alloc_shared_mem_buf((void **)&activation_ptr, &activation_fd,
                                 M * K * sizeof(float));
  ASSERT_EQ(err, 0) << "Failed to allocate activation buffer";

  err = htp.alloc_shared_mem_buf((void **)&weight_ptr, &weight_fd,
                                 N * K * sizeof(uint16_t));
  ASSERT_EQ(err, 0) << "Failed to allocate weight buffer";

  // Copy activation and convert weights to row-major fp16 [N x K] (no permutation)
  memcpy(activation_ptr, activation.data(), M * K * sizeof(float));
  for (uint32_t i = 0; i < N * K; ++i) {
    weight_ptr[i] = compute_fp32_to_fp16(weight[i]);
  }

  // Run on DSP
  err = htp.htp_ops_mat_mul_af32_wf16_of32(handle, output_fd, 0, activation_fd,
                                           0, weight_fd, 0, M, K, N);
  ASSERT_EQ(err, 0) << "htp_ops_mat_mul_af32_wf16_of32 failed";

  // Compare results
  std::vector<float> hmx_dst(M * N);
  memcpy(hmx_dst.data(), output_ptr, M * N * sizeof(float));

  float mse_err = mse<float>(hmx_dst.data(), ref_dst.data(), M * N);
  std::cout << "WF16A32 GEMM: " << M << " x " << K << " x " << N << std::endl;
  std::cout << " - MSE (vs mixed-precision ref): " << mse_err << std::endl;

  EXPECT_IN_RANGE(mse_err, 0.0f, 0.01f);

  // Cleanup
  htp.free_shared_mem_buf(output_ptr, output_fd, M * N * sizeof(float));
  htp.free_shared_mem_buf(activation_ptr, activation_fd,
                          M * K * sizeof(float));
  htp.free_shared_mem_buf(weight_ptr, weight_fd, N * K * sizeof(uint16_t));
}

#define DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(M, K, N)                                    \
  TEST(nntrainer_htp_kernels, mat_mul_af32_wf16_of32_##M##_##K##_##N) {                \
    run_mat_mul_af32_wf16_of32_test(M, K, N);                                                  \
  }

// Test square GEMM dimensions (K == N, M > 1)
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 32, 32);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 256, 256);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 512, 512);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 1024, 1024);

// Test rectangular GEMM dimensions (K != N, M > 1)
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 256, 512);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 512, 256);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 1024, 256);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 256, 1024);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 64, 512);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 512, 64);

// Test GEMV case (M = 1, K == N)
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(1, 32, 32);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(1, 256, 256);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(1, 512, 512);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(1, 1024, 1024);

// Test GEMV case with rectangular dimensions (M = 1, K != N)
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(1, 256, 512);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(1, 512, 256);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(1, 1024, 64);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(1, 64, 1024);

// Test non-power-of-2 M dimensions
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(28, 256, 256);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(68, 256, 256);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(28, 512, 256);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(68, 256, 512);

/**
 * @brief Quantize a row of K floats into K/32 block_q4_0 blocks.
 *
 * Each block covers 32 elements: find absmax, compute scale = absmax/8,
 * then quantize each element to 4-bit unsigned int (0..15) with zero-point 8.
 * Two consecutive quants are packed into one byte (low nibble first).
 */
static void quantize_row_q4_0_ref(const float *src, block_q4_0 *dst, int k) {
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
      float x0 = src[b * 32 + j]      * id;
      float x1 = src[b * 32 + 16 + j] * id;
      int q0 = (int)(x0 + 8.5f);
      int q1 = (int)(x1 + 8.5f);
      if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
      if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
      dst[b].qs[j] = (uint8_t)(q0 | (q1 << 4));
    }
  }
}

/**
 * @brief Dequantize one block_q4_0 (32 elements) into float.
 */
static void dequantize_block_q4_0_ref(const block_q4_0 *blk, float *dst) {
  float d = compute_fp16_to_fp32(blk->d);
  for (int j = 0; j < 16; ++j) {
    int q0 = (blk->qs[j] & 0x0F);
    int q1 = (blk->qs[j] >> 4);
    dst[j]      = ((float)q0 - 8.0f) * d;
    dst[16 + j] = ((float)q1 - 8.0f) * d;
  }
}

/**
 * @brief Repack block_q4_0 data into x4x2 row-strided format for HMX.
 *
 * x4x2 row layout: [packed_quants (K/2 bytes) | scale_blocks]
 * Each 256-element super-block has 8 groups packed into 128 quant bytes
 * and 16 bytes of scales (8 x fp16).
 *
 * @param src_q4_0  Quantized blocks in block_q4_0 format [N * K/32 blocks]
 * @param dst       Output buffer in x4x2 format
 * @param N         Number of rows (output dimension)
 * @param K         Number of columns (reduction dimension), must be % 256 == 0
 * @return          Row stride in bytes
 */
static size_t repack_q4_0_to_x4x2(const block_q4_0 *src_q4_0, uint8_t *dst,
                                   int N, int K) {
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

/**
 * @brief Run a single Q4_0 matmul test (x4x2 format) with the given
 * dimensions.
 *
 * The test:
 *   1. Generates random weight [N x K] in float
 *   2. Quantizes each row to block_q4_0 format
 *   3. Repacks block_q4_0 into x4x2 row-strided layout for HMX
 *   4. Computes CPU reference: fp32 activation x dequant(Q4_0) weight
 *   5. Calls htp_ops_mat_mul_af32_pwqk0_of32 on the DSP
 *   6. Compares the DSP result against the CPU reference using MSE
 *
 * @param M  Number of output rows (batch dimension)
 * @param K  Reduction dimension (must be multiple of 256 for x4x2)
 * @param N  Number of output columns (must be multiple of 32)
 */
static void run_mat_mul_af32_pwqk0_of32_test(const uint32_t M,
                                              const uint32_t K,
                                              const uint32_t N) {
  auto &htp = htp::HtpInterface::instance();

  ASSERT_NE(htp.htp_ops_mat_mul_af32_pwqk0_of32, nullptr)
    << "HTP library not loaded";

  auto handle = htp.get_global_handle();
  ASSERT_NE(handle, (uint64_t)0) << "DSP session not opened (handle == 0)";

  ASSERT_EQ(K % 256, 0u) << "K must be multiple of 256 for x4x2 format";
  ASSERT_EQ(N % 32, 0u) << "N must be multiple of 32";

  // Generate random test data
  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  // Weight stored in [N x K] layout (row = output neuron)
  std::vector<float> weight_f32 =
    generate_random_vector<float, false>(N * K, -0.1f, 0.1f);

  // Quantize each row of weight to block_q4_0
  const int groups_per_row = K / 32;
  std::vector<block_q4_0> weight_q4(N * groups_per_row);
  for (uint32_t row = 0; row < N; ++row) {
    quantize_row_q4_0_ref(weight_f32.data() + row * K,
                          weight_q4.data() + row * groups_per_row, K);
  }

  // Dequantize for CPU reference
  std::vector<float> weight_deq(N * K);
  for (uint32_t row = 0; row < N; ++row) {
    for (int g = 0; g < groups_per_row; ++g) {
      dequantize_block_q4_0_ref(&weight_q4[row * groups_per_row + g],
                                weight_deq.data() + row * K + g * 32);
    }
  }

  // Compute reference: C[i,j] = sum_l A[i,l] * W_deq[j,l]
  std::vector<float> ref_dst(M * N, 0.0f);
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (uint32_t l = 0; l < K; ++l) {
        sum += activation[i * K + l] * weight_deq[j * K + l];
      }
      ref_dst[i * N + j] = sum;
    }
  }

  // Repack to x4x2 format
  const int quants_per_row = K / 2;
  const int n_superblocks = K / 256;
  const size_t row_stride = (size_t)quants_per_row + n_superblocks * 16;
  const size_t weight_x4x2_size = N * row_stride;
  std::vector<uint8_t> weight_x4x2(weight_x4x2_size, 0);
  size_t actual_stride =
    repack_q4_0_to_x4x2(weight_q4.data(), weight_x4x2.data(), N, K);
  ASSERT_EQ(actual_stride, row_stride);

  // Allocate shared memory for DSP
  float *output_ptr = nullptr;
  float *activation_ptr = nullptr;
  uint8_t *weight_ptr = nullptr;
  int output_fd, activation_fd, weight_fd;

  int err = htp.alloc_shared_mem_buf((void **)&output_ptr, &output_fd,
                                     M * N * sizeof(float));
  ASSERT_EQ(err, 0) << "Failed to allocate output buffer";

  err = htp.alloc_shared_mem_buf((void **)&activation_ptr, &activation_fd,
                                 M * K * sizeof(float));
  ASSERT_EQ(err, 0) << "Failed to allocate activation buffer";

  err = htp.alloc_shared_mem_buf((void **)&weight_ptr, &weight_fd,
                                 weight_x4x2_size);
  ASSERT_EQ(err, 0) << "Failed to allocate weight buffer";

  // Copy data to shared buffers
  memcpy(activation_ptr, activation.data(), M * K * sizeof(float));
  memcpy(weight_ptr, weight_x4x2.data(), weight_x4x2_size);

  // Run on DSP (weight_type = 2 for GGML_TYPE_Q4_0)
  err = htp.htp_ops_mat_mul_af32_pwqk0_of32(handle, output_fd, 0,
                                              activation_fd, 0, weight_fd, 0,
                                              M, K, N, /*wgt_dt=*/2);
  ASSERT_EQ(err, 0) << "htp_ops_mat_mul_af32_pwqk0_of32 failed";

  // Compare results
  std::vector<float> hmx_dst(M * N);
  memcpy(hmx_dst.data(), output_ptr, M * N * sizeof(float));

  float mse_err = mse<float>(hmx_dst.data(), ref_dst.data(), M * N);
  std::cout << "Q4_0 x4x2 GEMM: " << M << " x " << K << " x " << N
            << std::endl;
  std::cout << " - MSE (vs Q4_0 dequant ref): " << mse_err << std::endl;

  // Q4_0 has higher quantization error than fp16, allow larger threshold
  EXPECT_IN_RANGE(mse_err, 0.0f, 0.05f);

  // Cleanup
  htp.free_shared_mem_buf(output_ptr, output_fd, M * N * sizeof(float));
  htp.free_shared_mem_buf(activation_ptr, activation_fd,
                          M * K * sizeof(float));
  htp.free_shared_mem_buf(weight_ptr, weight_fd, weight_x4x2_size);
}

#define DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(M, K, N)                   \
  TEST(nntrainer_htp_kernels, mat_mul_af32_pwqk0_of32_##M##_##K##_##N) {      \
    run_mat_mul_af32_pwqk0_of32_test(M, K, N);                                \
  }

// Test square GEMM (K == N, K multiple of 256)
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 256, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 512, 512);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 1024, 1024);

// Test rectangular GEMM
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 256, 512);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 512, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 1024, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 256, 1024);

// Test GEMV (M = 1)
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(1, 256, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(1, 512, 512);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(1, 1024, 1024);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(1, 256, 512);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(1, 512, 256);

// Test non-power-of-2 M dimensions
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(28, 256, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(68, 256, 512);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(28, 512, 256);

// Test large prefill-like dimensions (triggers out-stationary path)
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(128, 4096, 2048);

#else

TEST(nntrainer_htp_kernels, htp_not_enabled) {
  GTEST_SKIP() << "HTP is not enabled";
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
