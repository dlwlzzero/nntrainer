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

// ============================================================
// Q4_0 permuted weight matmul tests (HTP_OPS_MAT_MUL_PERMUTED_W4D16A32)
// ============================================================

/**
 * @brief Q4_0 quantization constants matching quants.h on the DSP side.
 *
 * my_block_q4_0 is a "super-block" of 8 groups x 32 elements = 256 elements.
 * Layout: 8 fp16 scales followed by 128 bytes of packed 4-bit quants.
 */
static constexpr int QK_K_VAL = 256;
static constexpr int QK4_0_VAL = 32;
static constexpr int GROUPS_PER_SUPER_BLOCK = 8;
static constexpr int SUPER_BLOCK_SCALES_SIZE = GROUPS_PER_SUPER_BLOCK * 2;
static constexpr int SUPER_BLOCK_QUANTS_SIZE =
  GROUPS_PER_SUPER_BLOCK * (QK4_0_VAL / 2);
static constexpr int SUPER_BLOCK_SIZE =
  SUPER_BLOCK_SCALES_SIZE + SUPER_BLOCK_QUANTS_SIZE;

/**
 * @brief Quantize a row of K float values into the pre-permuted Q4_0 layout
 *        (my_block_q4_0 super-blocks).
 *
 * Each super-block covers 256 elements (8 groups of 32).
 * Layout per super-block:
 *   - scales[0..7]: 8 x fp16 scale values (16 bytes)
 *   - quants[0..127]: 8 x 16 bytes of packed 4-bit quants (128 bytes)
 *
 * Within each group of 32 elements, the low 16 elements occupy the low nibble
 * and the high 16 elements occupy the high nibble of each byte, matching the
 * DSP dequantization LUT which maps {0..15} -> {-8..+7}.
 *
 * @param src       Input float values, length K
 * @param dst       Output buffer for my_block_q4_0 super-blocks
 * @param k         Number of elements (must be multiple of QK_K=256)
 */
static void quantize_to_permuted_q4_0(const float *src, uint8_t *dst, int k) {
  int n_super_blocks = k / QK_K_VAL;

  for (int sb = 0; sb < n_super_blocks; ++sb) {
    uint8_t *block = dst + sb * SUPER_BLOCK_SIZE;
    uint16_t *scales = reinterpret_cast<uint16_t *>(block);
    uint8_t *quants = block + SUPER_BLOCK_SCALES_SIZE;
    const float *block_src = src + sb * QK_K_VAL;

    for (int g = 0; g < GROUPS_PER_SUPER_BLOCK; ++g) {
      const float *group_src = block_src + g * QK4_0_VAL;

      // Find max absolute value in this group of 32
      float amax = 0.0f;
      for (int j = 0; j < QK4_0_VAL; ++j) {
        float av = std::fabs(group_src[j]);
        if (av > amax)
          amax = av;
      }

      // Q4_0 maps integer {0..15} to float {-8..+7} via LUT
      // So scale = amax / 7.0 (max positive representable value is 7)
      float scale = amax / 7.0f;
      float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

      scales[g] = compute_fp32_to_fp16(scale);

      // Pack 32 elements into 16 bytes: low 16 in low nibble, high 16 in high
      for (int j = 0; j < QK4_0_VAL / 2; ++j) {
        float v_lo = group_src[j];
        float v_hi = group_src[j + QK4_0_VAL / 2];

        // Quantize: round to nearest integer in [-8, 7] range, then add 8
        // to get unsigned [0, 15]
        int q_lo = static_cast<int>(std::round(v_lo * inv_scale)) + 8;
        int q_hi = static_cast<int>(std::round(v_hi * inv_scale)) + 8;

        q_lo = std::max(0, std::min(15, q_lo));
        q_hi = std::max(0, std::min(15, q_hi));

        quants[g * (QK4_0_VAL / 2) + j] =
          static_cast<uint8_t>((q_hi << 4) | q_lo);
      }
    }
  }
}

/**
 * @brief Dequantize a single Q4_0 super-block back to float for CPU reference.
 *
 * @param block     Pointer to my_block_q4_0 super-block bytes
 * @param dst       Output float array (256 elements)
 */
static void dequantize_permuted_q4_0(const uint8_t *block, float *dst) {
  const uint16_t *scales = reinterpret_cast<const uint16_t *>(block);
  const uint8_t *quants = block + SUPER_BLOCK_SCALES_SIZE;

  for (int g = 0; g < GROUPS_PER_SUPER_BLOCK; ++g) {
    float scale = compute_fp16_to_fp32(scales[g]);

    for (int j = 0; j < QK4_0_VAL / 2; ++j) {
      uint8_t packed = quants[g * (QK4_0_VAL / 2) + j];
      int q_lo = (packed & 0x0F) - 8;
      int q_hi = (packed >> 4) - 8;

      dst[g * QK4_0_VAL + j] = static_cast<float>(q_lo) * scale;
      dst[g * QK4_0_VAL + j + QK4_0_VAL / 2] =
        static_cast<float>(q_hi) * scale;
    }
  }
}

/**
 * @brief Run a single Q4_0 permuted-weight matmul test.
 *
 * The test:
 *   1. Generates random activation [M x K] and weight [N x K] (float)
 *   2. Quantizes weights into pre-permuted Q4_0 (my_block_q4_0) layout
 *   3. Dequantizes back to float for CPU reference computation
 *   4. Computes CPU reference: C[i,j] = sum_l A[i,l] * W_dequant[j,l]
 *   5. Calls htp_ops_mat_mul_af32_pwqk0_of32 on the DSP
 *   6. Compares DSP result against CPU reference using MSE
 *
 * @param M  Number of output rows (batch dimension)
 * @param K  Reduction dimension (must be multiple of 256)
 * @param N  Number of output columns (must be multiple of 32)
 */
static void run_mat_mul_af32_pwqk0_of32_test(const uint32_t M,
                                              const uint32_t K,
                                              const uint32_t N) {
  auto &htp = htp::HtpInterface::instance();

  ASSERT_NE(htp.htp_ops_mat_mul_af32_pwqk0_of32, nullptr)
    << "HTP library not loaded (htp_ops_mat_mul_af32_pwqk0_of32 is null)";

  auto handle = htp.get_global_handle();
  ASSERT_NE(handle, (uint64_t)0) << "DSP session not opened (handle == 0)";

  ASSERT_EQ(K % QK_K_VAL, 0u)
    << "K must be a multiple of QK_K (" << QK_K_VAL << ")";
  ASSERT_EQ(N % 32, 0u) << "N must be a multiple of 32";

  // Generate random test data
  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  // Weight in [N x K] layout (each of N rows has K elements)
  std::vector<float> weight =
    generate_random_vector<float, false>(N * K, -0.1f, 0.1f);

  // Quantize weights to permuted Q4_0 layout
  size_t n_super_blocks_per_row = K / QK_K_VAL;
  size_t weight_q4_size = N * n_super_blocks_per_row * SUPER_BLOCK_SIZE;
  std::vector<uint8_t> weight_q4(weight_q4_size);

  for (uint32_t row = 0; row < N; ++row) {
    quantize_to_permuted_q4_0(weight.data() + row * K,
                              weight_q4.data() +
                                row * n_super_blocks_per_row * SUPER_BLOCK_SIZE,
                              K);
  }

  // Dequantize back to float for CPU reference
  std::vector<float> weight_deq(N * K);
  for (uint32_t row = 0; row < N; ++row) {
    for (size_t sb = 0; sb < n_super_blocks_per_row; ++sb) {
      dequantize_permuted_q4_0(
        weight_q4.data() +
          (row * n_super_blocks_per_row + sb) * SUPER_BLOCK_SIZE,
        weight_deq.data() + row * K + sb * QK_K_VAL);
    }
  }

  // Compute CPU reference matching DSP precision:
  //   DSP converts activation fp32->fp16 and dequantizes weight to fp16
  //   before fp16 x fp16 matmul, so we simulate the same truncation here.
  std::vector<float> ref_dst(M * N, 0.0f);
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (uint32_t l = 0; l < K; ++l) {
        float a = compute_fp16_to_fp32(
          compute_fp32_to_fp16(activation[i * K + l]));
        float w = compute_fp16_to_fp32(
          compute_fp32_to_fp16(weight_deq[j * K + l]));
        sum += a * w;
      }
      ref_dst[i * N + j] = sum;
    }
  }

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
                                 weight_q4_size);
  ASSERT_EQ(err, 0) << "Failed to allocate weight buffer";

  // Copy data to shared memory
  memcpy(activation_ptr, activation.data(), M * K * sizeof(float));
  memcpy(weight_ptr, weight_q4.data(), weight_q4_size);

  // Run on DSP (wgt_dt = GGML_TYPE_Q4_0 = 2)
  static constexpr int GGML_TYPE_Q4_0_VAL = 2;
  err = htp.htp_ops_mat_mul_af32_pwqk0_of32(handle, output_fd, 0,
                                             activation_fd, 0, weight_fd, 0, M,
                                             K, N, GGML_TYPE_Q4_0_VAL);
  ASSERT_EQ(err, 0) << "htp_ops_mat_mul_af32_pwqk0_of32 failed";

  // Compare results
  std::vector<float> hmx_dst(M * N);
  memcpy(hmx_dst.data(), output_ptr, M * N * sizeof(float));

  float mse_err = mse<float>(hmx_dst.data(), ref_dst.data(), M * N);
  std::cout << "W4D16A32 GEMM (Q4_0): " << M << " x " << K << " x " << N
            << std::endl;
  std::cout << " - MSE (vs Q4_0-dequantized ref): " << mse_err << std::endl;

  // Higher tolerance than fp16 tests due to 4-bit quantization error
  EXPECT_IN_RANGE(mse_err, 0.0f, 0.1f);

  // Cleanup
  htp.free_shared_mem_buf(output_ptr, output_fd, M * N * sizeof(float));
  htp.free_shared_mem_buf(activation_ptr, activation_fd,
                          M * K * sizeof(float));
  htp.free_shared_mem_buf(weight_ptr, weight_fd, weight_q4_size);
}

#define DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(M, K, N)                   \
  TEST(nntrainer_htp_kernels, mat_mul_af32_pwqk0_of32_##M##_##K##_##N) {      \
    run_mat_mul_af32_pwqk0_of32_test(M, K, N);                                \
  }

// Test square GEMM dimensions (K == N, M > 1)
// Note: K must be multiple of 256 (QK_K), N must be multiple of 32
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 256, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 512, 512);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 1024, 1024);

// Test rectangular GEMM dimensions (K != N, M > 1)
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 256, 512);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 512, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 1024, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 256, 1024);

// Test GEMV case (M = 1)
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(1, 256, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(1, 512, 512);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(1, 1024, 1024);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(1, 256, 512);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(1, 512, 256);

// Test non-power-of-2 M dimensions
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(28, 256, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(68, 256, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(28, 512, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(68, 256, 512);

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
