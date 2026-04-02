// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_htp_kernels.cpp
 * @date	27 February 2026
 * @brief	Unit tests for HTP (Hexagon Tensor Processor) kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @bug		No known bugs except for NYI items
 */

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
 *        hmx_mat_mul_permuted_w16a32.
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
static void run_w16a32_test(const uint32_t M, const uint32_t K,
                            const uint32_t N) {
  auto &htp = htp::HtpInterface::instance();

  ASSERT_NE(htp.open_dsp_session, nullptr) << "HTP library not loaded";

  auto handle = htp.get_global_handle();
  if (handle == 0) {
    htp.open_dsp_session(CDSP_DOMAIN_ID, 1);
    handle = htp.get_global_handle();
  }
  ASSERT_NE(handle, (uint64_t)0) << "Failed to open DSP session";

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
  err = htp.htp_ops_rpc_mat_mul_permuted_w16a32(output_fd, 0, activation_fd, 0,
                                                weight_fd, 0, M, K, N);
  ASSERT_EQ(err, 0) << "htp_ops_rpc_mat_mul_permuted_w16a32 failed";

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

#define DECLARE_w16a32_test_M_K_N(M, K, N)                                     \
  TEST(nntrainer_htp_kernels, w16a32_matmul_##M##_##K##_##N) {                 \
    run_w16a32_test(M, K, N);                                                  \
  }

// Test various GEMM dimensions (M > 1)
DECLARE_w16a32_test_M_K_N(32, 32, 32);
DECLARE_w16a32_test_M_K_N(32, 256, 256);
DECLARE_w16a32_test_M_K_N(32, 512, 512);
DECLARE_w16a32_test_M_K_N(32, 1024, 1024);

// Test GEMV case (M = 1)
DECLARE_w16a32_test_M_K_N(1, 32, 32);
DECLARE_w16a32_test_M_K_N(1, 256, 256);
DECLARE_w16a32_test_M_K_N(1, 512, 512);
DECLARE_w16a32_test_M_K_N(1, 1024, 1024);

// Test non-power-of-2 M dimensions
DECLARE_w16a32_test_M_K_N(28, 256, 256);
DECLARE_w16a32_test_M_K_N(68, 256, 256);

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

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

#if defined(ENABLE_HTP) && ENABLE_HTP == 1
  auto &htp = nntrainer::htp::HtpInterface::instance();
  if (htp.close_dsp_session) {
    htp.close_dsp_session();
  }
#endif

  return result;
}
