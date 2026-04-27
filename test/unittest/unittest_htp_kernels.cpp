// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_htp_kernels.cpp
 * @date	27 February 2026
 * @brief	Unit tests for HTP backend via HexKL SDKL API
 * @see		https://github.com/nnstreamer/nntrainer
 * @bug		No known bugs except for NYI items
 */

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

#if defined(ENABLE_HTP)

#include <fp16.h>
#include <sdkl_interface.h>
#include <nntrainer_test_util.h>

using namespace nntrainer;

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif

/**
 * @brief Run a single SDKL FP32×FP16→FP32 matmul test with the given
 *        dimensions.
 *
 * The test:
 *   1. Generates random activation [M x K] (float) and weight [N x K] (float)
 *   2. Computes a mixed-precision CPU reference (fp32 activation x fp16 weight)
 *   3. Converts weights to FP16 row-major [N x K], then to WH layout via
 *      sdkl_cpu_rm_to_wh_f16_inplace
 *   4. Calls sdkl_npu_mm_f32f16_f32
 *   5. Compares the DSP result against the CPU reference using MSE
 *
 * @param M  Number of output rows (batch dimension)
 * @param K  Reduction dimension
 * @param N  Number of output columns
 */
static void run_sdkl_mm_f32f16_f32_test(const uint32_t M, const uint32_t K,
                                         const uint32_t N) {
  auto &sdkl = sdkl::SdklInterface::instance();

  ASSERT_TRUE(sdkl.is_available()) << "SDKL library not loaded";
  ASSERT_EQ(sdkl.ensure_initialized(CDSP_DOMAIN_ID), 0)
    << "SDKL initialization failed";

  // Generate random test data
  // Activation: [M x K], Weight: [N x K] (row = output neuron)
  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  std::vector<float> weight =
    generate_random_vector<float, false>(N * K, -0.1f, 0.1f);

  // Compute mixed-precision reference: fp32 activation x fp16 weight
  // C[i,j] = sum_l A[i,l] * W[j,l]  (W in [N x K])
  std::vector<float> ref_dst(M * N, 0.0f);
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (uint32_t l = 0; l < K; ++l) {
        float a = activation[i * K + l];
        float w =
          compute_fp16_to_fp32(compute_fp32_to_fp16(weight[j * K + l]));
        sum += a * w;
      }
      ref_dst[i * N + j] = sum;
    }
  }

  // Allocate weight in shared memory and convert to WH layout
  const size_t wt_size = N * K * sizeof(_FP16);
  void *wt_buf = nullptr;
  int err = sdkl.npu_alloc(wt_size, &wt_buf);
  ASSERT_EQ(err, 0) << "Failed to allocate weight buffer";
  ASSERT_NE(wt_buf, nullptr);

  // Convert FP32 weights to FP16 row-major [N x K]
  _FP16 *wt_fp16 = static_cast<_FP16 *>(wt_buf);
  for (uint32_t i = 0; i < N * K; ++i) {
    wt_fp16[i] = static_cast<_FP16>(weight[i]);
  }

  // Convert to WH layout in-place
  err = sdkl.cpu_rm_to_wh_f16_inplace(N, K, wt_fp16);
  ASSERT_EQ(err, 0) << "sdkl_cpu_rm_to_wh_f16_inplace failed";

  // Allocate output
  std::vector<float> dsp_dst(M * N, 0.0f);

  // Run matmul via SDKL
  err = sdkl.npu_mm_f32f16_f32(sdkl.domain, M, N, K, dsp_dst.data(),
                                activation.data(), wt_fp16);
  ASSERT_EQ(err, 0) << "sdkl_npu_mm_f32f16_f32 failed";

  // Compare results
  float mse_err = mse<float>(dsp_dst.data(), ref_dst.data(), M * N);
  std::cout << "SDKL F32xF16->F32 GEMM: " << M << " x " << K << " x " << N
            << std::endl;
  std::cout << " - MSE (vs mixed-precision ref): " << mse_err << std::endl;

  EXPECT_IN_RANGE(mse_err, 0.0f, 0.01f);

  // Cleanup
  sdkl.npu_free(wt_buf);
}

#define DECLARE_sdkl_mm_f32f16_f32_test(M, K, N)                              \
  TEST(nntrainer_htp_kernels, sdkl_mm_f32f16_f32_##M##_##K##_##N) {           \
    run_sdkl_mm_f32f16_f32_test(M, K, N);                                     \
  }

// Test square GEMM dimensions (K == N, M > 1)
DECLARE_sdkl_mm_f32f16_f32_test(32, 32, 32);
DECLARE_sdkl_mm_f32f16_f32_test(32, 256, 256);
DECLARE_sdkl_mm_f32f16_f32_test(32, 512, 512);
DECLARE_sdkl_mm_f32f16_f32_test(32, 1024, 1024);

// Test rectangular GEMM dimensions (K != N, M > 1)
DECLARE_sdkl_mm_f32f16_f32_test(32, 256, 512);
DECLARE_sdkl_mm_f32f16_f32_test(32, 512, 256);
DECLARE_sdkl_mm_f32f16_f32_test(32, 1024, 256);
DECLARE_sdkl_mm_f32f16_f32_test(32, 256, 1024);
DECLARE_sdkl_mm_f32f16_f32_test(32, 64, 512);
DECLARE_sdkl_mm_f32f16_f32_test(32, 512, 64);

// Test GEMV case (M = 1, K == N)
DECLARE_sdkl_mm_f32f16_f32_test(1, 32, 32);
DECLARE_sdkl_mm_f32f16_f32_test(1, 256, 256);
DECLARE_sdkl_mm_f32f16_f32_test(1, 512, 512);
DECLARE_sdkl_mm_f32f16_f32_test(1, 1024, 1024);

// Test GEMV case with rectangular dimensions (M = 1, K != N)
DECLARE_sdkl_mm_f32f16_f32_test(1, 256, 512);
DECLARE_sdkl_mm_f32f16_f32_test(1, 512, 256);
DECLARE_sdkl_mm_f32f16_f32_test(1, 1024, 64);
DECLARE_sdkl_mm_f32f16_f32_test(1, 64, 1024);

// Test non-power-of-2 M dimensions
DECLARE_sdkl_mm_f32f16_f32_test(28, 256, 256);
DECLARE_sdkl_mm_f32f16_f32_test(68, 256, 256);
DECLARE_sdkl_mm_f32f16_f32_test(28, 512, 256);
DECLARE_sdkl_mm_f32f16_f32_test(68, 256, 512);

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
  auto &sdkl = nntrainer::sdkl::SdklInterface::instance();
  if (!sdkl.is_available()) {
    std::cerr << "SDKL library not available" << std::endl;
    return 1;
  }
  if (sdkl.ensure_initialized(CDSP_DOMAIN_ID) != 0) {
    std::cerr << "SDKL initialization failed" << std::endl;
    return 1;
  }
#endif

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

#if defined(ENABLE_HTP) && ENABLE_HTP == 1
  if (sdkl.npu_finalize) {
    sdkl.npu_finalize(sdkl.domain);
    sdkl.initialized = false;
  }
#endif

  return result;
}
