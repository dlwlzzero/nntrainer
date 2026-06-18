// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_htp_quantizer.cpp
 * @date	23 April 2026
 * @brief	Unit tests for the ARM-side Q4_0x4 -> x4x2 repacker.
 *              Verifies it produces byte-identical output to the reference
 *              q4_0 -> x4x2 path and that the DSP consumes it correctly.
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

#include "unittest_htp_common.h"

using namespace nntrainer;

static void run_repack_q4_0x4_to_x4x2_test(const uint32_t M, const uint32_t K,
                                            const uint32_t N) {
  auto &htp = htp::HtpInterface::instance();
  ASSERT_NE(htp.htp_ops_mat_mul_af32_pwqk0_of32, nullptr);
  auto handle = htp.get_global_handle();
  ASSERT_NE(handle, (uint64_t)0);
  ASSERT_EQ(K % 256, 0u);
  ASSERT_EQ(N % 32, 0u);

  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  std::vector<float> weight_f32 =
    generate_random_vector<float, false>(N * K, -0.1f, 0.1f);

  const int nblocks = K / 32;
  std::vector<block_q4_0> weight_q4(N * nblocks);
  for (uint32_t row = 0; row < N; ++row)
    quantize_row_q4_0_ref(weight_f32.data() + row * K,
                          weight_q4.data() + row * nblocks, K);

  std::vector<block_q4_0x4> weight_x4((N / 4) * nblocks);
  pack_q4_0_to_q4_0x4(weight_q4.data(), weight_x4.data(), N, K);

  // Path A: block_q4_0 -> x4x2 (direct, known-good)
  size_t row_stride = (size_t)(K / 2) + (size_t)(K / 256) * 16;
  size_t wt_size = N * row_stride;
  std::vector<uint8_t> x4x2_A(wt_size, 0);
  repack_q4_0_to_x4x2(weight_q4.data(), x4x2_A.data(), N, K);

  // Path B: block_q4_0x4 -> x4x2 (under test)
  std::vector<uint8_t> x4x2_B(wt_size, 0);
  size_t actual_stride = 0;
  nntrainer::Q4_0Utils::repackToX4x2_Q4_0x4(weight_x4.data(), x4x2_B.data(), N,
                                             K, &actual_stride);
  ASSERT_EQ(actual_stride, row_stride);

  EXPECT_EQ(x4x2_A, x4x2_B)
    << "repackToX4x2_Q4_0x4 output differs from repackToX4x2_Q4_0 output";

  // End-to-end sanity: DSP produces correct results via path B.
  std::vector<float> weight_deq(N * K);
  for (uint32_t row = 0; row < N; ++row)
    for (int g = 0; g < nblocks; ++g)
      dequantize_block_q4_0_ref(&weight_q4[row * nblocks + g],
                                weight_deq.data() + row * K + g * 32);

  std::vector<float> ref_dst(M * N, 0.0f);
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (uint32_t l = 0; l < K; ++l)
        sum += activation[i * K + l] * weight_deq[j * K + l];
      ref_dst[i * N + j] = sum;
    }

  float *output_ptr = nullptr;
  float *act_ptr = nullptr;
  uint8_t *wt_ptr = nullptr;
  int out_fd, act_fd, wt_fd;
  ASSERT_EQ(htp.alloc_shared_mem_buf((void **)&output_ptr, &out_fd,
                                      M * N * sizeof(float)),
            0);
  ASSERT_EQ(htp.alloc_shared_mem_buf((void **)&act_ptr, &act_fd,
                                      M * K * sizeof(float)),
            0);
  ASSERT_EQ(htp.alloc_shared_mem_buf((void **)&wt_ptr, &wt_fd, wt_size), 0);

  memcpy(act_ptr, activation.data(), M * K * sizeof(float));
  memcpy(wt_ptr, x4x2_B.data(), wt_size);

  int err = htp.htp_ops_mat_mul_af32_pwqk0_of32(handle, out_fd, 0, act_fd, 0,
                                                wt_fd, 0, M, K, N, 2);
  ASSERT_EQ(err, 0);

  std::vector<float> hmx_dst(M * N);
  memcpy(hmx_dst.data(), output_ptr, M * N * sizeof(float));

  float mse_err = mse<float>(hmx_dst.data(), ref_dst.data(), M * N);
  std::cout << "Q4_0x4->x4x2 GEMM: " << M << "x" << K << "x" << N
            << "  MSE=" << mse_err << std::endl;
  EXPECT_IN_RANGE(mse_err, 0.0f, 0.05f);

  htp.free_shared_mem_buf(output_ptr, out_fd, M * N * sizeof(float));
  htp.free_shared_mem_buf(act_ptr, act_fd, M * K * sizeof(float));
  htp.free_shared_mem_buf(wt_ptr, wt_fd, wt_size);
}

#define DECLARE_repack_q4_0x4_test(M, K, N)                                   \
  TEST(nntrainer_htp_quantizer, repack_q4_0x4_to_x4x2_##M##_##K##_##N) {      \
    run_repack_q4_0x4_to_x4x2_test(M, K, N);                                  \
  }

DECLARE_repack_q4_0x4_test(32, 256, 256);    // typical power-of-2
DECLARE_repack_q4_0x4_test(128, 4096, 2048); // large prefill-like

#else

TEST(nntrainer_htp_quantizer, htp_not_enabled) {
  GTEST_SKIP() << "HTP is not enabled";
}

#endif // ENABLE_HTP
