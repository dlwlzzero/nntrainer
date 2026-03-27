// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_hmx_kernels.cpp
 * @date	27 February 2026
 * @brief	Test setup for Q4_0 HMX kernels
 * @see		https://github.com/nnstreamer/nntrainer
 * @bug		No known bugs except for NYI items
 */

#include <cstring>
#include <gtest/gtest.h>
#include <utility>

#if defined(ENABLE_HTP) && ENABLE_HTP == 1

#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <cpu_backend.h>
#include <nntrainer_test_util.h>
#include <q4_0_utils.h>
#include <tensor.h>
#include <timer.h>

#include <htp_ops.h>
#include <remote.h>
#include <rpcmem.h>
#include <session.h>

using namespace nntrainer;

#define QK4_0 32

typedef struct {
  __fp16 scales[8];
  uint8_t quants[8 * QK4_0 / 2];
} __attribute__((packed)) my_block_q4_0;

static inline int align_up(size_t size, size_t align) {
  return (size + align - 1) / align * align;
}

static void quantize_to_q4_0_superblocks(const float *weight_f32,
                                         uint8_t *weight_q4, int n, int k) {
  const int n_super_blocks = (n * k) / 256;
  my_block_q4_0 *mw = (my_block_q4_0 *)weight_q4;

  for (int sb = 0; sb < n_super_blocks; ++sb) {
    int base_col = sb * 8;
    for (int g = 0; g < 8; ++g) {
      int col = base_col + g;
      if (col >= n) {
        mw[sb].scales[g] = (__fp16)0.0f;
        for (int qq = 0; qq < QK4_0 / 2; ++qq) {
          mw[sb].quants[g * (QK4_0 / 2) + qq] = 0;
        }
        continue;
      }

      // compute absolute max and sign-extreme value
      float amax = 0.0f;
      float maxv = 0.0f;
      for (int r = 0; r < k; ++r) {
        float v = weight_f32[r * n + col];
        if (amax < fabsf(v)) {
          amax = fabsf(v);
          maxv = v;
        }
      }

      // follow quantize_row_q4_0_ref logic: d = max / -8
      float d = maxv / -8.0f;
      float id = d ? 1.0f / d : 0.0f;
      mw[sb].scales[g] = (__fp16)d;

      uint8_t *qptr = &mw[sb].quants[g * (QK4_0 / 2)];
      // pack 32 values into 16 bytes: pairs (0..15) and (16..31)
      for (int j = 0; j < QK4_0 / 2; ++j) {
        float x0 = weight_f32[(0 + j) * n + col] * id;
        float x1 = weight_f32[(QK4_0 / 2 + j) * n + col] * id;

        int xi0 = (int)(x0 + 8.5f);
        int xi1 = (int)(x1 + 8.5f);
        if (xi0 < 0)
          xi0 = 0;
        if (xi0 > 15)
          xi0 = 15;
        if (xi1 < 0)
          xi1 = 0;
        if (xi1 > 15)
          xi1 = 15;

        qptr[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
      }
    }
  }
}

static void run_q4_0_test(const uint32_t M, const uint32_t K,
                          const uint32_t N) {
  nntrainer::init_backend();

  // Open DSP session
  auto handle = get_global_handle();
  if (handle == -1) {
    open_dsp_session(CDSP_DOMAIN_ID, 1);
    handle = get_global_handle();
  }

  ASSERT_NE(handle, -1) << "Failed to open DSP session";

  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  std::vector<float> weight =
    generate_random_vector<float, false>(N * K, -0.01f, 0.01f);

  std::vector<float> ref_dst(M * N, 0.0f);
  std::vector<float> hmx_q4_dst(M * N, 0.0f);

  const int n_super_blocks = (N * K) / 256;
  const size_t weight_q4_size = n_super_blocks * 144;

  // Generate result from SGEMM
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);

  // Allocate shared memory for HMX
  float *output_ptr;
  float *activation_ptr;
  uint8_t *weight_ptr;
  int output_fd, activation_fd, weight_fd;

  int err = alloc_shared_mem_buf((void **)&output_ptr, &output_fd,
                                 M * N * sizeof(float));
  ASSERT_EQ(err, 0) << "Failed to allocate output buffer";

  err = alloc_shared_mem_buf((void **)&activation_ptr, &activation_fd,
                             M * K * sizeof(float));
  ASSERT_EQ(err, 0) << "Failed to allocate activation buffer";

  err = alloc_shared_mem_buf((void **)&weight_ptr, &weight_fd, weight_q4_size);
  ASSERT_EQ(err, 0) << "Failed to allocate weight buffer";

  // Copy data to shared memory
  memcpy(activation_ptr, activation.data(), M * K * sizeof(float));
  quantize_to_q4_0_superblocks(weight.data(), weight_ptr, N, K);

  // Warmup & Calibration
  unsigned int run_count = 5;
  auto t_w1 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < run_count; ++i) {
    htp_ops_mat_mul_permuted_qk_0_d16a32(handle, output_fd, 0, activation_fd, 0,
                                         weight_fd, 0, M, K, N, 2);
  }
  auto t_w2 = std::chrono::high_resolution_clock::now();
  double avg_time =
    std::chrono::duration<double>(t_w2 - t_w1).count() / run_count;

  if (avg_time > 0) {
    run_count = std::max(1u, (unsigned int)(0.5 / avg_time));
  } else {
    run_count = 100;
  }

  // HMX Q4_0 GEMM
  Timer timer1{};
  for (unsigned int i = 0; i < run_count; ++i) {
    htp_ops_mat_mul_permuted_qk_0_d16a32(handle, output_fd, 0, activation_fd, 0,
                                         weight_fd, 0, M, K, N, 2);
  }
  auto t2 = timer1.GetElapsedMilliseconds();

  // Copy result back
  memcpy(hmx_q4_dst.data(), output_ptr, M * N * sizeof(float));

  std::cout << "Q4_0 GEMM: " << M << " x " << K << " x " << N << std::endl;
  std::cout << " - HMX time: " << t2 / (run_count * 1.0f) << " ms" << std::endl;

  float mse_err = mse<float>(ref_dst.data(), hmx_q4_dst.data(), M * N);
  std::cout << " - MSE (vs FP32): " << mse_err << std::endl;

  // Q4_0 quantization is lossy, expect some error
  EXPECT_IN_RANGE(mse_err, 0.0f, 0.1f);

  // Free shared memory
  free_shared_mem_buf(output_ptr, output_fd, M * N * sizeof(float));
  free_shared_mem_buf(activation_ptr, activation_fd, M * K * sizeof(float));
  free_shared_mem_buf(weight_ptr, weight_fd, weight_q4_size);
}

#define DECLARE_q4_0_test_M_K_N(M, K, N)                                       \
  TEST(nntrainer_hmx_kernels, q4_0_test_##M##_##K##_##N) {                     \
    run_q4_0_test(M, K, N);                                                    \
  }

// Test various matrix dimensions
DECLARE_q4_0_test_M_K_N(32, 32, 32);
DECLARE_q4_0_test_M_K_N(68, 256, 256);
DECLARE_q4_0_test_M_K_N(68, 512, 512);
DECLARE_q4_0_test_M_K_N(68, 1024, 1024);

DECLARE_q4_0_test_M_K_N(28, 256, 256);
DECLARE_q4_0_test_M_K_N(28, 512, 512);

// Test GEMV case (M = 1)
TEST(nntrainer_hmx_kernels, q4_0_gemv_test) {
  nntrainer::init_backend();

  auto handle = get_global_handle();
  if (handle == -1) {
    open_dsp_session(CDSP_DOMAIN_ID, 1);
    handle = get_global_handle();
  }

  ASSERT_NE(handle, -1) << "Failed to open DSP session";

  const uint32_t M = 1;
  const uint32_t K = 512;
  const uint32_t N = 512;

  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  std::vector<float> weight =
    generate_random_vector<float, false>(N * K, -0.01f, 0.01f);

  std::vector<float> ref_dst(M * N, 0.0f);
  std::vector<float> hmx_q4_dst(M * N, 0.0f);

  const int n_super_blocks = (N * K) / 256;
  const size_t weight_q4_size = n_super_blocks * 144;

  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);

  float *output_ptr;
  float *activation_ptr;
  uint8_t *weight_ptr;
  int output_fd, activation_fd, weight_fd;

  alloc_shared_mem_buf((void **)&output_ptr, &output_fd, M * N * sizeof(float));
  alloc_shared_mem_buf((void **)&activation_ptr, &activation_fd,
                       M * K * sizeof(float));
  alloc_shared_mem_buf((void **)&weight_ptr, &weight_fd, weight_q4_size);

  memcpy(activation_ptr, activation.data(), M * K * sizeof(float));
  quantize_to_q4_0_superblocks(weight.data(), weight_ptr, N, K);

  htp_ops_mat_mul_permuted_qk_0_d16a32(handle, output_fd, 0, activation_fd, 0,
                                       weight_fd, 0, M, K, N, 2);

  memcpy(hmx_q4_dst.data(), output_ptr, M * N * sizeof(float));

  std::cout << "Q4_0 GEMV: " << M << " x " << K << " x " << N << std::endl;

  float mse_err = mse<float>(ref_dst.data(), hmx_q4_dst.data(), M * N);
  std::cout << " - MSE (vs FP32): " << mse_err << std::endl;

  EXPECT_IN_RANGE(mse_err, 0.0f, 0.1f);

  free_shared_mem_buf(output_ptr, output_fd, M * N * sizeof(float));
  free_shared_mem_buf(activation_ptr, activation_fd, M * K * sizeof(float));
  free_shared_mem_buf(weight_ptr, weight_fd, weight_q4_size);
}

// Test quantization accuracy
TEST(nntrainer_hmx_kernels, q4_0_quantization_accuracy) {
  nntrainer::init_backend();

  auto handle = get_global_handle();
  if (handle == -1) {
    open_dsp_session(CDSP_DOMAIN_ID, 1);
    handle = get_global_handle();
  }

  ASSERT_NE(handle, -1) << "Failed to open DSP session";

  const uint32_t M = 1;
  const uint32_t K = 256;
  const uint32_t N = 256;

  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  std::vector<float> weight =
    generate_random_vector<float, false>(N * K, -0.01f, 0.01f);

  std::vector<float> ref_dst(M * N, 0.0f);
  std::vector<float> hmx_q4_dst(M * N, 0.0f);

  const int n_super_blocks = (N * K) / 256;
  const size_t weight_q4_size = n_super_blocks * 144;

  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);

  float *output_ptr;
  float *activation_ptr;
  uint8_t *weight_ptr;
  int output_fd, activation_fd, weight_fd;

  alloc_shared_mem_buf((void **)&output_ptr, &output_fd, M * N * sizeof(float));
  alloc_shared_mem_buf((void **)&activation_ptr, &activation_fd,
                       M * K * sizeof(float));
  alloc_shared_mem_buf((void **)&weight_ptr, &weight_fd, weight_q4_size);

  memcpy(activation_ptr, activation.data(), M * K * sizeof(float));
  quantize_to_q4_0_superblocks(weight.data(), weight_ptr, N, K);

  htp_ops_mat_mul_permuted_qk_0_d16a32(handle, output_fd, 0, activation_fd, 0,
                                       weight_fd, 0, M, K, N, 2);

  memcpy(hmx_q4_dst.data(), output_ptr, M * N * sizeof(float));

  float mse_err = mse<float>(ref_dst.data(), hmx_q4_dst.data(), M * N);

  std::cout << "Q4_0 Quantization Test: " << M << " x " << K << " x " << N
            << std::endl;
  std::cout << " - MSE: " << mse_err << std::endl;

  EXPECT_LT(mse_err, 0.1f);

  free_shared_mem_buf(output_ptr, output_fd, M * N * sizeof(float));
  free_shared_mem_buf(activation_ptr, activation_fd, M * K * sizeof(float));
  free_shared_mem_buf(weight_ptr, weight_fd, weight_q4_size);
}

#else

TEST(nntrainer_hmx_kernels, hmx_not_enabled) {
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
  close_dsp_session();
#endif

  return result;
}