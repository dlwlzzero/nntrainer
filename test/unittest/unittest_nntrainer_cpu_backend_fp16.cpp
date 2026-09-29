// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_cpu_backend_fp16.cpp
 * @date	03 April 2025
 * @brief	This is unittest for cpu_backend standalone
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "../htp/host/attn_m1_cases.h"
#include "int4_utils.h"
#include "kleidiai_interface.h"
#include "m1_ops_det.h"
#include "nntrainer_test_util.h"
#include <arm_neon.h>
#include <cfloat>
#include <climits>
#include <cpu_backend.h>
#include <fallback_internal.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <numeric>
#include <random>
#include <thread_manager.h>
#include <tuple>
#include <vector>

#include <chrono>
#include <iostream>
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;

template <typename T>
static inline double find_max_diff(T *src, T *src2, int M, int N) {
  float max_diff = 0;
  double err_sum = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      max_diff = std::max(max_diff, std::abs(static_cast<float>(
                                      src[i * N + j] - src2[i * N + j])));
      err_sum += std::abs(static_cast<float>(src[i * N + j] - src2[i * N + j]));
    }
  }
  // std::cout << "err_sum : " << err_sum << std::endl;
  return max_diff;
}

#define QK4_0 32
/**
 * @brief q4_0 block
 *
 */
typedef struct {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0_testonly;

/**
 * @brief q8_K block
 *
 */
typedef struct {
  float d;                 // delta
  int8_t qs[256];          // quants
  int16_t bsums[256 / 16]; // sum of quants in groups of 16
} block_q8_K_testonly;

#define QK_K 256
typedef struct {
  uint8_t ql[QK_K / 2];     // quants, lower 4 bits
  uint8_t qh[QK_K / 4];     // quants, upper 2 bits
  int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
  uint16_t d;               // super-block scale
} block_q6_K_testonly;

template <typename T = float>
float compute_mse(const uint32_t M, const uint32_t N, std::vector<T> &ref_dst,
                  std::vector<T> &dst, bool print = false) {
  auto mean_squared_error = mse<T, T>(ref_dst.data(), dst.data(), M * N);
  auto cos_sim = cosine_similarity<T, T>(ref_dst.data(), dst.data(), M * N);
  auto max_differ = find_max_diff<T>(ref_dst.data(), dst.data(), M, N);

  auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
  auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);
  if (print) {
    std::cout << "[INFO]            MSE: " << mean_squared_error
              << ", COS_SIM: " << cos_sim << ", MAX_DIFFER: " << max_differ
              << ", SUM: " << sum << ", SUM_GT: " << sum_gt << std::endl;
  }
  return mean_squared_error;
}

float test_gemm_q4_0_fp16(const uint32_t M, const uint32_t K, const uint32_t N,
                          const float *weights, const _FP16 *activations,
                          std::vector<_FP16> &ref_dst, bool print = false) {
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  int64_t q4_0_block_size = 32;
  int64_t q4_0_num_blocks = (K * N) / q4_0_block_size;
  size_t q4_0_data_size = q4_0_type_size * N / q4_0_block_size;
  q4_0_data_size *= K;
  std::vector<char> q4_0_offline_qWeight = std::vector<char>(q4_0_data_size);

  char *q4_0_offline_qWeight_ptr = (char *)q4_0_offline_qWeight.data();
  nntrainer::quantize_q4_0(weights, (void *)q4_0_offline_qWeight_ptr, N, K,
                           nullptr);

  std::vector<char> q4_0_repacked_qWeight = std::vector<char>(q4_0_data_size);
  nntrainer::repack_q4_0(q4_0_repacked_qWeight.data(), q4_0_offline_qWeight_ptr,
                         q4_0_data_size, N, K);
  std::vector<_FP16> dst(M * N);
  auto t1 = high_resolution_clock::now();
  nntrainer::gemm_q4_0<_FP16>(M, N, K, activations, K,
                              (void *)q4_0_repacked_qWeight.data(), N,
                              dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q4_0: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  auto mean_squared_error = compute_mse<_FP16>(M, N, ref_dst, dst, print);

  return mean_squared_error;
}

float test_gemm_q6_K_fp16(const uint32_t M, const uint32_t K, const uint32_t N,
                          const float *weights, const _FP16 *activations,
                          std::vector<_FP16> &ref_dst, bool print = false) {
  int64_t q6_k_block_size = 256;
  int64_t q6_k_type_size = sizeof(block_q6_K_testonly);
  int64_t num_blocks = (K * N) / q6_k_block_size;
  size_t data_size = q6_k_type_size * N / q6_k_block_size;
  data_size *= K;
  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q6_K(weights, (void *)offline_qWeight_ptr, N, K, nullptr);

  std::vector<_FP16> dst(M * N);
  auto t1 = high_resolution_clock::now();
  nntrainer::gemm_q6_K<_FP16>(M, N, K, activations, K,
                              (void *)offline_qWeight_ptr, N, dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q6_K: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  auto mean_squared_error = compute_mse<_FP16>(M, N, ref_dst, dst, print);

  return mean_squared_error;
}

void run_quant_test_fp16(const uint32_t M, const uint32_t K, const uint32_t N,
                         float &q4_0_mse, float &q6_K_mse, bool print = false) {
  nntrainer::init_backend();

  if (print) {
    std::cout << "[INFO] Quantization Test (M:" << M << ", K:" << K
              << ", N:" << N << ")" << std::endl;
  }
  ///@note A(M, K) * W.T(N, K) = (M, N)
  ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

  ///@note q4_K GEMM is a Row-Major, transB GEMM
  std::vector<_FP16> activation = generate_random_vector<_FP16>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<_FP16> weight_fp16(N * K);
  nntrainer::scopy(N * K, weight.data(), 1, weight_fp16.data(), 1);
  std::vector<_FP16> ref_dst(M * N);

  // GROUND TRUTH TRANSB SGEMM for reference
  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < 20; ++tc) {
    nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                     weight_fp16.data(), K, 0.F, ref_dst.data(), N);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] hgemm :    " << dt.count() / 20 << " ns "
              << dt.count() / 20 / 1'000 << " us "
              << dt.count() / 20 / 1'000'000 << " ms " << std::endl;
  }
  q4_0_mse = test_gemm_q4_0_fp16(M, K, N, weight.data(), activation.data(),
                                 ref_dst, print);
  q6_K_mse = test_gemm_q6_K_fp16(M, K, N, weight.data(), activation.data(),
                                 ref_dst, print);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_256x1024x512) {
  const unsigned int M = 256;
  const unsigned int K = 1024;
  const unsigned int N = 512;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_457x3072x3072) {
  const unsigned int M = 457;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_458x3072x3072) {
  const unsigned int M = 458;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_459x3072x3072) {
  const unsigned int M = 459;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x3072x3072) {
  const unsigned int M = 1024;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x768x1024) {
  const unsigned int M = 1;
  const unsigned int K = 768;
  const unsigned int N = 1024;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x3072x3072) {
  const unsigned int M = 1;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

static void run_trigonometric_values_test(const unsigned int N,
                                          bool print = false) {
  const int TEST_CNT = 20;
  nanoseconds ref_mul_time = (nanoseconds)0;
  nanoseconds mul_time = (nanoseconds)0;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<_FP16> X = generate_random_vector<_FP16, false>(N);
    std::vector<float> X_ref = generate_random_vector<float, false>(N);
    std::vector<_FP16> Y = generate_random_vector<_FP16, false>(N);
    std::vector<float> Y_ref = generate_random_vector<float, false>(N);

    std::vector<_FP16> X2 = generate_random_vector<_FP16, false>(N);
    std::vector<float> X2_ref = generate_random_vector<float, false>(N);
    std::vector<_FP16> Y2 = generate_random_vector<_FP16, false>(N);
    std::vector<float> Y2_ref = generate_random_vector<float, false>(N);
    {
      // #### GROUND TRUTH ####
      auto t1 = high_resolution_clock::now();
      nntrainer::sine(N, X_ref.data(), Y_ref.data());
      nntrainer::cosine(N, X2_ref.data(), Y2_ref.data());
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_mul_time += dt;
      }
    }
    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      nntrainer::sine(N, X.data(), Y.data());
      nntrainer::cosine(N, X2.data(), Y2.data());
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        mul_time += dt;
      }
    }

    auto mean_squared_error = mse<float, _FP16>(Y_ref.data(), Y.data(), N);
    auto cos_sim = cosine_similarity<float, _FP16>(Y2_ref.data(), Y2.data(), N);

    ASSERT_LE(mean_squared_error, 1e-3);
    ASSERT_GE(cos_sim, 0.99);
  }

  if (print) {
    std::cout << "[INFO] trigonometric_values: TEST CNT: " << TEST_CNT
              << ", N: " << N
              << ", Average ref_time: " << ref_mul_time.count() / TEST_CNT
              << " ns, Average test_time: " << mul_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
}

template <typename T = uint32_t>
std::pair<T, size_t> most_frequent(const std::vector<T> &data) {
  // Range is fixed 0–7, so use a small fixed array for counting
  std::array<size_t, 8> counts{};
  counts.fill(0);

  for (T v : data) {
    counts[v]++;
  }

  T most_value = 0;
  size_t most_count = 0;
  for (uint32_t i = 0; i < counts.size(); ++i) {
    if (counts[i] > most_count) {
      most_count = counts[i];
      most_value = i;
    }
  }

  return {most_value, most_count};
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x3072x512_CMP) {
  const unsigned int M = 1;
  const unsigned int K = 3072;
  const unsigned int N = 512;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_768x768x768_CMP) {
  const unsigned int M = 768;
  const unsigned int K = 768;
  const unsigned int N = 768;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_512x768x2048_CMP) {
  const unsigned int M = 512;
  const unsigned int K = 768;
  const unsigned int N = 2048;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_3072x512x512_CMP) {
  const unsigned int M = 3072;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float q4_0_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test_fp16(M, K, N, q4_0_mse, q6_k_mse, false);
  ASSERT_LE(q4_0_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_0_mse);
}

std::tuple<float, uint32_t> test_gemm_qsi8d32p_qsi4c32p_unpacked(
  const uint32_t M, const uint32_t K, const uint32_t N, const float *weights,
  const float *activations, std::vector<float> &ref_dst, bool transB = true,
  bool print = false) {
  // Step1. Set qsi8d32p_qsi4c32p quant test components
  // For qs4c32 format with block size 32:
  // - Each block has: sizeof(uint16_t) (scale as fp16) + bl/2 bytes (4-bit
  // packed data)
  // - Number of blocks per row: K / bl
  const size_t bl = 32; // block length
  const size_t num_blocks_per_row = K / bl;
  const size_t bytes_per_block =
    sizeof(uint16_t) + bl / 2; // fp16 scale + packed 4-bit data
  const size_t rhs_native_size_qs4c32 =
    static_cast<size_t>(N) * num_blocks_per_row * bytes_per_block;

  uint8_t *rhs_native_mtx_qs4c32 = new uint8_t[rhs_native_size_qs4c32];
  std::memset(rhs_native_mtx_qs4c32, 0, rhs_native_size_qs4c32);

  // Step2. 4-bit Weight quantization with block size 32 (qsi4c32p format)
  nntrainer::nntr_quant_qs4c32_f32(N, K, bl, (void *)weights,
                                   (void *)rhs_native_mtx_qs4c32);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(static_cast<size_t>(M) * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  uint32_t opt_kernel_variant_idx =
    nntrainer::nntr_gemm_qsi8d32p_qsi4c32p_unpacked(
      M, N, K, (void *)activations, (void *)rhs_native_mtx_qs4c32, nullptr,
      dst.data(), transB); // scales are embedded in qs4c32 format
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] test_gemm_qsi8d32p_qsi4c32p_unpacked: " << dt.count()
              << " ns " << dt.count() / 1'000 << " us "
              << dt.count() / 1'000'000 << " ms " << std::endl;
  }

  // Step4. Compute quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);

  delete[] rhs_native_mtx_qs4c32;

  return {mean_squared_error, opt_kernel_variant_idx};
}

static uint32_t run_qsi8d32p_qsi4c32p_test_unpacked(
  const uint32_t M, const uint32_t K, const uint32_t N,
  float &qsi8d32p_qsi4c32p_mse, bool transB = true, bool print = false) {
  if (print) {
    std::cout << "[INFO] qsi8d32p_qsi4c32p Test (M:" << M << ", K:" << K
              << ", N:" << N << ")" << std::endl;
  }

  std::vector<float> activation =
    generate_random_vector<float>(static_cast<std::size_t>(M) * K);
  std::vector<float> weight =
    generate_random_vector<float>(static_cast<std::size_t>(N) * K);
  std::vector<float> ref_dst(static_cast<std::size_t>(M) * N);

  // GROUND TRUTH TRANSB SGEMM for reference
  auto t1 = high_resolution_clock::now();
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] sgemm :    " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }
  const auto [mse, opt_kernel_variant_idx] =
    test_gemm_qsi8d32p_qsi4c32p_unpacked(
      M, K, N, weight.data(), activation.data(), ref_dst, transB, print);

  qsi8d32p_qsi4c32p_mse = mse;

  return opt_kernel_variant_idx;
}

float test_gemm_qsi8d32p_qsi4c32p_packed(
  const uint32_t M, const uint32_t K, const uint32_t N, const float *weights,
  const float *activations, std::vector<float> &ref_dst,
  uint32_t opt_kernel_idx, bool transB = true, bool print = false) {
  // Step1. Set qsi8d32p_qsi4c32p quant test components using qs4c32 format
  // For qs4c32 format with block size 32:
  // - Each block has: sizeof(uint16_t) (scale as fp16) + bl/2 bytes (4-bit
  // packed data)
  // - Number of blocks per row: K / bl
  const size_t bl = 32; // block length
  const size_t num_blocks_per_row = K / bl;
  const size_t bytes_per_block =
    sizeof(uint16_t) + bl / 2; // fp16 scale + packed 4-bit data
  const size_t rhs_native_size_qs4c32 =
    static_cast<size_t>(N) * num_blocks_per_row * bytes_per_block;

  uint8_t *rhs_native_mtx_qs4c32 = new uint8_t[rhs_native_size_qs4c32];
  std::memset(rhs_native_mtx_qs4c32, 0, rhs_native_size_qs4c32);

  // Step2. 4-bit Weight quantization with block size 32 (qsi4c32 format with
  // embedded fp16 scales)
  nntrainer::nntr_quant_qs4c32_f32(N, K, bl, (void *)weights,
                                   (void *)rhs_native_mtx_qs4c32);

  // Step3. Offline weight packing
  size_t packed_weight_size =
    nntrainer::nntr_get_rhs_packed_size_qsi8d32p_qsi4c32p(N, K, opt_kernel_idx,
                                                          transB);
  uint8_t *packed_weight = new uint8_t[packed_weight_size];

  nntrainer::nntr_qsi8d32p_qsi4c32p_rhs_pack(
    N, K, packed_weight, rhs_native_mtx_qs4c32, nullptr, opt_kernel_idx,
    transB); // scales are embedded in qs4c32 format

  // Step4. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(static_cast<size_t>(M) * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::nntr_gemm_qsi8d32p_qsi4c32p_packed(
    M, N, K, (void *)activations, (void *)packed_weight, dst.data(),
    opt_kernel_idx, transB);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] test_gemm_qsi8d32p_qsi4c32p_packed: " << dt.count()
              << " ns " << dt.count() / 1'000 << " us "
              << dt.count() / 1'000'000 << " ms " << std::endl;
  }

  // Step5. Compute quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);

  delete[] rhs_native_mtx_qs4c32;
  delete[] packed_weight;

  return mean_squared_error;
}

void run_qsi8d32p_qsi4c32p_test_packed(const uint32_t M, const uint32_t K,
                                       const uint32_t N,
                                       float &qsi8d32p_qsi4c32p_mse,
                                       uint32_t opt_kernel_idx,
                                       bool transB = true, bool print = false) {
  if (print) {
    std::cout << "[INFO] run_qsi8d32p_qsi4c32p_test_packed Test (M:" << M
              << ", K:" << K << ", N:" << N
              << ") with opt_kernel_idx : " << opt_kernel_idx << std::endl;
  }

  std::vector<float> activation =
    generate_random_vector<float>(static_cast<std::size_t>(M) * K);
  std::vector<float> weight =
    generate_random_vector<float>(static_cast<std::size_t>(N) * K);
  std::vector<float> ref_dst(static_cast<std::size_t>(M) * N);

  // GROUND TRUTH TRANSB SGEMM for reference
  auto t1 = high_resolution_clock::now();
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] sgemm :    " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }
  qsi8d32p_qsi4c32p_mse = test_gemm_qsi8d32p_qsi4c32p_packed(
    M, K, N, weight.data(), activation.data(), ref_dst, opt_kernel_idx, transB,
    print);
}

TEST(nntrainer_cpu_backend_standalone, qsi8d32p_qsi4c32p_1x3072x512_CMP) {
  const unsigned int M = 1;
  const unsigned int K = 3072;
  const unsigned int N = 512;
  float qsi8d32p_qsi4c32p_mse;
  float qsi8d32p_qsi4c32p_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qsi8d32p_qsi4c32p_test_unpacked(
      M, K, N, qsi8d32p_qsi4c32p_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qsi8d32p_qsi4c32p_test_packed(M, K, N, qsi8d32p_qsi4c32p_mse_packed,
                                    opt_idx_variant, true, false);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse, eps * M * K * N);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse_packed, eps * M * K * N);
}

TEST(nntrainer_cpu_backend_standalone, qsi8d32p_qsi4c32p_768x768x768_CMP) {
  const unsigned int M = 768;
  const unsigned int K = 768;
  const unsigned int N = 768;
  float qsi8d32p_qsi4c32p_mse;
  float qsi8d32p_qsi4c32p_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qsi8d32p_qsi4c32p_test_unpacked(
      M, K, N, qsi8d32p_qsi4c32p_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qsi8d32p_qsi4c32p_test_packed(M, K, N, qsi8d32p_qsi4c32p_mse_packed,
                                    opt_idx_variant, true, false);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse, eps * M * K * N);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse_packed, eps * M * K * N);
}

TEST(nntrainer_cpu_backend_standalone, qsi8d32p_qsi4c32p_512x768x2048_CMP) {
  const unsigned int M = 512;
  const unsigned int K = 768;
  const unsigned int N = 2048;
  float qsi8d32p_qsi4c32p_mse;
  float qsi8d32p_qsi4c32p_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qsi8d32p_qsi4c32p_test_unpacked(
      M, K, N, qsi8d32p_qsi4c32p_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qsi8d32p_qsi4c32p_test_packed(M, K, N, qsi8d32p_qsi4c32p_mse_packed,
                                    opt_idx_variant, true, false);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse, eps * M * K * N);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse_packed, eps * M * K * N);
}

TEST(nntrainer_cpu_backend_standalone, qsi8d32p_qsi4c32p_3072x512x512_CMP) {
  const unsigned int M = 3072;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float qsi8d32p_qsi4c32p_mse;
  float qsi8d32p_qsi4c32p_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qsi8d32p_qsi4c32p_test_unpacked(
      M, K, N, qsi8d32p_qsi4c32p_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qsi8d32p_qsi4c32p_test_packed(M, K, N, qsi8d32p_qsi4c32p_mse_packed,
                                    opt_idx_variant, true, false);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse, eps * M * K * N);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse_packed, eps * M * K * N);
}

TEST(nntrainer_cpu_backend_standalone, qsi8d32p_qsi4c32p_3072x102x1024_CMP) {
  const unsigned int M = 3072;
  const unsigned int K = 1024;
  const unsigned int N = 1024;
  float qsi8d32p_qsi4c32p_mse;
  float qsi8d32p_qsi4c32p_mse_packed;
  constexpr float eps = 1e-5;
  const uint32_t TC = 20;
  std::vector<uint32_t> opt_idx_variant_candidates;
  uint32_t opt_idx_variant = 0;
  for (uint32_t tc = 0; tc < TC; ++tc) {
    opt_idx_variant = run_qsi8d32p_qsi4c32p_test_unpacked(
      M, K, N, qsi8d32p_qsi4c32p_mse, true, false);
    opt_idx_variant_candidates.push_back(opt_idx_variant);
  }
  auto result = most_frequent(opt_idx_variant_candidates);
  opt_idx_variant = result.first;

  run_qsi8d32p_qsi4c32p_test_packed(M, K, N, qsi8d32p_qsi4c32p_mse_packed,
                                    opt_idx_variant, true, false);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse, eps * M * K * N);
  ASSERT_LE(qsi8d32p_qsi4c32p_mse_packed, eps * M * K * N);
}

/**
 * @brief Test helper function for osv32_isv2 to qsi4c32p transform
 *
 * Tests the lossless transformation from OpenVINO osv32_isv2 format to
 * KleidiAI qsi4c32p packed format by:
 * 1. Generating random FP32 weights
 * 2. Quantizing to osv32_isv2 format using Int4Utils
 * 3. Transforming to qsi4c32p using nntr_kai_repack_osv32_to_qsi4c32p
 * 4. Running GEMM with packed weights
 * 5. Comparing against FP32 reference GEMM
 */
static void run_transform_osv32_to_qsi4c32p_test(const uint32_t K,
                                                 const uint32_t N,
                                                 uint32_t kernel_idx = 3,
                                                 bool print = false) {
  const uint32_t M = 3072;      // Batch size for GEMM test
  const size_t group_size = 32; // Fixed group size

  // Step 1: Generate random FP32 weights
  std::vector<float> weight_fp32 =
    generate_random_vector<float>(N * K, -1.0f, 1.0f);

  // Step 2: Quantize to osv32_isv2 format
  std::vector<uint8_t> osv32_weights;
  std::vector<uint16_t> osv32_scales;
  nntrainer::Int4Utils::quantizeAndRepack(weight_fp32.data(), N, K, group_size,
                                          osv32_weights, osv32_scales);

  // Step 3: Transform osv32_isv2 -> qsi4c32p packed
  size_t packed_size = 0;
  size_t expected_packed_size =
    nntr_kai_get_rhs_packed_size_qsi8d32p_qsi4c32p(N, K, kernel_idx, true);
  std::vector<uint8_t> qsi4c32p_packed(expected_packed_size);

  auto t0 = high_resolution_clock::now();
  nntr_kai_repack_osv32_to_qsi4c32p(N, K, osv32_weights.data(),
                                    osv32_scales.data(), qsi4c32p_packed.data(),
                                    packed_size, kernel_idx, true);
  auto t1 = high_resolution_clock::now();
  auto transform_time = duration_cast<microseconds>(t1 - t0);

  if (print) {
    std::cout << "[INFO] Transform time: " << transform_time.count() << " us"
              << std::endl;
    std::cout << "[INFO] Packed size: " << packed_size << " bytes" << std::endl;
  }

  // Step 4: Generate random FP32 activations
  std::vector<float> activations =
    generate_random_vector<float>(M * K, -1.0f, 1.0f);

  // Step 5: Run FP32 reference GEMM
  std::vector<float> ref_dst(M * N, 0.0f);
  nntrainer::sgemm(0, false, true, M, N, K, 1.0f, activations.data(), K,
                   weight_fp32.data(), K, 0.0f, ref_dst.data(), N);

  // Step 6: Run GEMM with transformed qsi4c32p weights
  std::vector<float> qsi4c32p_dst(M * N, 0.0f);
  nntrainer::nntr_gemm_qsi8d32p_qsi4c32p_packed(
    M, N, K, (void *)activations.data(), (void *)qsi4c32p_packed.data(),
    qsi4c32p_dst.data(), kernel_idx, true);

  // Step 7: Compute MSE and cosine similarity
  float mean_squared_error = compute_mse(M, N, ref_dst, qsi4c32p_dst, print);
  float cos_sim =
    cosine_similarity<float, float>(ref_dst.data(), qsi4c32p_dst.data(), M * N);

  if (print) {
    std::cout << "[INFO] MSE: " << mean_squared_error
              << ", Cosine Sim: " << cos_sim << std::endl;
  }

  // Step 8: Assert quality metrics
  // For 4-bit quantization, expect some quantization noise
  const float mse_threshold = 0.6f;      // Allow quantization noise
  const float cos_sim_threshold = 0.99f; // High similarity expected

  EXPECT_LE(mean_squared_error, mse_threshold);
  EXPECT_GE(cos_sim, cos_sim_threshold);
}

#define DECLARE_transform_osv32_to_qsi4c32p_test(K, N)                         \
  TEST(nntrainer_cpu_backend_standalone,                                       \
       transform_osv32_to_qsi4c32p_K##K##_N##N) {                              \
    run_transform_osv32_to_qsi4c32p_test(K, N, 3, true);                       \
  }

// Test cases with various K and N dimensions
DECLARE_transform_osv32_to_qsi4c32p_test(128, 64);
DECLARE_transform_osv32_to_qsi4c32p_test(256, 128);
DECLARE_transform_osv32_to_qsi4c32p_test(512, 256);
DECLARE_transform_osv32_to_qsi4c32p_test(512, 512);
DECLARE_transform_osv32_to_qsi4c32p_test(1024, 512);
DECLARE_transform_osv32_to_qsi4c32p_test(1024, 1024);

TEST(nntrainer_cpu_backend_standalone, trigonometric_values_test) {

  const unsigned int N = 3072;
  run_trigonometric_values_test(N);
}

/**
 * @brief Benchmark comparison of three GEMM implementations
 *
 * Compares latency of:
 * - nntr_gemm_qsi8d32p_qsi4c32p_packed (KleidiAI with block size 32)
 * - nntr_gemm_qai8dxp_qsi4cxp_packed (KleidiAI with dynamic block)
 * - gemm_q4_0<float> (GGML-style Q4_0 GEMM)
 */
void run_gemm_benchmark_comparison(const uint32_t M, const uint32_t K,
                                   const uint32_t N,
                                   const uint32_t warmup_iters = 3,
                                   const uint32_t test_iters = 5,
                                   bool print = false) {
  nntrainer::init_backend();

  if (print) {
    std::cout << "\n=========================================" << std::endl;
    std::cout << "[BENCHMARK] GEMM Latency Comparison (M:" << M << ", K:" << K
              << ", N:" << N << ")" << std::endl;
    std::cout << "=========================================\n" << std::endl;
  }

  // Generate random data
  std::vector<float> activation =
    generate_random_vector<float>(static_cast<std::size_t>(M) * K);
  std::vector<float> weight =
    generate_random_vector<float>(static_cast<std::size_t>(N) * K);

  // ============================================================
  // Setup 1: qsi8d32p_qsi4c32p (KleidiAI with block size 32)
  // ============================================================
  const size_t bl = 32;
  const size_t num_blocks_per_row = K / bl;
  const size_t bytes_per_block = sizeof(uint16_t) + bl / 2;
  const size_t rhs_native_size_qs4c32 =
    static_cast<size_t>(N) * num_blocks_per_row * bytes_per_block;

  std::vector<uint8_t> rhs_native_mtx_qs4c32(rhs_native_size_qs4c32, 0);
  nntrainer::nntr_quant_qs4c32_f32(N, K, bl, (void *)weight.data(),
                                   (void *)rhs_native_mtx_qs4c32.data());

  // Get optimal kernel index for qsi8d32p_qsi4c32p by running unpacked version
  float dummy_mse;
  std::vector<float> ref_dst(static_cast<std::size_t>(M) * N);
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);

  const auto [mse_qsi8d32p, opt_idx_qsi8d32p] =
    test_gemm_qsi8d32p_qsi4c32p_unpacked(
      M, K, N, weight.data(), activation.data(), ref_dst, true, false);
  if (print) {
    std::cout << "[INFO] qsi8d32p_qsi4c32p optimal kernel index: "
              << opt_idx_qsi8d32p << std::endl;
  }

  // Pack weights for qsi8d32p_qsi4c32p
  size_t packed_weight_size_qsi8d32p =
    nntrainer::nntr_get_rhs_packed_size_qsi8d32p_qsi4c32p(
      N, K, opt_idx_qsi8d32p, true);
  std::vector<uint8_t> packed_weight_qsi8d32p(packed_weight_size_qsi8d32p);
  nntrainer::nntr_qsi8d32p_qsi4c32p_rhs_pack(
    N, K, packed_weight_qsi8d32p.data(), rhs_native_mtx_qs4c32.data(), nullptr,
    opt_idx_qsi8d32p, true);

  // ============================================================
  // Setup 3: gemm_q4_0<float> (GGML-style Q4_0)
  // ============================================================
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  int64_t q4_0_block_size = 32;
  size_t q4_0_data_size = q4_0_type_size * N / q4_0_block_size;
  q4_0_data_size *= K;
  std::vector<char> q4_0_offline_qWeight(q4_0_data_size);
  nntrainer::quantize_q4_0(weight.data(), (void *)q4_0_offline_qWeight.data(),
                           N, K, nullptr);

  std::vector<char> q4_0_repacked_qWeight(q4_0_data_size);
  nntrainer::repack_q4_0(q4_0_repacked_qWeight.data(),
                         q4_0_offline_qWeight.data(), q4_0_data_size, N, K);

  // Output buffers
  std::vector<float> dst_qsi8d32p(static_cast<size_t>(M) * N);
  std::vector<float> dst_qai8dxp(static_cast<size_t>(M) * N);
  std::vector<float> dst_q4_0(static_cast<size_t>(M) * N);

  // ============================================================
  // Warm-up runs
  // ============================================================
  if (print) {
    std::cout << "[INFO] Warm-up (" << warmup_iters << " iterations)..."
              << std::endl;
  }
  for (uint32_t i = 0; i < warmup_iters; ++i) {
    nntrainer::nntr_gemm_qsi8d32p_qsi4c32p_packed(
      M, N, K, (void *)activation.data(), (void *)packed_weight_qsi8d32p.data(),
      dst_qsi8d32p.data(), opt_idx_qsi8d32p, true);

    nntrainer::gemm_q4_0<float>(M, N, K, activation.data(), K,
                                (void *)q4_0_repacked_qWeight.data(), N,
                                dst_q4_0.data(), N);
  }

  // ============================================================
  // Benchmark: qsi8d32p_qsi4c32p_packed
  // ============================================================
  nanoseconds total_time_qsi8d32p = nanoseconds(0);
  for (uint32_t i = 0; i < test_iters; ++i) {
    auto t1 = high_resolution_clock::now();
    nntrainer::nntr_gemm_qsi8d32p_qsi4c32p_packed(
      M, N, K, (void *)activation.data(), (void *)packed_weight_qsi8d32p.data(),
      dst_qsi8d32p.data(), opt_idx_qsi8d32p, true);
    auto t2 = high_resolution_clock::now();
    total_time_qsi8d32p += duration_cast<nanoseconds>(t2 - t1);
  }

  // ============================================================
  // Benchmark: gemm_q4_0<float>
  // ============================================================
  nanoseconds total_time_q4_0 = nanoseconds(0);
  for (uint32_t i = 0; i < test_iters; ++i) {
    auto t1 = high_resolution_clock::now();
    nntrainer::gemm_q4_0<float>(M, N, K, activation.data(), K,
                                (void *)q4_0_repacked_qWeight.data(), N,
                                dst_q4_0.data(), N);
    auto t2 = high_resolution_clock::now();
    total_time_q4_0 += duration_cast<nanoseconds>(t2 - t1);
  }

  // ============================================================
  // Print results
  // ============================================================
  auto avg_ns_qsi8d32p = total_time_qsi8d32p.count() / test_iters;
  auto avg_ns_q4_0 = total_time_q4_0.count() / test_iters;

  if (print) {
    std::cout << "\n-----------------------------------------" << std::endl;
    std::cout << "[RESULT] Average latency over " << test_iters
              << " iterations:" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "  qsi8d32p_qsi4c32p_packed: " << avg_ns_qsi8d32p << " ns ("
              << avg_ns_qsi8d32p / 1'000 << " us, "
              << avg_ns_qsi8d32p / 1'000'000 << " ms)" << std::endl;
    std::cout << "  gemm_q4_0<float>:         " << avg_ns_q4_0 << " ns ("
              << avg_ns_q4_0 / 1'000 << " us, " << avg_ns_q4_0 / 1'000'000
              << " ms)" << std::endl;
    std::cout << "-----------------------------------------\n" << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, gemm_benchmark_comparison_32x1024x4096) {
  run_gemm_benchmark_comparison(32, 1024, 4096);
}

TEST(nntrainer_cpu_backend_standalone, gemm_benchmark_comparison_1x3072x512) {
  run_gemm_benchmark_comparison(1, 3072, 512);
}

/* ---- [#152] the fp16 CPU attention equals attn_m1_det.h ----------------- */

namespace {

/** @brief LFM2.5's attention shape. */
constexpr unsigned kF16Kv = 8, kF16Gqa = 4, kF16Hd = 64,
                   kF16Nq = kF16Kv * kF16Gqa;

uint32_t f16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

/** @brief Bit mismatches of fp16 results widened to f32 against the spec;
 *         the first one printed. */
int f16_count_bad(const std::vector<_FP16> &cpu, const std::vector<float> &spec,
                  const char *what) {
  int bad = 0;
  for (size_t i = 0; i < spec.size(); ++i) {
    const float c = static_cast<float>(cpu[i]);
    if (f16_bits(c) != f16_bits(spec[i])) {
      if (bad == 0) {
        std::cout << "ATTN_M1_F16 " << what << " first mismatch i=" << i
                  << std::hexfloat << " cpu=" << c << " spec=" << spec[i]
                  << std::defaultfloat << std::endl;
      }
      ++bad;
    }
  }
  return bad;
}

/** @brief The cos | sin row of one position in f32, the CPU's formula (the
 *         host check's rope_cs); the CPU's fp16 table is its (_FP16) cast. */
void f16_rope_cs(float *cs, unsigned pos) {
  for (unsigned i = 0; i < 32u; ++i) {
    const double ang = (double)pos * std::pow(5e6, -(2.0 * i) / 64.0);
    cs[i] = (float)std::cos(ang);
    cs[32u + i] = (float)std::sin(ang);
  }
}

/** @brief f32 -> fp16 with the conversion Tensor::copyData uses. */
std::vector<_FP16> f16_copy(const float *x, size_t n) {
  std::vector<_FP16> y(n);
  nntrainer::scopy(static_cast<unsigned>(n), x, 1u, y.data(), 1u);
  return y;
}

} // namespace

/**
 * @brief compute_rotary_emb_value(__fp16) against m1_rope64_det on 32 q +
 *        8 k heads of every kind at positions 0, 1, 511, 1023, 4095.
 */
TEST(AttnM1F16Det, RopeMatchesNeonFp16) {
  const unsigned heads = kF16Nq + kF16Kv, n = heads * kF16Hd;
  amc_rng rng{0x15200301u};
  std::vector<float> x(n), cs(64);
  for (unsigned h = 0; h < heads; ++h) {
    amc_fill_row(&rng, x.data() + h * kF16Hd, kF16Hd, h < 3u ? (int)h + 1 : 0);
  }
  for (unsigned pos : {0u, 1u, 511u, 1023u, 4095u}) {
    f16_rope_cs(cs.data(), pos);
    std::vector<_FP16> c16(32), s16(32);
    for (unsigned i = 0; i < 32u; ++i) {
      c16[i] = static_cast<_FP16>(cs[i]);
      s16[i] = static_cast<_FP16>(cs[32u + i]);
    }
    std::vector<_FP16> x16 = f16_copy(x.data(), n);
    nntrainer::compute_rotary_emb_value(n, kF16Hd, kF16Hd / 2u, x16.data(),
                                        x16.data(), c16.data(), s16.data());
    std::vector<float> y = x;
    for (unsigned h = 0; h < heads; ++h) {
      m1_rope64_det(y.data() + h * kF16Hd, cs.data());
    }
    const int bad = f16_count_bad(x16, y, "rope64");
    std::cout << "ATTN_M1_F16 rope64 pos=" << pos << " bad=" << bad << " of "
              << n << std::endl;
    EXPECT_EQ(bad, 0) << "pos " << pos;
  }
}

/**
 * @brief The Android CPU's decode attention, chained as mha_core.cpp's
 *        ENABLE_FP16 branch calls it (copyData, compute_rotary_emb_value
 *        on q and on the new k row into the cache, compute_kcaches,
 *        softmax_row_inplace, compute_fp16vcache_transposed), against
 *        m1_rope64_det + attn_m1_det.h, bit for bit, at L = 1 .. 1024 with
 *        attn_m1_cases.h's rows (the fused-FMA midpoint cases). Each L runs
 *        twice: RoPE at position 0 (the identity, so the adversarial rows
 *        survive it; the case counts are printed there only) and at
 *        position L - 1.
 */
TEST(AttnM1F16Det, AttentionMatchesNeonFp16) {
  const size_t row = (size_t)kF16Kv * kF16Hd, nq = (size_t)kF16Nq * kF16Hd;
  for (unsigned L : {1u, 2u, 63u, 64u, 65u, 512u, 513u, 1024u}) {
    amc_rng rng{0x15200400u + L};
    std::vector<float> q(nq), k(L * row), v(L * row), cs(64);
    amc_fill_q(&rng, q.data(), kF16Nq, kF16Hd);
    amc_fill_kv(&rng, k.data(), v.data(), L, kF16Kv, kF16Gqa, kF16Hd);
    const uint32_t planted =
      amc_plant_pv(q.data(), k.data(), v.data(), L, kF16Kv, kF16Gqa, kF16Hd);
    for (unsigned pos : {0u, L - 1u}) {
      f16_rope_cs(cs.data(), pos);
      std::vector<_FP16> c16(32), s16(32);
      for (unsigned i = 0; i < 32u; ++i) {
        c16[i] = static_cast<_FP16>(cs[i]);
        s16[i] = static_cast<_FP16>(cs[32u + i]);
      }
      /* The CPU: rows 0 .. L-2 of the caches as the earlier steps left
         them (fp16), the new k row roped into the cache, q roped. */
      std::vector<_FP16> q16 = f16_copy(q.data(), nq);
      std::vector<_FP16> kc = f16_copy(k.data(), L * row);
      std::vector<_FP16> vc = f16_copy(v.data(), L * row);
      std::vector<_FP16> knew = f16_copy(k.data() + (L - 1u) * row, row);
      nntrainer::compute_rotary_emb_value(row, kF16Hd, kF16Hd / 2u, knew.data(),
                                          kc.data() + (L - 1u) * row,
                                          c16.data(), s16.data());
      nntrainer::compute_rotary_emb_value(nq, kF16Hd, kF16Hd / 2u, q16.data(),
                                          q16.data(), c16.data(), s16.data());
      std::vector<_FP16> sc((size_t)L * kF16Nq), o16(nq);
      nntrainer::compute_kcaches(q16.data(), kc.data(), sc.data(), (int)L,
                                 (int)kF16Kv, (int)kF16Hd, (int)kF16Gqa, 4);
      nntrainer::softmax_row_inplace(sc.data(), 0, L, kF16Nq);
      nntrainer::compute_fp16vcache_transposed((int)L - 1, sc.data(), vc.data(),
                                               o16.data(), (int)kF16Kv,
                                               (int)kF16Gqa, (int)kF16Hd);
      /* The spec: the same rows, RoPE on q and on the last k row. */
      std::vector<float> qs = q, ks = k, kt((size_t)kF16Kv * kF16Hd * L),
                         vv((size_t)kF16Kv * L * kF16Hd), e(L), out(nq);
      for (unsigned h = 0; h < kF16Nq; ++h) {
        m1_rope64_det(qs.data() + h * kF16Hd, cs.data());
      }
      for (unsigned h = 0; h < kF16Kv; ++h) {
        m1_rope64_det(ks.data() + (L - 1u) * row + h * kF16Hd, cs.data());
      }
      for (unsigned p = 0; p < L; ++p) {
        for (unsigned h = 0; h < kF16Kv; ++h) {
          attn_m1_det_append(kt.data() + (size_t)h * kF16Hd * L,
                             vv.data() + (size_t)h * L * kF16Hd, kF16Hd, L, p,
                             ks.data() + p * row + h * kF16Hd,
                             v.data() + p * row + h * kF16Hd);
        }
      }
      attn_m1_det_forward(qs.data(), kt.data(), vv.data(), kF16Kv, kF16Gqa,
                          kF16Hd, L, L, 0.125f, e.data(), out.data(), nullptr);
      const int bad = f16_count_bad(o16, out, "attention out");
      std::cout << "ATTN_M1_F16 L=" << L << " rope_pos=" << pos
                << " out bad=" << bad << " of " << nq
                << " pv_cases=" << (pos == 0u ? planted : 0u)
                << " score_cases=" << (pos == 0u && L > 3u ? 2u * (L - 3u) : 0u)
                << std::endl;
      EXPECT_EQ(bad, 0) << "L=" << L << " rope_pos=" << pos;
    }
  }
}

/**
 * @brief exp16 through the CPU's own softmax: softmax_row_inplace(_FP16)
 *        on two rows, 0 and d, one head per fp16 d <= 0 (all 31744 finite
 *        values, -0 to -65504, which include plan 152's [-17.5, 0]). Then
 *        p0 = 1 / (1 + e(d)) and p1 = e(d) / (1 + e(d)) against
 *        attn_m1_det_softmax on {0, d}; for d < -7.6, 1 + e(d) rounds to 1
 *        and p1 is e(d) itself, so the probe reads the exp directly.
 */
TEST(AttnM1F16Det, ExpProbeExhaustive) {
  std::vector<float> d;
  for (uint32_t b = 0x8000u; b < 0xFC00u; ++b) {
    const uint16_t hb = static_cast<uint16_t>(b);
    _FP16 h;
    std::memcpy(&h, &hb, sizeof(h));
    d.push_back(static_cast<float>(h));
  }
  const size_t heads = (d.size() + 7u) & ~(size_t)7u;
  std::vector<_FP16> sc(2u * heads, static_cast<_FP16>(0.0f));
  for (size_t i = 0; i < d.size(); ++i) {
    sc[heads + i] = static_cast<_FP16>(d[i]);
  }
  nntrainer::softmax_row_inplace(sc.data(), 0, 2, heads);
  int bad0 = 0, bad1 = 0, bad_range = 0;
  for (size_t i = 0; i < d.size(); ++i) {
    float s[2] = {0.0f, d[i]};
    attn_m1_det_softmax(s, 2u, nullptr, nullptr);
    const bool b0 = f16_bits(static_cast<float>(sc[i])) != f16_bits(s[0]);
    const bool b1 =
      f16_bits(static_cast<float>(sc[heads + i])) != f16_bits(s[1]);
    if ((b0 || b1) && bad0 + bad1 == 0) {
      std::cout << "ATTN_M1_F16 exp probe first mismatch d=" << std::hexfloat
                << d[i] << " cpu=(" << static_cast<float>(sc[i]) << ", "
                << static_cast<float>(sc[heads + i]) << ") spec=(" << s[0]
                << ", " << s[1] << ")" << std::defaultfloat << std::endl;
    }
    bad0 += b0;
    bad1 += b1;
    bad_range += (b0 || b1) && d[i] >= -17.5f;
  }
  std::cout << "ATTN_M1_F16 exp probe d<=0 n=" << d.size() << " bad_p0=" << bad0
            << " bad_p1=" << bad1 << " bad_in[-17.5,0]=" << bad_range
            << std::endl;
  EXPECT_EQ(bad0 + bad1, 0);
}

/* ---- [#162] decode attention phases: time and bit-compare --------------- */

namespace {

/** @brief The library's -march for the fp16 NEON prototypes below; the test
 *         itself builds without it (test/jni ARM_MARCH_FLAGS is empty unless
 *         MESON_ARM_MARCH is set). */
#define MHA_M1_FP16 __attribute__((target("arch=armv8.2-a+fp16")))

/** @brief One score as compute_kcaches(__fp16) forms it: the fmla chain in
 *         acc, the vpaddq tree, 0 + lane, / sqrt(64). */
MHA_M1_FP16 inline __fp16 mha_m1_score(float16x8_t acc) {
  acc = vpaddq_f16(acc, acc);
  acc = vpaddq_f16(acc, acc);
  acc = vpaddq_f16(acc, acc);
  __fp16 sum = 0.0f;
  sum += vgetq_lane_f16(acc, 0);
  return sum / sqrt((float)kF16Hd);
}

/** @brief kv_ilp kcache prototype for kv head n: compute_kcaches(__fp16)'s
 *         per-position chain unchanged, 4 positions interleaved in 4
 *         independent accumulators (plan 162 section 3.2 (a2)). */
MHA_M1_FP16 void mha_m1_kcache_ilp(const __fp16 *q, const __fp16 *kc,
                                   __fp16 *out, int L, int n) {
  const int stride = kF16Kv * kF16Hd;
  for (int r0 = 0; r0 < L; r0 += 4) {
    const int nr = std::min(4, L - r0);
    for (unsigned g = 0; g < kF16Gqa; ++g) {
      const __fp16 *qp = q + (n * kF16Gqa + g) * kF16Hd;
      const __fp16 *k0 = kc + (size_t)r0 * stride + n * kF16Hd;
      __fp16 *o = out + (size_t)r0 * kF16Nq + n * kF16Gqa + g;
      if (nr == 4) {
        float16x8_t a0 = vdupq_n_f16(0.0), a1 = a0, a2 = a0, a3 = a0;
        for (unsigned i = 0; i < kF16Hd; i += 8) {
          const float16x8_t qv = vld1q_f16(qp + i);
          a0 = vfmaq_f16(a0, qv, vld1q_f16(k0 + i));
          a1 = vfmaq_f16(a1, qv, vld1q_f16(k0 + stride + i));
          a2 = vfmaq_f16(a2, qv, vld1q_f16(k0 + 2 * stride + i));
          a3 = vfmaq_f16(a3, qv, vld1q_f16(k0 + 3 * stride + i));
        }
        o[0] = mha_m1_score(a0);
        o[kF16Nq] = mha_m1_score(a1);
        o[2 * kF16Nq] = mha_m1_score(a2);
        o[3 * kF16Nq] = mha_m1_score(a3);
        continue;
      }
      for (int r = 0; r < nr; ++r) {
        float16x8_t a = vdupq_n_f16(0.0);
        for (unsigned i = 0; i < kF16Hd; i += 8) {
          a = vfmaq_f16(a, vld1q_f16(qp + i), vld1q_f16(k0 + r * stride + i));
        }
        o[r * kF16Nq] = mha_m1_score(a);
      }
    }
  }
}

/** @brief kv_ilp vcache prototype for kv head n: gqa 4, head_dim 64, the 32
 *         accumulators of compute_fp16vcache_transposed held in registers
 *         over two passes of 16; each one's fmla sequence over positions
 *         stays ascending (plan 162 section 3.2 (a2)). */
MHA_M1_FP16 void mha_m1_vcache_reg(const __fp16 *s, const __fp16 *vc,
                                   __fp16 *out, int L, int n) {
  for (unsigned half = 0; half < 2u; ++half) {
    float16x8_t acc[kF16Gqa][4];
    for (unsigned h = 0; h < kF16Gqa; ++h) {
      for (unsigned b = 0; b < 4u; ++b) {
        acc[h][b] = vdupq_n_f16(0.0f);
      }
    }
    for (int j = 0; j < L; ++j) {
      const __fp16 *vp =
        vc + ((size_t)j * kF16Kv + n) * kF16Hd + half * (kF16Hd / 2u);
      const __fp16 *sp = s + (size_t)j * kF16Nq + n * kF16Gqa;
      float16x8_t v[4];
      for (unsigned b = 0; b < 4u; ++b) {
        v[b] = vld1q_f16(vp + 8u * b);
      }
      for (unsigned h = 0; h < kF16Gqa; ++h) {
        const float16x8_t a = vdupq_n_f16(sp[h]);
        for (unsigned b = 0; b < 4u; ++b) {
          acc[h][b] = vfmaq_f16(acc[h][b], a, v[b]);
        }
      }
    }
    for (unsigned h = 0; h < kF16Gqa; ++h) {
      for (unsigned b = 0; b < 4u; ++b) {
        vst1q_f16(out + (n * kF16Gqa + h) * kF16Hd + half * (kF16Hd / 2u) +
                    8u * b,
                  acc[h][b]);
      }
    }
  }
}

/** @brief split4 softmax: each group of 8 heads gathered into a
 *         thread-local [L][8] buffer, the unchanged softmax_row_inplace on
 *         it, scattered back, one group per pool job (plan 162 section 3.2
 *         (a1)). */
void mha_m1_softmax_split4(__fp16 *sc, int L) {
  nntrainer::ThreadManager::Global().parallel_for(
    0, kF16Nq / 8u, [=](size_t grp) {
      thread_local std::vector<__fp16> buf;
      buf.resize((size_t)L * 8u);
      for (int r = 0; r < L; ++r) {
        std::memcpy(buf.data() + r * 8u, sc + (size_t)r * kF16Nq + grp * 8u,
                    8u * sizeof(__fp16));
      }
      nntrainer::softmax_row_inplace(buf.data(), 0, L, 8);
      for (int r = 0; r < L; ++r) {
        std::memcpy(sc + (size_t)r * kF16Nq + grp * 8u, buf.data() + r * 8u,
                    8u * sizeof(__fp16));
      }
    });
}

double mha_m1_median(std::vector<double> v) {
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

} // namespace

/**
 * @brief [#162 step 0] The decode mha_core sequence at LFM2.5's shape (32 q
 *        / 8 kv heads, head_dim 64, fp16) on ThreadManager::Global() as
 *        mha_core.cpp runs it (kcache over the 8 kv heads, softmax, vcache
 *        over the 8 kv heads), per phase timed at L = 513 / 1024 / 1536:
 *        impl=as_is (the exported functions), split4 (softmax per 8-head
 *        group on the pool), kv_ilp (the two register kernels above), all
 *        (both). 50 timed iterations (medians) after 5 warm-ups; bad counts,
 *        over all 55, the fp16 outputs and the softmax rows that differ
 *        bitwise from as_is. Run with NNTR_NUM_THREADS=8.
 */
TEST(MhaM1Phases, Decode) {
  using clk = std::chrono::steady_clock;
  const int kWarm = 5, kIters = 50;
  const size_t row = (size_t)kF16Kv * kF16Hd, nq = (size_t)kF16Nq * kF16Hd;
  auto &tm = nntrainer::ThreadManager::Global();
  const char *impls[] = {"as_is", "split4", "kv_ilp", "all"};
  for (int L : {513, 1024, 1536}) {
    std::mt19937 rng(0x16200000u + L);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<_FP16> q(nq), kc(L * row), vc(L * row);
    for (auto *x : {&q, &kc, &vc}) {
      for (auto &e : *x) {
        e = static_cast<_FP16>(dist(rng));
      }
    }
    std::vector<_FP16> sc((size_t)L * kF16Nq), o(nq), ref_sc, ref_o;
    for (int impl = 0; impl < 4; ++impl) {
      const bool split = impl == 1 || impl == 3, ilp = impl >= 2;
      std::vector<double> t_k, t_s, t_v, t_all;
      int bad = 0;
      for (int it = 0; it < kWarm + kIters; ++it) {
        const _FP16 *qd = q.data(), *kd = kc.data(), *vd = vc.data();
        _FP16 *sd = sc.data(), *od = o.data();
        const auto t0 = clk::now();
        tm.parallel_for(0, kF16Kv, [=](size_t n) {
          if (ilp) {
            mha_m1_kcache_ilp(qd, kd, sd, L, (int)n);
          } else {
            nntrainer::compute_kcaches(qd, kd, sd, L, kF16Kv, kF16Hd, kF16Gqa,
                                       4, UINT_MAX, (int)n, (int)n + 1);
          }
        });
        const auto t1 = clk::now();
        if (split) {
          mha_m1_softmax_split4(sd, L);
        } else {
          nntrainer::softmax_row_inplace(sd, 0, L, kF16Nq);
        }
        const auto t2 = clk::now();
        tm.parallel_for(0, kF16Kv, [=](size_t n) {
          if (ilp) {
            mha_m1_vcache_reg(sd, vd, od, L, (int)n);
          } else {
            nntrainer::compute_fp16vcache_transposed(L - 1, sd, vd, od, kF16Kv,
                                                     kF16Gqa, kF16Hd, UINT_MAX,
                                                     (int)n, (int)n + 1);
          }
        });
        const auto t3 = clk::now();
        if (it >= kWarm) {
          auto us = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double, std::micro>(b - a).count();
          };
          t_k.push_back(us(t0, t1));
          t_s.push_back(us(t1, t2));
          t_v.push_back(us(t2, t3));
          t_all.push_back(us(t0, t3));
        }
        if (impl == 0 && it == 0) {
          ref_sc = sc;
          ref_o = o;
        }
        for (size_t i = 0; i < nq; ++i) {
          bad += f16_bits(static_cast<float>(o[i])) !=
                 f16_bits(static_cast<float>(ref_o[i]));
        }
        for (int r = 0; r < L; ++r) {
          bad += std::memcmp(sc.data() + (size_t)r * kF16Nq,
                             ref_sc.data() + (size_t)r * kF16Nq,
                             kF16Nq * sizeof(_FP16)) != 0;
        }
      }
      std::cout << std::fixed << std::setprecision(1) << "MHA_M1_PHASE L=" << L
                << " impl=" << impls[impl]
                << " threads=" << tm.getComputeThreadCount()
                << " kcache_us=" << mha_m1_median(t_k)
                << " softmax_us=" << mha_m1_median(t_s)
                << " vcache_us=" << mha_m1_median(t_v)
                << " total_us=" << mha_m1_median(t_all) << " bad=" << bad
                << " iters=" << kIters << std::defaultfloat << std::endl;
      EXPECT_EQ(bad, 0) << "L=" << L << " impl=" << impls[impl];
    }
  }
}

int main(int argc, char **argv) {
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

  return result;
}
