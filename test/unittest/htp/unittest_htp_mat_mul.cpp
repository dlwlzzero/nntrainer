// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_htp_mat_mul.cpp
 * @date	23 April 2026
 * @brief	Unit tests for HTP matrix multiplication kernels.
 *              Covers FastRPC paths for pwf16 / wf16 / pwqk0 weight layouts
 *              and the shared-memory chan path for pwf16, so MSE and
 *              round-trip latency can be compared case-by-case.
 * @see		https://github.com/nnstreamer/nntrainer
 * @bug		No known bugs except for NYI items
 */

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

#if defined(ENABLE_HTP)

#include <fp16.h>
#include <htp_interface.h>
#include <message.h>
#include <nntrainer_test_util.h>
#include <op_reg.h>
#include <q4_0_utils.h>

#include "unittest_htp_common.h"

using namespace nntrainer;

/* ============================================================
 * mat_mul_af32_pwf16_of32  (FastRPC; permuted fp16 weight)
 * ============================================================
 */

static void run_mat_mul_af32_pwf16_of32_test(const uint32_t M, const uint32_t K,
                                             const uint32_t N) {
  auto &htp = htp::HtpInterface::instance();

  ASSERT_NE(htp.htp_ops_mat_mul_af32_pwf16_of32, nullptr)
    << "HTP library not loaded";

  auto handle = htp.get_global_handle();
  ASSERT_NE(handle, (uint64_t)0) << "DSP session not opened (handle == 0)";

  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  std::vector<float> weight =
    generate_random_vector<float, false>(N * K, -0.1f, 0.1f);

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

  memcpy(activation_ptr, activation.data(), M * K * sizeof(float));
  memset(weight_ptr, 0, K * N * sizeof(uint16_t));
  permute_weight_to_fp16_tiles(weight.data(), weight_ptr, K, N);

  err = htp.htp_ops_mat_mul_af32_pwf16_of32(handle, output_fd, 0, activation_fd,
                                            0, weight_fd, 0, M, K, N);
  ASSERT_EQ(err, 0) << "htp_ops_mat_mul_af32_pwf16_of32 failed";

  std::vector<float> hmx_dst(M * N);
  memcpy(hmx_dst.data(), output_ptr, M * N * sizeof(float));

  float mse_err = mse<float>(hmx_dst.data(), ref_dst.data(), M * N);
  std::cout << "W16A32 GEMM: " << M << " x " << K << " x " << N << std::endl;
  std::cout << " - MSE (vs mixed-precision ref): " << mse_err << std::endl;

  EXPECT_IN_RANGE(mse_err, 0.0f, 0.01f);

  htp.free_shared_mem_buf(output_ptr, output_fd, M * N * sizeof(float));
  htp.free_shared_mem_buf(activation_ptr, activation_fd,
                          M * K * sizeof(float));
  htp.free_shared_mem_buf(weight_ptr, weight_fd, K * N * sizeof(uint16_t));
}

#define DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(M, K, N)                   \
  TEST(nntrainer_htp_mat_mul, mat_mul_af32_pwf16_of32_##M##_##K##_##N) {      \
    run_mat_mul_af32_pwf16_of32_test(M, K, N);                                \
  }

DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 32, 32);      // min tile
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 256, 256);    // medium square
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 1024, 1024);  // large square
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(1, 1024, 1024);   // GEMV large
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 1024, 256);   // K > N
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(32, 256, 1024);   // K < N
DECLARE_mat_mul_af32_pwf16_of32_test_M_K_N(68, 256, 256);    // non-power-of-2 M

/* ============================================================
 * mat_mul_af32_pwf16_of32_chan  (shared-memory message channel path)
 *
 * Dispatches the same DSP kernel as the FastRPC variant above via
 * HTP_OPS_MAT_MUL_PERMUTED_W16A32 in op_executor.cc, so MSE must match
 * the RPC result. Round-trip latency is printed for comparison.
 * ============================================================
 */

static void run_mat_mul_af32_pwf16_of32_chan_test(const uint32_t M,
                                                  const uint32_t K,
                                                  const uint32_t N) {
  auto &htp = htp::HtpInterface::instance();

  ASSERT_NE(htp.create_htp_message_channel, nullptr) << "HTP library not loaded";
  ASSERT_NE(htp.alloc_shared_mem_buf, nullptr) << "HTP library not loaded";

  ChanCtx &cctx = get_chan_ctx();
  if (cctx.chan == nullptr) {
    GTEST_SKIP() << "Could not create HTP message channel";
  }
  auto *msg = reinterpret_cast<MessageHeader *>(cctx.chan);

  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  std::vector<float> weight =
    generate_random_vector<float, false>(N * K, -0.1f, 0.1f);

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

  float    *output_ptr = nullptr;
  float    *activation_ptr = nullptr;
  uint16_t *weight_ptr = nullptr;
  int       output_fd, activation_fd, weight_fd;

  int err = htp.alloc_shared_mem_buf((void **)&output_ptr, &output_fd,
                                     M * N * sizeof(float));
  ASSERT_EQ(err, 0) << "Failed to allocate output buffer";

  err = htp.alloc_shared_mem_buf((void **)&activation_ptr, &activation_fd,
                                 M * K * sizeof(float));
  ASSERT_EQ(err, 0) << "Failed to allocate activation buffer";

  err = htp.alloc_shared_mem_buf((void **)&weight_ptr, &weight_fd,
                                 K * N * sizeof(uint16_t));
  ASSERT_EQ(err, 0) << "Failed to allocate weight buffer";

  memcpy(activation_ptr, activation.data(), M * K * sizeof(float));
  memset(weight_ptr, 0, K * N * sizeof(uint16_t));
  permute_weight_to_fp16_tiles(weight.data(), weight_ptr, K, N);

  RequestHeader req_hdr = {};
  req_hdr.state = 0;
  req_hdr.type  = REQUEST_TYPE_OP_COMPUTE;

  OpComputeRequest cr = {};
  cr.op = HTP_OPS_MAT_MUL_PERMUTED_W16A32;

  MatMulParams params = {};
  params.output.fd         = output_fd;
  params.output.offset     = 0;
  params.activation.fd     = activation_fd;
  params.activation.offset = 0;
  params.weight.fd         = weight_fd;
  params.weight.offset     = 0;
  params.m = (int32_t)M;
  params.k = (int32_t)K;
  params.n = (int32_t)N;

  size_t req_size = sizeof(req_hdr) + sizeof(cr) + sizeof(params);
  msg->state.d        = 0;
  msg->n_reqs         = 1;
  msg->req_offsets[0] = message_header_size(msg);
  msg->req_offsets[1] = msg->req_offsets[0] + req_size;

  auto *p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg, 0));
  *reinterpret_cast<RequestHeader *>(p)    = req_hdr;    p += sizeof(req_hdr);
  *reinterpret_cast<OpComputeRequest *>(p) = cr;         p += sizeof(cr);
  *reinterpret_cast<MatMulParams *>(p)     = params;

  int64_t t0 = htp_now_us();
  msg->state.v[0] = 1;
  while (msg->state.v[1] != 1) {
    /* busy-poll */
  }
  int64_t elapsed_us = htp_now_us() - t0;

  int dsp_err = message_header_get_request_ptr(msg, 0)->state;
  ASSERT_EQ(dsp_err, 0) << "chan matmul returned err 0x" << std::hex << dsp_err;

  std::vector<float> hmx_dst(M * N);
  memcpy(hmx_dst.data(), output_ptr, M * N * sizeof(float));

  float mse_err = mse<float>(hmx_dst.data(), ref_dst.data(), M * N);
  std::cout << "W16A32 GEMM (chan): " << M << " x " << K << " x " << N
            << std::endl;
  std::cout << " - MSE (vs mixed-precision ref): " << mse_err << std::endl;
  std::cout << " - chan round-trip: " << elapsed_us << " us" << std::endl;

  EXPECT_IN_RANGE(mse_err, 0.0f, 0.01f);

  // Release DSP-side fd mappings so the next test's allocations do not hit
  // the rpcmem/fastrpc limit.
  {
    RequestHeader rel_hdr = {};
    rel_hdr.state = 0;
    rel_hdr.type  = REQUEST_TYPE_RPCMEM_MAP;

    RpcmemMapRequest rel_req = {};
    rel_req.n_puts = 3;
    rel_req.n_gets = 0;

    size_t rel_size = sizeof(rel_hdr) + sizeof(rel_req) + 3 * sizeof(int);
    msg->state.d        = 0;
    msg->n_reqs         = 1;
    msg->req_offsets[0] = message_header_size(msg);
    msg->req_offsets[1] = msg->req_offsets[0] + rel_size;

    auto *rp =
      reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg, 0));
    *reinterpret_cast<RequestHeader *>(rp)    = rel_hdr;   rp += sizeof(rel_hdr);
    *reinterpret_cast<RpcmemMapRequest *>(rp) = rel_req;   rp += sizeof(rel_req);
    *reinterpret_cast<int *>(rp) = output_fd;              rp += sizeof(int);
    *reinterpret_cast<int *>(rp) = activation_fd;          rp += sizeof(int);
    *reinterpret_cast<int *>(rp) = weight_fd;

    msg->state.v[0] = 1;
    while (msg->state.v[1] != 1) {
      /* busy-poll for DSP ack */
    }
  }

  htp.free_shared_mem_buf(output_ptr, output_fd, M * N * sizeof(float));
  htp.free_shared_mem_buf(activation_ptr, activation_fd,
                          M * K * sizeof(float));
  htp.free_shared_mem_buf(weight_ptr, weight_fd, K * N * sizeof(uint16_t));
}

#define DECLARE_mat_mul_af32_pwf16_of32_chan_test_M_K_N(M, K, N)                \
  TEST(nntrainer_htp_mat_mul, mat_mul_af32_pwf16_of32_chan_##M##_##K##_##N) {   \
    run_mat_mul_af32_pwf16_of32_chan_test(M, K, N);                             \
  }

DECLARE_mat_mul_af32_pwf16_of32_chan_test_M_K_N(32, 32, 32);
DECLARE_mat_mul_af32_pwf16_of32_chan_test_M_K_N(32, 256, 256);
DECLARE_mat_mul_af32_pwf16_of32_chan_test_M_K_N(32, 1024, 1024);
DECLARE_mat_mul_af32_pwf16_of32_chan_test_M_K_N(1, 1024, 1024);
DECLARE_mat_mul_af32_pwf16_of32_chan_test_M_K_N(32, 1024, 256);
DECLARE_mat_mul_af32_pwf16_of32_chan_test_M_K_N(32, 256, 1024);
DECLARE_mat_mul_af32_pwf16_of32_chan_test_M_K_N(68, 256, 256);

/* ============================================================
 * mat_mul_af32_wf16_of32  (FastRPC; row-major fp16 weight)
 * ============================================================
 */

static void run_mat_mul_af32_wf16_of32_test(const uint32_t M, const uint32_t K,
                                            const uint32_t N) {
  auto &htp = htp::HtpInterface::instance();

  ASSERT_NE(htp.htp_ops_mat_mul_af32_wf16_of32, nullptr)
    << "HTP library not loaded";

  auto handle = htp.get_global_handle();
  ASSERT_NE(handle, (uint64_t)0) << "DSP session not opened (handle == 0)";

  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  // weight stored in [N x K]: row = output dim, col = reduction dim
  std::vector<float> weight =
    generate_random_vector<float, false>(N * K, -0.1f, 0.1f);

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

  memcpy(activation_ptr, activation.data(), M * K * sizeof(float));
  for (uint32_t i = 0; i < N * K; ++i) {
    weight_ptr[i] = compute_fp32_to_fp16(weight[i]);
  }

  err = htp.htp_ops_mat_mul_af32_wf16_of32(handle, output_fd, 0, activation_fd,
                                           0, weight_fd, 0, M, K, N);
  ASSERT_EQ(err, 0) << "htp_ops_mat_mul_af32_wf16_of32 failed";

  std::vector<float> hmx_dst(M * N);
  memcpy(hmx_dst.data(), output_ptr, M * N * sizeof(float));

  float mse_err = mse<float>(hmx_dst.data(), ref_dst.data(), M * N);
  std::cout << "WF16A32 GEMM: " << M << " x " << K << " x " << N << std::endl;
  std::cout << " - MSE (vs mixed-precision ref): " << mse_err << std::endl;

  EXPECT_IN_RANGE(mse_err, 0.0f, 0.01f);

  htp.free_shared_mem_buf(output_ptr, output_fd, M * N * sizeof(float));
  htp.free_shared_mem_buf(activation_ptr, activation_fd,
                          M * K * sizeof(float));
  htp.free_shared_mem_buf(weight_ptr, weight_fd, N * K * sizeof(uint16_t));
}

#define DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(M, K, N)                    \
  TEST(nntrainer_htp_mat_mul, mat_mul_af32_wf16_of32_##M##_##K##_##N) {       \
    run_mat_mul_af32_wf16_of32_test(M, K, N);                                 \
  }

DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 32, 32);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 256, 256);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 1024, 1024);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(1, 1024, 1024);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 1024, 256);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(32, 256, 1024);
DECLARE_mat_mul_af32_wf16_of32_test_M_K_N(68, 256, 256);

/* ============================================================
 * mat_mul_af32_pwqk0_of32  (FastRPC; Q4_0 quantized weight in x4x2 layout)
 * ============================================================
 */

static void run_mat_mul_af32_pwqk0_of32_test(const uint32_t M, const uint32_t K,
                                             const uint32_t N) {
  auto &htp = htp::HtpInterface::instance();

  ASSERT_NE(htp.htp_ops_mat_mul_af32_pwqk0_of32, nullptr)
    << "HTP library not loaded";

  auto handle = htp.get_global_handle();
  ASSERT_NE(handle, (uint64_t)0) << "DSP session not opened (handle == 0)";

  ASSERT_EQ(K % 256, 0u) << "K must be multiple of 256 for x4x2 format";
  ASSERT_EQ(N % 32, 0u) << "N must be multiple of 32";

  std::vector<float> activation =
    generate_random_vector<float, false>(M * K, -0.1f, 0.1f);
  std::vector<float> weight_f32 =
    generate_random_vector<float, false>(N * K, -0.1f, 0.1f);

  const int groups_per_row = K / 32;
  std::vector<block_q4_0> weight_q4(N * groups_per_row);
  for (uint32_t row = 0; row < N; ++row) {
    quantize_row_q4_0_ref(weight_f32.data() + row * K,
                          weight_q4.data() + row * groups_per_row, K);
  }

  std::vector<float> weight_deq(N * K);
  for (uint32_t row = 0; row < N; ++row) {
    for (int g = 0; g < groups_per_row; ++g) {
      dequantize_block_q4_0_ref(&weight_q4[row * groups_per_row + g],
                                weight_deq.data() + row * K + g * 32);
    }
  }

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

  const int quants_per_row = K / 2;
  const int n_superblocks = K / 256;
  const size_t row_stride = (size_t)quants_per_row + n_superblocks * 16;
  const size_t weight_x4x2_size = N * row_stride;
  std::vector<uint8_t> weight_x4x2(weight_x4x2_size, 0);
  size_t actual_stride =
    repack_q4_0_to_x4x2(weight_q4.data(), weight_x4x2.data(), N, K);
  ASSERT_EQ(actual_stride, row_stride);

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

  memcpy(activation_ptr, activation.data(), M * K * sizeof(float));
  memcpy(weight_ptr, weight_x4x2.data(), weight_x4x2_size);

  // weight_type = 2 for GGML_TYPE_Q4_0
  err = htp.htp_ops_mat_mul_af32_pwqk0_of32(handle, output_fd, 0, activation_fd,
                                            0, weight_fd, 0, M, K, N,
                                            /*wgt_dt=*/2);
  ASSERT_EQ(err, 0) << "htp_ops_mat_mul_af32_pwqk0_of32 failed";

  std::vector<float> hmx_dst(M * N);
  memcpy(hmx_dst.data(), output_ptr, M * N * sizeof(float));

  float mse_err = mse<float>(hmx_dst.data(), ref_dst.data(), M * N);
  std::cout << "Q4_0 x4x2 GEMM: " << M << " x " << K << " x " << N << std::endl;
  std::cout << " - MSE (vs Q4_0 dequant ref): " << mse_err << std::endl;

  // Q4_0 quantization error is higher than fp16; allow a looser threshold.
  EXPECT_IN_RANGE(mse_err, 0.0f, 0.05f);

  htp.free_shared_mem_buf(output_ptr, output_fd, M * N * sizeof(float));
  htp.free_shared_mem_buf(activation_ptr, activation_fd,
                          M * K * sizeof(float));
  htp.free_shared_mem_buf(weight_ptr, weight_fd, weight_x4x2_size);
}

#define DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(M, K, N)                   \
  TEST(nntrainer_htp_mat_mul, mat_mul_af32_pwqk0_of32_##M##_##K##_##N) {      \
    run_mat_mul_af32_pwqk0_of32_test(M, K, N);                                \
  }

// pwqk0 requires K % 256 == 0 and N % 32 == 0, so K=32 minimum isn't usable.
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 256, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 512, 512);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 1024, 1024);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(1, 1024, 1024);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 1024, 256);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(32, 256, 1024);
DECLARE_mat_mul_af32_pwqk0_of32_test_M_K_N(28, 256, 256);

#else

TEST(nntrainer_htp_mat_mul, htp_not_enabled) {
  GTEST_SKIP() << "HTP is not enabled";
}

#endif // ENABLE_HTP
