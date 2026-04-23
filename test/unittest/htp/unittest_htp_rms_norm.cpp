// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_htp_rms_norm.cpp
 * @date	23 April 2026
 * @brief	Unit tests for the HTP rms_norm_f32 kernel (FastRPC path).
 * @see		https://github.com/nnstreamer/nntrainer
 * @bug		No known bugs except for NYI items
 */

#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

#if defined(ENABLE_HTP)

#include <htp_interface.h>
#include <nntrainer_test_util.h>

#include "unittest_htp_common.h"

using namespace nntrainer;

/**
 * @brief CPU reference implementation of RMS normalization (f32).
 *
 * For each row of ne0 elements:
 *   scale = 1 / sqrt(mean(x^2) + eps)
 *   y[i]  = x[i] * scale
 */
static void rms_norm_f32_ref(float *dst, const float *src, int ne0, int ne1,
                             float eps) {
  for (int j = 0; j < ne1; ++j) {
    const float *x = src + j * ne0;
    float       *y = dst + j * ne0;
    float sum = 0.0f;
    for (int i = 0; i < ne0; ++i) {
      sum += x[i] * x[i];
    }
    float mean = sum / ne0;
    float scale = 1.0f / std::sqrt(mean + eps);
    for (int i = 0; i < ne0; ++i) {
      y[i] = x[i] * scale;
    }
  }
}

static void run_rms_norm_f32_test(const int ne0, const int ne1,
                                  const float eps) {
  auto &htp = htp::HtpInterface::instance();

  ASSERT_NE(htp.htp_ops_rms_norm_f32, nullptr)
    << "HTP library not loaded (htp_ops_rms_norm_f32 missing)";

  auto handle = htp.get_global_handle();
  ASSERT_NE(handle, (uint64_t)0) << "DSP session not opened (handle == 0)";

  std::vector<float> input =
    generate_random_vector<float, false>(ne0 * ne1, -1.0f, 1.0f);

  std::vector<float> ref_dst(ne0 * ne1);
  rms_norm_f32_ref(ref_dst.data(), input.data(), ne0, ne1, eps);

  int32_t eps_bits;
  memcpy(&eps_bits, &eps, sizeof(float));

  // The kernel writes ceil(ne0/32) full vectors per row, so the last row may
  // touch past ne0*ne1 floats. Pad the allocation accordingly.
  int ne0_padded = ((ne0 + 31) / 32) * 32;
  size_t buf_size =
    (size_t)((ne1 > 0 ? (ne1 - 1) * ne0 : 0) + ne0_padded) * sizeof(float);

  float *src_ptr = nullptr;
  float *dst_ptr = nullptr;
  int src_fd, dst_fd;

  int err = htp.alloc_shared_mem_buf((void **)&dst_ptr, &dst_fd, buf_size);
  ASSERT_EQ(err, 0) << "Failed to allocate output buffer";

  err = htp.alloc_shared_mem_buf((void **)&src_ptr, &src_fd, buf_size);
  ASSERT_EQ(err, 0) << "Failed to allocate input buffer";

  memset(src_ptr, 0, buf_size);
  memcpy(src_ptr, input.data(), ne0 * ne1 * sizeof(float));

  err = htp.htp_ops_rms_norm_f32(handle, dst_fd, 0, src_fd, 0, ne0, ne1,
                                 eps_bits);
  ASSERT_EQ(err, 0) << "htp_ops_rms_norm_f32 failed";

  std::vector<float> dsp_dst(ne0 * ne1);
  memcpy(dsp_dst.data(), dst_ptr, ne0 * ne1 * sizeof(float));

  float mse_err = mse<float>(dsp_dst.data(), ref_dst.data(), ne0 * ne1);
  std::cout << "RMS Norm F32: ne0=" << ne0 << ", ne1=" << ne1
            << ", eps=" << eps << std::endl;
  std::cout << " - MSE (vs CPU ref): " << mse_err << std::endl;

  EXPECT_IN_RANGE(mse_err, 0.0f, 1e-6f);

  htp.free_shared_mem_buf(dst_ptr, dst_fd, buf_size);
  htp.free_shared_mem_buf(src_ptr, src_fd, buf_size);
}

#define DECLARE_rms_norm_f32_test(NE0, NE1, EPS_TAG, EPS_VAL)                 \
  TEST(nntrainer_htp_rms_norm, rms_norm_f32_##NE0##_##NE1##_##EPS_TAG) {      \
    run_rms_norm_f32_test(NE0, NE1, EPS_VAL);                                 \
  }

// Representative subset covering: tiny vector, large vector, batched,
// non-power-of-2 ne0, and alternate epsilon (Qwen3 default 1e-6 vs LLaMA 1e-5).
DECLARE_rms_norm_f32_test(32, 1, eps1e5, 1e-5f);    // tiny
DECLARE_rms_norm_f32_test(4096, 1, eps1e5, 1e-5f);  // large
DECLARE_rms_norm_f32_test(4096, 16, eps1e5, 1e-5f); // batched
DECLARE_rms_norm_f32_test(100, 1, eps1e5, 1e-5f);   // non-power-of-2
DECLARE_rms_norm_f32_test(4096, 1, eps1e6, 1e-6f);  // alt eps

#else

TEST(nntrainer_htp_rms_norm, htp_not_enabled) {
  GTEST_SKIP() << "HTP is not enabled";
}

#endif // ENABLE_HTP
