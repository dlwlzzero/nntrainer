// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_moe_cpu.h
 * @date   29 Sep 2026
 * @brief  The CPU's share of a split M=1 MoE call (#157): moe_m1_det.h's
 *         experts on the ThreadManager pool
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HTP_MOE_CPU_H__
#define __NNTRAINER_HTP_MOE_CPU_H__

#include <cstdint>
#include <vector>

#include <moe_m1_det.h>

namespace nntrainer {

/**
 * @brief Scratch and driver for the CPU experts of one split call. Every
 *        result bit is moe_m1_expert()'s: the pool only picks which
 *        columns each thread computes, and every column is independent.
 *        Its own translation unit so the aarch64 object can be counted
 *        for fmla / fmls (plan 157 section 4 step 3: there must be none).
 */
class HtpMoeCpu {
public:
  /**
   * @brief Expert w[i] on the token row act[K], for i < n, on the
   *        ThreadManager pool (the caller helps); the results stay here
   *        for merge().
   */
  void run(const float *act, uint32_t K, uint32_t inter, uint32_t N_out,
           const moe_m1_weights *w, uint32_t n);

  /**
   * @brief out = out + res_i * weight[i], i = 0 .. n-1 in order, over the
   *        last run's n experts: the DSP's scatter continued (step 8).
   */
  void merge(float *out, const float *weight) const;

private:
  std::vector<uint8_t> q_, mid_;
  std::vector<int8_t> qs_, mid_s_;
  std::vector<float> gate_, rs_, res_;
  uint32_t n_ = 0, n_out_ = 0;
  std::vector<int32_t> rz_;
};

} // namespace nntrainer

#endif // __NNTRAINER_HTP_MOE_CPU_H__
