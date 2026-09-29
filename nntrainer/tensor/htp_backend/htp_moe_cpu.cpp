// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_moe_cpu.cpp
 * @date   29 Sep 2026
 * @brief  The CPU's share of a split M=1 MoE call (#157)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <htp_moe_cpu.h>

#include <thread_manager.h>

namespace nntrainer {

void HtpMoeCpu::run(const float *act, uint32_t K, uint32_t inter,
                    uint32_t N_out, const moe_m1_weights *w, uint32_t n) {
  n_ = n;
  n_out_ = N_out;
  res_.resize(static_cast<size_t>(n) * N_out);
  float *res = res_.data();
  q_.resize(K);
  qs_.resize(K);
  gate_.resize(static_cast<size_t>(n) * inter);
  mid_.resize(static_cast<size_t>(n) * inter);
  mid_s_.resize(static_cast<size_t>(n) * inter);
  float s = 1.0f;
  int32_t z = 0;
  moe_m1_act(act, K, &s, &z, q_.data(), qs_.data());

  ThreadManager &tm = ThreadManager::Global();
  const size_t T = tm.getComputeThreadCount();
  const uint32_t inter_nt = inter / 32u, dn_nt = N_out / 32u;
  // Contiguous column ranges, T per expert: each thread streams adjacent
  // tiles of one k-tile row at a time (moe_m1_gemv_block).
  tm.parallel_for(0, n * T, [&](size_t i) {
    const size_t e = i / T, p = i % T;
    moe_m1_gate_range(&w[e], K, inter, q_.data(), qs_.data(), s, z,
                      static_cast<uint32_t>(inter_nt * p / T),
                      static_cast<uint32_t>(inter_nt * (p + 1) / T),
                      gate_.data() + e * inter);
  });
  rs_.resize(n);
  rz_.resize(n);
  tm.parallel_for(0, n, [&](size_t e) {
    moe_m1_act(gate_.data() + e * inter, inter, &rs_[e], &rz_[e],
               mid_.data() + e * inter, mid_s_.data() + e * inter);
  });
  tm.parallel_for(0, n * T, [&](size_t i) {
    const size_t e = i / T, p = i % T;
    moe_m1_down_range(
      &w[e], inter, N_out, mid_.data() + e * inter, mid_s_.data() + e * inter,
      rs_[e], rz_[e], static_cast<uint32_t>(dn_nt * p / T),
      static_cast<uint32_t>(dn_nt * (p + 1) / T), res + e * N_out);
  });
}

void HtpMoeCpu::merge(float *out, const float *weight) const {
  for (uint32_t i = 0; i < n_; ++i)
    moe_m1_scale_add(out, res_.data() + static_cast<size_t>(i) * n_out_,
                     weight[i], n_out_);
}

} // namespace nntrainer
