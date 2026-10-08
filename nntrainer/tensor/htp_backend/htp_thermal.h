// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon Lee <dlwlzzero@gmail.com>
 *
 * @file   htp_thermal.h
 * @date   8 Oct 2026
 * @brief  Samples /sys/class/thermal while an HtpTrace run is on
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * A production Snapdragon phone exposes no per-rail energy counter to a
 * shell user (no powercap, no PowerStats HAL, /sys/class/power_supply
 * denied), but it does expose every thermal zone (NPU HMX/HVX, CPU
 * clusters, GPU, DDR, battery, PMIC) and the cooling devices the kernel
 * throttles with, all readable and updating sub-second. This reads a
 * selected set of them on a background thread and hands the samples to
 * HtpTrace::write(), which emits them as Chrome-trace counter tracks
 * (pid 3) next to the prefill/token spans. Temperature is a proxy for
 * which block is working and when throttling starts; it is not energy.
 *
 * Selection: NNTR_TRACE_THERMAL is a comma-separated list of fnmatch
 * patterns matched against the `type` of thermal_zone* and
 * cooling_device* entries ("*" takes everything). The default list is the
 * S25 Ultra set that matters for decode: nsphmx-*, nsphvx-*, cpuss-*,
 * gpuss-0, ddr, battery, cdsp, cdsp_sw_hmx, cdsp_sw_hvx, cpu-cluster*,
 * cpufreq-cpu*, gpu. NNTR_TRACE_THERMAL_MS sets the period (default 200).
 *
 * ponytail: battery current is not sampled here because the process cannot
 * read /sys/class/power_supply on a production device; `cmd battery` from
 * a shell script is the only route and lives outside this process.
 */

#ifndef __HTP_THERMAL_H__
#define __HTP_THERMAL_H__
#ifdef __cplusplus

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nntrainer {

class HtpThermal {
public:
  struct Source {
    std::string name; /**< "temp <type>" (milli-degC) or "throttle <type>" */
    int fd;
    bool is_temp;
  };
  struct Sample {
    uint64_t t_us; /**< HtpTrace::nowUs() */
    uint32_t src;  /**< index into sources() */
    int32_t value; /**< raw: milli-degC, or cooling cur_state */
  };

  HtpThermal() = default;
  ~HtpThermal();
  HtpThermal(const HtpThermal &) = delete;
  HtpThermal &operator=(const HtpThermal &) = delete;

  /** @brief Opens the selected sysfs nodes and starts sampling. No-op when
   *  nothing matches (then sources() is empty and nothing is written). */
  void start();
  /** @brief Stops the thread. Samples stay readable; idempotent. */
  void stop();

  const std::vector<Source> &sources() const { return sources_; }
  const std::vector<Sample> &samples() const { return samples_; }
  unsigned intervalMs() const { return interval_ms_; }
  uint64_t dropped() const { return dropped_; }

private:
  void open();
  void loop();
  bool readOne(unsigned i, int32_t &v);

  static constexpr size_t kMaxSamples = 1u << 20; /**< 12 MB */
  std::vector<Source> sources_;
  std::vector<Sample> samples_;
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool started_ = false;
  bool stop_ = false;
  uint64_t dropped_ = 0;
  unsigned interval_ms_ = 200;
};

} // namespace nntrainer

#endif // __cplusplus
#endif // __HTP_THERMAL_H__
