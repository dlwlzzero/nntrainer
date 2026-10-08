// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon Lee <dlwlzzero@gmail.com>
 *
 * @file   htp_thermal.cpp
 * @date   8 Oct 2026
 * @brief  Samples /sys/class/thermal while an HtpTrace run is on
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include "htp_thermal.h"
#include "htp_trace.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <dirent.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <unistd.h>
#endif

namespace nntrainer {

namespace {

const char *kDefaultPatterns = "nsphmx-*,nsphvx-*,cpuss-*,gpuss-0,ddr,battery,"
                               "cdsp,cdsp_sw_hmx,cdsp_sw_hvx,cpu-cluster*,"
                               "cpufreq-cpu*,gpu";

std::vector<std::string> split(const char *s) {
  std::vector<std::string> out;
  while (s && *s) {
    const char *c = std::strchr(s, ',');
    out.emplace_back(s, c ? c - s : std::strlen(s));
    if (!c)
      break;
    s = c + 1;
  }
  return out;
}

#if defined(__linux__)
bool readFile(const std::string &path, char *buf, size_t n) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return false;
  ssize_t r = ::read(fd, buf, n - 1);
  ::close(fd);
  if (r <= 0)
    return false;
  buf[r] = '\0';
  while (r > 0 && (buf[r - 1] == '\n' || buf[r - 1] == ' '))
    buf[--r] = '\0';
  return true;
}
#endif

} // namespace

HtpThermal::~HtpThermal() {
  stop();
  for (Source &s : sources_)
    if (s.fd >= 0) {
#if defined(__linux__)
      ::close(s.fd);
#endif
      s.fd = -1;
    }
}

void HtpThermal::open() {
#if defined(__linux__)
  const char *env = std::getenv("NNTR_TRACE_THERMAL");
  const std::vector<std::string> pats =
    split(env && *env ? env : kDefaultPatterns);
  const char *ms = std::getenv("NNTR_TRACE_THERMAL_MS");
  if (ms && std::atoi(ms) > 0)
    interval_ms_ = static_cast<unsigned>(std::atoi(ms));

  DIR *d = ::opendir("/sys/class/thermal");
  if (!d)
    return;
  char buf[64];
  while (dirent *e = ::readdir(d)) {
    const bool zone = std::strncmp(e->d_name, "thermal_zone", 12) == 0;
    const bool cool = std::strncmp(e->d_name, "cooling_device", 14) == 0;
    if (!zone && !cool)
      continue;
    const std::string dir = std::string("/sys/class/thermal/") + e->d_name;
    if (!readFile(dir + "/type", buf, sizeof(buf)))
      continue;
    bool hit = false;
    for (const std::string &p : pats)
      if (::fnmatch(p.c_str(), buf, 0) == 0) {
        hit = true;
        break;
      }
    if (!hit)
      continue;
    const int fd = ::open((dir + (zone ? "/temp" : "/cur_state")).c_str(),
                          O_RDONLY | O_CLOEXEC);
    if (fd < 0)
      continue;
    std::string name = std::string(zone ? "temp " : "throttle ") + buf;
    // `type` is not unique (x86 lists one "Processor" cooling device per
    // core); the counter name must be, or the samples merge into one track.
    unsigned dup = 0;
    for (const Source &s : sources_)
      if (s.name == name ||
          s.name.compare(0, name.size() + 2, name + " #") == 0)
        ++dup;
    if (dup)
      name += " #" + std::to_string(dup + 1);
    sources_.push_back(Source{name, fd, zone});
  }
  ::closedir(d);
#endif
}

bool HtpThermal::readOne(unsigned i, int32_t &v) {
#if defined(__linux__)
  char buf[32];
  const ssize_t r = ::pread(sources_[i].fd, buf, sizeof(buf) - 1, 0);
  if (r <= 0)
    return false;
  buf[r] = '\0';
  v = static_cast<int32_t>(std::atol(buf));
  return true;
#else
  (void)i;
  (void)v;
  return false;
#endif
}

void HtpThermal::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (started_)
    return;
  started_ = true;
  open();
  if (sources_.empty())
    return;
  samples_.reserve(1 << 14);
  thread_ = std::thread([this] { loop(); });
}

void HtpThermal::loop() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!stop_) {
    const uint64_t t = HtpTrace::nowUs();
    for (unsigned i = 0; i < sources_.size(); ++i) {
      int32_t v;
      if (!readOne(i, v))
        continue;
      if (samples_.size() >= kMaxSamples) {
        ++dropped_;
        continue;
      }
      samples_.push_back(Sample{t, i, v});
    }
    cv_.wait_for(lock, std::chrono::milliseconds(interval_ms_),
                 [this] { return stop_; });
  }
}

void HtpThermal::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!started_ || stop_)
      return;
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable())
    thread_.join();
}

} // namespace nntrainer
