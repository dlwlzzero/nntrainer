// x86 self-test for htp_trace.cpp: the call shapes of one LFM2 prefill and
// three decode tokens, laid on the real clock (sleeps between tokens) so the
// thermal sampler (htp_thermal.cpp, every host zone, 100 ms) lands inside
// the phases. Writes selftest_trace.json; test_converters.py reads the copy
// in test/data/htp_trace_selftest.json.
#include "htp_trace.h"
#include <chrono>
#include <cstdlib>
#include <thread>
using nntrainer::HtpTrace;
static void nap(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
int main() {
  setenv("NNTR_TRACE", "selftest_trace.json", 1);
  setenv("NNTR_TRACE_THERMAL", "*", 1);
  setenv("NNTR_TRACE_THERMAL_MS", "100", 1);
  HtpTrace &t = HtpTrace::global();
  t.setMeta(2, 2);
  // registration
  t.registration(HtpTrace::nowUs(), 1500, 700, 90, 2048, 2048);
  nap(5);
  // prefill phase with one MoE call, one conv call and one FC call
  const uint64_t p0 = HtpTrace::nowUs();
  uint32_t moe[19] = {14916, 274, 3300, 905, 2927, 171, 53, 126, 1060, 1035, 8606, 164786, 0, 107, 1024, 60, 57, 383, 32};
  t.staging(HtpTrace::nowUs(), 40, 1 << 20);
  t.call(HtpTrace::KIND_MOE, 444, 2048, 2048, 4, HtpTrace::nowUs(), 17296, moe, 19, 444 * 2048 * 4, 444 * 2048 * 4);
  nap(18);
  uint32_t conv[19] = {4000, 120, 3300, 300, 400, 80, 0, 0, 0, 64, 2500, 0, 0, 30, 0, 0, 20, 60, 32};
  t.call(HtpTrace::KIND_CONV, 444, 2048, 2048, 2, HtpTrace::nowUs(), 4700, conv, 19, 444 * 2048 * 4, 444 * 2048 * 4);
  nap(5);
  uint32_t fc[7] = {3202, 357, 794, 501, 0, 58, 32};
  t.call(HtpTrace::KIND_FC, 444, 2048, 6144, 1, HtpTrace::nowUs(), 3984, fc, 7, 444 * 2048 * 4, 444 * 6144 * 4);
  nap(120);
  t.phase(p0, HtpTrace::nowUs() - p0, 0, 444);
  // three decode tokens, ~120 ms apart
  for (int tok = 0; tok < 3; ++tok) {
    const uint64_t t0 = HtpTrace::nowUs();
    for (int l = 0; l < 4; ++l) {
      uint32_t d[19] = {1354, 15, 120, 13, 259, 106, 1, 139, 42, 4, 748, 21504, 0, 0, 1024, 8, 3, 4, 32};
      t.call(HtpTrace::KIND_MOE, 1, 2048, 2048, 4, HtpTrace::nowUs(), 2169, d, 19, 8192, 8192);
      nap(3);
      t.call(HtpTrace::KIND_GATE_UP, 1, 2048, 12288, 1, HtpTrace::nowUs(), 900, nullptr, 0, 8192, 12288);
      nap(2);
    }
    nap(100);
    t.phase(t0, HtpTrace::nowUs() - t0, 444 + tok, 445 + tok);
  }
  t.write();
  return 0;
}
