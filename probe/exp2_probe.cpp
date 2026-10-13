#include <cstdio>
#include "hvx_softmax_q.h"
int main() {
  for (uint32_t F = 7; F <= 9; ++F)
    for (uint32_t f = 0; f < 4; ++f)
      std::printf("F=%u f=%u -> %u\n", F, f, (unsigned)hvx_softmax_q_exp2_q14(f, F));
  std::printf("exp2(100,9)=%u exp2(511,9)=%u\n", (unsigned)hvx_softmax_q_exp2_q14(100, 9), (unsigned)hvx_softmax_q_exp2_q14(511, 9));
  int k = 0; uint16_t rho = 0;
  std::printf("scale rc=%d k=%d rho=%u\n", hvx_softmax_q_scale(1.0f / 2000.0f, 9, &k, &rho), k, (unsigned)rho);
  std::printf("sizeof(long)=%u sizeof(block)=%u\n", (unsigned)sizeof(long), (unsigned)sizeof(hvx_softmax_q_block));
  return 0;
}
