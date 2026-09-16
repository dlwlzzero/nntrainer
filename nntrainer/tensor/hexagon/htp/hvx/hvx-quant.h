// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-quant.h
 * @date	18 August 2026
 * @brief	Per-token dynamic int8 (W8A8) and int16 (W8A16) quantization
 *		(HVX; the vrmpy tile kernel lives in ops/hvx-matmul.c)
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef HVX_QUANT_H
#define HVX_QUANT_H

#include <stdbool.h>
#include <stdint.h>

#include "hvx-base.h"

/* Per-token quantization of one row (k is a multiple of 128, enforced by
 * nntr_htp_oplist_validate) into int8 (W8A8) or int16 (W8A16); rows are
 * spread over the worker pool by the caller. Returns scale (absmax / qmax).
 * The scalar definitions this implements are ref_quant_row() (int8) and
 * ref_quant_row_i16() (int16) in test/hexagon/sim/ref_ops.c.
 *
 * q = lrintf(x * inv): qf32 product, then round-to-nearest-even, then a
 * saturating pack. Adding the magic (1.5*2^8 for int8, 1.5*2^16 for int16)
 * puts the product p in [2^8, 2^9) or [2^16, 2^17) where one sf ulp is 2^-15
 * or 2^-7, so (sum - magic) read as an integer is p in those units. qf32
 * rounds down to a grid one bit coarser than sf and then reports half of it,
 * so that integer is 2*floor(p*2^sh)+1 and a shift toward zero recovers |p|
 * floored at 2^-sh (toward zero, because the qf32 product itself is half an
 * ulp away from zero), where sh is 14 for int8 and 6 for int16. The integer
 * "+half-1+lsb" step is then ties-to-even like lrintf. The rounding
 * resolution is therefore 2^-sh: a true product landing within 2^-sh above a
 * .5 tie rounds to even instead of up, which makes about 3.6e-5 of the int8
 * elements (and about 1% of the int16 elements) differ from the scalar
 * reference by one LSB. The int8 3.6e-5 figure is a modelled/theoretical rate
 * over uniformly random ties; test_quant's frand-generated data is dyadic, so
 * near-ties in that test always land on an odd n and are not truly random,
 * which is why the test's observed pm1=0/65536 is expected rather than a
 * contradiction of the ~3.6e-5 figure. A NaN input also counts as the absmax
 * here (its sign-cleared bits exceed every finite one) while the scalar
 * reference skips it. */
static inline __attribute__((always_inline)) float
quant_row(const __fp16 *x, void *q, uint32_t k, bool i8) {
  /* absmax: sign-cleared fp16 bits are monotonic in |x|, so an unsigned max
   * over them is the fp16 absmax exactly. */
  HVX_Vector vmax = Q6_V_vzero();
  for (uint32_t i = 0; i < k; i += 64u)
    vmax = Q6_Vuh_vmax_VuhVuh(vmax, hvx_vec_abs_f16(hvx_vmemu(x + i)));
  for (int s = 2; s < 128; s <<= 1)
    vmax = Q6_Vuh_vmax_VuhVuh(vmax, Q6_V_vror_VR(vmax, s));
  const float amax = (float)hvx_vec_get_f16(vmax);
  const float qmax = i8 ? 127.f : 32767.f;
  const float inv = amax > 0.f ? qmax / amax : 0.f;
  const HVX_Vector vinv = hvx_vec_splat_f32(inv);
  const HVX_Vector magic = Q6_V_vsplat_R(i8 ? 0x43C00000 : 0x47C00000);
  const HVX_Vector one = Q6_V_vsplat_R(1);
  const HVX_Vector half1 = Q6_V_vsplat_R((1 << (i8 ? 13 : 5)) - 1);
  for (uint32_t i = 0; i < k; i += 128u) {
    HVX_Vector hq[2];
    for (int j = 0; j < 2; ++j) {
      HVX_VectorPair p =
        hvx_vec_f16_to_f32(hvx_vmemu(x + i + 64u * (uint32_t)j));
      HVX_Vector w[2] = {Q6_V_lo_W(p), Q6_V_hi_W(p)};
      for (int e = 0; e < 2; ++e) {
        HVX_Vector f = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(w[e], vinv));
        f = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(f, magic));
        HVX_Vector d = Q6_Vw_vsub_VwVw(f, magic);
        d = Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwR(d, 1), Q6_Vuw_vlsr_VuwR(d, 31));
        HVX_Vector a = Q6_Vw_vadd_VwVw(d, half1);
        a = Q6_Vw_vadd_VwVw(
          a,
          Q6_V_vand_VV(i8 ? Q6_Vw_vasr_VwR(d, 14) : Q6_Vw_vasr_VwR(d, 6), one));
        w[e] = i8 ? Q6_Vw_vasr_VwR(a, 14) : Q6_Vw_vasr_VwR(a, 6);
      }
      HVX_Vector h = Q6_Vh_vpack_VwVw_sat(w[1], w[0]);
      if (i8)
        hq[j] = h;
      else
        hvx_vmemu((int16_t *)q + i + 64u * (uint32_t)j) = h;
    }
    if (i8)
      hvx_vmemu((int8_t *)q + i) = Q6_Vb_vpack_VhVh_sat(hq[1], hq[0]);
  }
  return amax / qmax;
}

static inline float htp_quant_row_fp16(const __fp16 *x, int8_t *q, uint32_t k) {
  return quant_row(x, q, k, true);
}
static inline float htp_quant_row_fp16_i16(const __fp16 *x, int16_t *q,
                                           uint32_t k) {
  return quant_row(x, q, k, false);
}

#endif /* HVX_QUANT_H */
