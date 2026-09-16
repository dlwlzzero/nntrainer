// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hvx-quant.h
 * @date	18 August 2026
 * @brief	Per-token dynamic int8 quantization (HVX; the vrmpy tile
 *		kernel lives in ops/hvx-matmul.c)
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef HVX_QUANT_H
#define HVX_QUANT_H

#include <stdint.h>

#include "hvx-base.h"

/* Per-token quantization of one row (k is a multiple of 128, enforced by
 * nntr_htp_oplist_validate); rows are spread over the worker pool by the
 * caller. Returns scale (absmax / 127). The scalar definition it implements
 * is ref_quant_row() in test/hexagon/sim/ref_ops.c. */
static inline float htp_quant_row_fp16(const __fp16 *x, int8_t *q, uint32_t k) {
  /* absmax: sign-cleared fp16 bits are monotonic in |x|, so an unsigned max
   * over them is the fp16 absmax exactly. */
  HVX_Vector vmax = Q6_V_vzero();
  for (uint32_t i = 0; i < k; i += 64u)
    vmax = Q6_Vuh_vmax_VuhVuh(vmax, hvx_vec_abs_f16(hvx_vmemu(x + i)));
  for (int s = 2; s < 128; s <<= 1)
    vmax = Q6_Vuh_vmax_VuhVuh(vmax, Q6_V_vror_VR(vmax, s));
  const float amax = (float)hvx_vec_get_f16(vmax);
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  /* q = lrintf(x * inv): qf32 product, then round-to-nearest-even, then two
   * saturating packs. Adding the 1.5*2^8 magic puts the product p in
   * [2^8, 2^9) where one sf ulp is 2^-15, so (sum - magic) read as an integer
   * is p in 2^-15 units. qf32 rounds down to a grid one bit coarser than sf
   * and then reports half of it, so that integer is 2*floor(p*2^14)+1 and a
   * shift toward zero recovers |p| floored at 2^-14 (toward zero, because the
   * qf32 product itself is half an ulp away from zero). The integer
   * "+half-1+lsb" step is then ties-to-even like lrintf. The rounding
   * resolution is therefore 2^-14: a true product landing within 2^-14 above a
   * .5 tie rounds to even instead of up, which makes about 3.6e-5 of the
   * elements differ from ref_quant_row by one LSB. That 3.6e-5 figure is a
   * modelled/theoretical rate over uniformly random ties; test_quant's
   * frand-generated data is dyadic, so near-ties in that test always land on
   * an odd n and are not truly random, which is why the test's observed
   * pm1=0/65536 is expected rather than a contradiction of the ~3.6e-5
   * figure. A NaN input also counts as the absmax here (its sign-cleared
   * bits exceed every finite one) while the scalar reference skips it. */
  const HVX_Vector vinv = hvx_vec_splat_f32(inv);
  const HVX_Vector magic = Q6_V_vsplat_R(0x43C00000);
  const HVX_Vector one = Q6_V_vsplat_R(1);
  const HVX_Vector half1 = Q6_V_vsplat_R((1 << 13) - 1);
  for (uint32_t i = 0; i < k; i += 128u) {
    HVX_Vector h[2];
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
        a = Q6_Vw_vadd_VwVw(a, Q6_V_vand_VV(Q6_Vw_vasr_VwR(d, 14), one));
        w[e] = Q6_Vw_vasr_VwR(a, 14);
      }
      h[j] = Q6_Vh_vpack_VwVw_sat(w[1], w[0]);
    }
    hvx_vmemu(q + i) = Q6_Vb_vpack_VhVh_sat(h[1], h[0]);
  }
  return amax / 127.f;
}

#endif /* HVX_QUANT_H */
