// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_hexagon_protos.h
 * @date   27 Sep 2026
 * @brief  Host emulation of the HVX intrinsics the M=1 small-op and
 *         decode-attention kernels use, one IEEE f32 operation per lane
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * THE PREMISE: one Vsf multiply / add / subtract is one IEEE-754 f32
 * operation, round to nearest even, subnormals kept. That is what
 * HvxSwigluDet.MatchesScalarBitExact confirmed on the device (rule 24), and
 * what HvxM1Ops.* re-checks for these kernels. Each lane below stores
 * through a volatile so the host compiler cannot contract or reassociate.
 * The integer and bitwise ops, the byte rotate, the predicate ops and the
 * mux move bits exactly; vmin / vmax on finite values are exact by definition,
 * and Vsf_equals_Vw is one exactly representable int -> f32 conversion for |w|
 * <= 2^24 (plan 81 section 3.3 adds these for the attention kernel). What this
 * emulation cannot see: an aligned-load fault (the small ops use HVX_UVector
 * everywhere; the attention kernel's cache loads are aligned
 * -- a review item), inf/NaN encodings, and timing.
 *
 * One scalar-core op is emulated too (#164): Q6_R_sfmpyacc_RR, the sffma
 * instruction, is fmaf -- one IEEE fused multiply-add, which the ISS
 * matched bit for bit on 784 RMSNORM rows and two subnormal-sum probes
 * (plan 164 section 0); HvxM1Ops.* re-checks it on silicon.
 *
 * vror follows the PRM: Vd.ub[i] = Vu.ub[(i + Rt) mod 128]. The reduction
 * that uses it gives the same bits in either direction (IEEE add
 * commutes), which m1_ops_det.h's comment spells out.
 */

#ifndef __NNTRAINER_HVX_EMU_HVX_HEXAGON_PROTOS_H__
#define __NNTRAINER_HVX_EMU_HVX_HEXAGON_PROTOS_H__

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "hexagon_types.h"

static inline float hvx_emu_f(int32_t w) {
  float f;
  memcpy(&f, &w, sizeof(f));
  return f;
}
static inline int32_t hvx_emu_w(float f) {
  int32_t w;
  memcpy(&w, &f, sizeof(w));
  return w;
}

/** @brief sffma: x + s * t with one rounding (the SDK's
 *         __builtin_HEXAGON_F2_sffma, accumulator first). */
static inline float Q6_R_sfmpyacc_RR(float x, float s, float t) {
  volatile float r = fmaf(s, t, x);
  return r;
}

static inline HVX_Vector Q6_V_vzero(void) {
  HVX_Vector r;
  memset(&r, 0, sizeof(r));
  return r;
}

static inline HVX_Vector Q6_V_vsplat_R(int32_t x) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = x;
  }
  return r;
}

#define HVX_EMU_SF_BINOP(name, op)                                             \
  static inline HVX_Vector name(HVX_Vector a, HVX_Vector b) {                  \
    HVX_Vector r;                                                              \
    for (int i = 0; i < HVX_EMU_LANES; ++i) {                                  \
      volatile float t = hvx_emu_f(a.w[i]) op hvx_emu_f(b.w[i]);               \
      r.w[i] = hvx_emu_w(t);                                                   \
    }                                                                          \
    return r;                                                                  \
  }
HVX_EMU_SF_BINOP(Q6_Vsf_vadd_VsfVsf, +)
HVX_EMU_SF_BINOP(Q6_Vsf_vsub_VsfVsf, -)
HVX_EMU_SF_BINOP(Q6_Vsf_vmpy_VsfVsf, *)
#undef HVX_EMU_SF_BINOP

/* max / min: exact on finite values (the kernels keep every lane finite;
   the mask fill is -FLT_MAX, not -inf). Equal values are the same bits
   except +-0, which attn_m1_det.h shows cannot both occur in a score. */
static inline HVX_Vector Q6_Vsf_vmax_VsfVsf(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    const float x = hvx_emu_f(a.w[i]), y = hvx_emu_f(b.w[i]);
    r.w[i] = (x > y) ? a.w[i] : b.w[i];
  }
  return r;
}
static inline HVX_Vector Q6_Vsf_vmin_VsfVsf(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    const float x = hvx_emu_f(a.w[i]), y = hvx_emu_f(b.w[i]);
    r.w[i] = (x < y) ? a.w[i] : b.w[i];
  }
  return r;
}

/* int32 -> f32, numeric (hvx_convert.h: NOT a reinterpret on v79). Exact
   for |w| <= 2^24; hvx_exp_det_sf's k is within [-127, 123]. */
static inline HVX_Vector Q6_Vsf_equals_Vw(HVX_Vector a) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = hvx_emu_w((float)a.w[i]);
  }
  return r;
}

#define HVX_EMU_W_BINOP(name, op)                                              \
  static inline HVX_Vector name(HVX_Vector a, HVX_Vector b) {                  \
    HVX_Vector r;                                                              \
    for (int i = 0; i < HVX_EMU_LANES; ++i) {                                  \
      r.w[i] = (int32_t)((uint32_t)a.w[i] op(uint32_t) b.w[i]);                \
    }                                                                          \
    return r;                                                                  \
  }
HVX_EMU_W_BINOP(Q6_Vw_vadd_VwVw, +)
HVX_EMU_W_BINOP(Q6_Vw_vsub_VwVw, -)
#undef HVX_EMU_W_BINOP

/* Bitwise ops and signed word max / min / arithmetic shift (#152: the
   fp16 rounding and the round-to-odd step); exact by definition. */
#define HVX_EMU_V_BITOP(name, op)                                              \
  static inline HVX_Vector name(HVX_Vector a, HVX_Vector b) {                  \
    HVX_Vector r;                                                              \
    for (int i = 0; i < HVX_EMU_LANES; ++i) {                                  \
      r.w[i] = a.w[i] op b.w[i];                                               \
    }                                                                          \
    return r;                                                                  \
  }
HVX_EMU_V_BITOP(Q6_V_vand_VV, &)
HVX_EMU_V_BITOP(Q6_V_vor_VV, |)
HVX_EMU_V_BITOP(Q6_V_vxor_VV, ^)
#undef HVX_EMU_V_BITOP

static inline HVX_Vector Q6_Vw_vmax_VwVw(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = a.w[i] > b.w[i] ? a.w[i] : b.w[i];
  }
  return r;
}
static inline HVX_Vector Q6_Vw_vmin_VwVw(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = a.w[i] < b.w[i] ? a.w[i] : b.w[i];
  }
  return r;
}

static inline HVX_Vector Q6_Vw_vasr_VwR(HVX_Vector a, int32_t n) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = a.w[i] >> (n & 31); /* arithmetic on every host compiler */
  }
  return r;
}

static inline HVX_Vector Q6_Vuw_vlsr_VuwR(HVX_Vector a, int32_t n) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = (int32_t)((uint32_t)a.w[i] >> (n & 31));
  }
  return r;
}

static inline HVX_Vector Q6_Vw_vasl_VwR(HVX_Vector a, int32_t n) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = (int32_t)((uint32_t)a.w[i] << (n & 31));
  }
  return r;
}

/* Vx += Vu << Rt, word lanes, wrapping. */
static inline HVX_Vector Q6_Vw_vaslacc_VwVwR(HVX_Vector x, HVX_Vector u,
                                             int32_t n) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = (int32_t)((uint32_t)x.w[i] + ((uint32_t)u.w[i] << (n & 31)));
  }
  return r;
}

/* Predicates: one flag per byte; the word compare sets a lane's 4 bytes. */
static inline HVX_VectorPred Q6_Q_vcmp_gt_VwVw(HVX_Vector a, HVX_Vector b) {
  HVX_VectorPred q;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    const uint8_t f = (a.w[i] > b.w[i]) ? 1u : 0u;
    memset(q.q + 4 * i, f, 4);
  }
  return q;
}

static inline HVX_VectorPred Q6_Q_vcmp_eq_VwVw(HVX_Vector a, HVX_Vector b) {
  HVX_VectorPred q;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    const uint8_t f = (a.w[i] == b.w[i]) ? 1u : 0u;
    memset(q.q + 4 * i, f, 4);
  }
  return q;
}

/* Predicate logic, per byte: Qs & Qt, Qs | Qt and Qs & ~Qt. */
static inline HVX_VectorPred Q6_Q_and_QQ(HVX_VectorPred a, HVX_VectorPred b) {
  HVX_VectorPred q;
  for (int i = 0; i < 4 * HVX_EMU_LANES; ++i) {
    q.q[i] = a.q[i] & b.q[i];
  }
  return q;
}
static inline HVX_VectorPred Q6_Q_or_QQ(HVX_VectorPred a, HVX_VectorPred b) {
  HVX_VectorPred q;
  for (int i = 0; i < 4 * HVX_EMU_LANES; ++i) {
    q.q[i] = a.q[i] | b.q[i];
  }
  return q;
}
static inline HVX_VectorPred Q6_Q_and_QQn(HVX_VectorPred a, HVX_VectorPred b) {
  HVX_VectorPred q;
  for (int i = 0; i < 4 * HVX_EMU_LANES; ++i) {
    q.q[i] = a.q[i] & (uint8_t)!b.q[i];
  }
  return q;
}

/* vsetq2(Rt): bytes 0 .. ((Rt - 1) & 127) set, so 128 (and 0) set all --
   unlike vsetq, whose Rt = 128 wraps to none. The kernels pass 4..128. */
static inline HVX_VectorPred Q6_Q_vsetq2_R(int32_t n) {
  HVX_VectorPred q;
  const int last = (n - 1) & (4 * HVX_EMU_LANES - 1);
  for (int i = 0; i < 4 * HVX_EMU_LANES; ++i) {
    q.q[i] = (i <= last) ? 1u : 0u;
  }
  return q;
}

/* Vd = Qt ? Vu : Vv, per byte. */
static inline HVX_Vector Q6_V_vmux_QVV(HVX_VectorPred q, HVX_Vector a,
                                       HVX_Vector b) {
  HVX_Vector r;
  const uint8_t *pa = (const uint8_t *)a.w, *pb = (const uint8_t *)b.w;
  uint8_t *pr = (uint8_t *)r.w;
  for (int i = 0; i < 4 * HVX_EMU_LANES; ++i) {
    pr[i] = q.q[i] ? pa[i] : pb[i];
  }
  return r;
}

static inline HVX_Vector Q6_V_vror_VR(HVX_Vector a, int32_t bytes) {
  HVX_Vector r;
  const uint8_t *src = (const uint8_t *)a.w;
  uint8_t *dst = (uint8_t *)r.w;
  for (int i = 0; i < 4 * HVX_EMU_LANES; ++i) {
    dst[i] = src[(i + bytes) & (4 * HVX_EMU_LANES - 1)];
  }
  return r;
}

static inline int32_t Q6_R_vextract_VR(HVX_Vector a, int32_t byte_off) {
  int32_t w;
  memcpy(&w, (const uint8_t *)a.w + (byte_off & (4 * HVX_EMU_LANES - 1)),
         sizeof(w));
  return w;
}

/* [#132 PR 2] the Q4_0 GEMV's integer part (hvx_q4_gemv_f32.c). */

/* Vd.uh[i] = Vu.uh[i] >> Rt, halfword lanes. */
static inline HVX_Vector Q6_Vuh_vlsr_VuhR(HVX_Vector a, int32_t n) {
  HVX_Vector r;
  uint16_t h[2 * HVX_EMU_LANES];
  memcpy(h, a.w, sizeof(h));
  for (int i = 0; i < 2 * HVX_EMU_LANES; ++i) {
    h[i] = (uint16_t)(h[i] >> (n & 15));
  }
  memcpy(r.w, h, sizeof(h));
  return r;
}

/* Vx.w[i] += sum_j Vu.ub[4i + j] * Rt.b[j]. */
static inline HVX_Vector Q6_Vw_vrmpyacc_VwVubRb(HVX_Vector x, HVX_Vector u,
                                                int32_t rt) {
  HVX_Vector r;
  const uint8_t *pu = (const uint8_t *)u.w;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    int32_t s = 0;
    for (int j = 0; j < 4; ++j) {
      s += (int32_t)pu[4 * i + j] * (int32_t)(int8_t)(rt >> (8 * j));
    }
    r.w[i] = (int32_t)((uint32_t)x.w[i] + (uint32_t)s);
  }
  return r;
}

/* Vd.w[i] = Vu.w[i] * Vv.uh[2i] (the even unsigned halfword), low 32. */
static inline HVX_Vector Q6_Vw_vmpyie_VwVuh(HVX_Vector a, HVX_Vector b) {
  HVX_Vector r;
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.w[i] = (int32_t)((uint32_t)a.w[i] * (uint32_t)(b.w[i] & 0xffff));
  }
  return r;
}

/* Vdd.w = vunpack(Vu.h): sign-extended, in order -- lo gets halfwords
   0..31, hi 32..63 (the prototype's d layout matched the v79 ISS so). */
static inline HVX_VectorPair Q6_Ww_vunpack_Vh(HVX_Vector a) {
  HVX_VectorPair r;
  int16_t h[2 * HVX_EMU_LANES];
  memcpy(h, a.w, sizeof(h));
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    r.v[0].w[i] = h[i];
    r.v[1].w[i] = h[HVX_EMU_LANES + i];
  }
  return r;
}
static inline HVX_Vector Q6_V_lo_W(HVX_VectorPair p) { return p.v[0]; }
static inline HVX_Vector Q6_V_hi_W(HVX_VectorPair p) { return p.v[1]; }

/* [#132 PR 2] the quantizer's byte pack: vpacke(u, v) keeps the even (low)
   half of each element, v's into the low half of the result, u's into the
   high half (checked against the SDK's libnative). */
static inline HVX_Vector Q6_Vh_vpacke_VwVw(HVX_Vector u, HVX_Vector v) {
  HVX_Vector r;
  uint16_t h[2 * HVX_EMU_LANES];
  for (int i = 0; i < HVX_EMU_LANES; ++i) {
    h[i] = (uint16_t)v.w[i];
    h[HVX_EMU_LANES + i] = (uint16_t)u.w[i];
  }
  memcpy(r.w, h, sizeof(h));
  return r;
}
static inline HVX_Vector Q6_Vb_vpacke_VhVh(HVX_Vector u, HVX_Vector v) {
  HVX_Vector r;
  uint16_t hu[2 * HVX_EMU_LANES], hv[2 * HVX_EMU_LANES];
  uint8_t b[4 * HVX_EMU_LANES];
  memcpy(hu, u.w, sizeof(hu));
  memcpy(hv, v.w, sizeof(hv));
  for (int i = 0; i < 2 * HVX_EMU_LANES; ++i) {
    b[i] = (uint8_t)hv[i];
    b[2 * HVX_EMU_LANES + i] = (uint8_t)hu[i];
  }
  memcpy(r.w, b, sizeof(b));
  return r;
}

#endif /* __NNTRAINER_HVX_EMU_HVX_HEXAGON_PROTOS_H__ */
