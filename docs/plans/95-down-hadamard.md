# 95 — Hadamard rotation on the MoE down_proj input (`QS4CX_WH_HAD`)

Issue: dlwlzzero/nntrainer#95 (`hexagon`, `prio:p1`). Base branch
**`htp_hadamard`** @ `1ac18c39` (= upstream PR #4327 head `3006d255` + one
tooling commit). Work branch `htp/95-down-hadamard`, PR into `htp_hadamard`.
Contract: `git show htp_moe:docs/plans/0001-htp-moe-decode-agent-system.md`
(not on this branch). Line numbers below are from this tree and were read,
not guessed.

## 1. Goal and gate

From the issue: reduce the u8 requantization error of the expert down_proj
input (SwiGLU output, K = 1792) with a fixed block-256 Hadamard rotation:
`h · W = (h · H/16) · (Hᵀ · W/16)`, `Hᵀ·W/16` folded offline into a new
dtype `QS4CX_WH_HAD`, `h · H/16` computed on the DSP as an FWHT (IEEE `sf`
add/sub only) right before the existing u8 requant. **No speed gate is set;
"no measurable effect" closes the issue with the numbers recorded.**

Verdict metrics, on (variant C) vs off (variant B), same sitting:

| # | metric | where it is read | pass / result |
|---|---|---|---|
| 0 | `NNTR_PPL=1` teacher-forced perplexity of the 512-token prompt (`causal_lm.cpp:530-626`, `[PPL] ... ppl=`), CPU `q40`, NPU WH, NPU WH_HAD | handoff table col. `ppl` | result only; the user decides. Sanity: PPL(B) == PPL(A) (same weights, same arithmetic) |
| 1 | per-expert-call **requant SNR** from `NNTR_L2_DIFF` on the MoE-layer path (§3.4): device output vs an f32 reference that uses the device's own gate_up output, host `swiglu_det`, host `fwht_rows_f32_ref` when rotated, and an **unquantized** f32 down matmul | `[L2-DIFF-MOE]` lines, summarised as min / median / count < 100 dB over one prefill | result only |
| 2 | prefill tok/s, decode tok/s (gen 64 / 512 / 1024), `[HTP-PROFILE]` per-stage µs (the `requant` column carries the FWHT) | handoff table | **standing gate**: prefill(C) ≥ 0.95 × prefill(A); decode within noise |
| 3 | text identical to the CPU `q40` run (y/n, first differing token) | handoff table | B: identical to A bit for bit (flag off = untouched arithmetic). C: reported, **not gated** — a rotated int4 weight is a different quantization, so identity with `q40` is not a property C can be required to have; metric 0 is C's accuracy verdict |

Why metric 1 is not the issue's literal "per-call SNR + `total_flips`":
on this tree both sides of the split-call `l2Diff` already run the same
`swiglu_det` bits, and doc 44 §566/§614 records **`total_flips` 32/32 = 0,
`snr=999.00 dB max_abs_err=0` on all 32 calls**. The existing comparison
measures DSP-vs-reference agreement, which is already exact and cannot
move with the rotation. The quantity the rotation changes is the requant
noise against an *unquantized* `mid`, so the reference is redefined that
way (§3.4). Also, a `QS4CX_WH*` model never reaches `l2Diff`'s home
(`gemm_qs4cx_fused_swiglu_fp32`, `htp_compute_ops.cpp:958-985`): for WH
weights `tryMoeLayerOnAccelerator` (`lfm2_moe_layer.cpp:484-503`) returns
true at every M, so the diagnostic has to live on the MoE-layer path. The
5 "recorded bad calls" are identified only by M in a 2026-09-09 run of a
different code state and cannot be re-addressed by name; the plan reports
the distribution instead. **User may object (non-blocking, see §7).**

Standing gates (contract §1, §9): kernel bit-identical to its scalar spec
(device gtest, no simulator in this project), prefill ≥ −5 % of variant A,
`clang-format-14`, host checks, skel and app md5s recorded.

## 2. Where it lives

### 2.1 Requant sites on this tree (verified; the issue's "three" is htp_moe's count)

| site | file:line | runs when | change |
|---|---|---|---|
| HMX block loop | `hmx/hexkl_mm_u8i4_moe.c:1067-1076` (`hvx_quant_rows_u8_params((const float *)(vtcm_base + L.gate_off), m_blk, BR, inter, ...)` inside the `HEXKL_PROBE_REQUANT` window) | every MoE-layer call, prefill and decode (M=1 pads to one block, `m_blk=1`) | insert `hvx_fwht_rows_f32(gate_off, m_blk, inter, pool)` before it when the flag is set |
| prefill tail | `hmx/hexkl_mm_u8i4_moe.c:340-360` (`moe_tail_requant_unit`, `sh->gate_f32`) | only when `MOE_TAIL_MAX_ROWS > 0` — default 0 (`:275-276`), host check builds with 16 | same insert, `pool=NULL` (it already runs inside a pool unit) |
| M=1 GEMV | — | does not exist here (htp_moe #86) | none |

Outside the MoE-layer kernel, two more requants of the same intermediate
exist and are **out of scope**: `hmx/hexkl_mm_u8i4_dma.c:1023`
(`hexkl_mm_u8i4_gate_up_swiglu_run`, the split-call path, reachable only
with plain `QS4CX` and only when the MoE-layer path declined) and `:773`
(`hexkl_mm_u8i4_fused_run`, dormant). Neither can see a `QS4CX_WH_HAD`
weight: the fused-SwiGLU branch is gated on `getDataType() == QS4CX`
(`lfm2_moe_layer.cpp:667-668`). The kernel additionally refuses the flag
with `AEE_EBADPARM` if `inter % 256 != 0`, mirroring the converter.

Dense FFN: its dtype follows `fc_dtype`, never `moe_dtype`
(`quantize.cpp:554-557`; `dense_ffn_layer.cpp` uses `swiglu_det` and the
FC kernels), so it never carries `_HAD`. The loader distinguishes purely by
the per-weight tensor dtype: `transformer.cpp:479-497` registers each
`QS4CX`/`QS4CX_WH` weight and passes `weights_wh`; `lfm2_moe_layer.cpp:482-490`
requires every expert half of a layer to have the same dtype.

### 2.2 Files that change

| area | file:line | change |
|---|---|---|
| scalar spec (host + DSP) | **new** `nntrainer/tensor/fwht_det.h` (pattern: `nntrainer/tensor/swiglu_det.h`, included by the skel through `$BACKEND/..`, `test/htp/build.sh`) | `fwht_rows_f32_ref(float *x, uint32_t rows, uint32_t k)` and `FWHT_DET_BLOCK 256u` (with the `ponytail:` "lift to a parameter" note) |
| HVX kernel | **new** `nntrainer/tensor/htp_backend/hvx/hvx_fwht_f32.{h,c}` | `hvx_fwht_rows_f32(float *x, uint32_t rows, uint32_t k, hvx_worker_pool *pool)`, in place, `HVX_FWHT_BLOCK` = `FWHT_DET_BLOCK` |
| MoE kernel | `hmx/hexkl_mm_u8i4_moe.h:210-216` (`hexkl_mm_u8i4_moe_layer_run` signature), `:1067`, `:354` | add `uint32_t flags` (last parameter, as htp_moe does) and `#define HEXKL_MOE_FLAG_DOWN_HADAMARD 2u` (1u stays reserved for htp_moe's `HEXKL_MOE_FLAG_M1_GEMV`); `inter % 256` guard |
| IDL | `test/htp/nntr_hvx.idl` after `mm_u8i4_moe_layer_timed` (`:331-340`) | **additive** `moe_set_opts(in uint32 flags, rout uint32 applied)` (same entry htp_moe added for #80, so a later merge lines up); debug entry `fwht_rows_f32(in sequence<float> x, in uint32 rows, in uint32 k, rout sequence<float> y)` next to `swiglu_det_f32` (`:218-228`) |
| skel wrappers | `test/htp/nntr_hvx_mm_u8i4.c:967-1000` (both `moe_layer*` wrappers pass `s->moe_flags`), `test/htp/nntr_hvx_session.h:54-62` (`uint32_t moe_flags`), `test/htp/nntr_hvx_softmax.c:67` (the fwht debug entry beside `swiglu_det_f32`) | |
| skel build | `test/htp/build.sh:70-74` `SRCS` | add `$BACKEND/hvx/hvx_fwht_f32.c` (htp_moe rule 17 / #97: a missing file here is an undefined symbol on the phone, not a build error) |
| stub | `nntrainer/tensor/htp_backend/generate_stub.sh` | regenerated by `build.sh`; app rebuild required (trap: `AEE_EBADPARM 0x8000040E` on a stale skel) |
| host stubs / checks | `test/htp/host/hvx_scalar_stubs.{h,c}` (stand-in = `fwht_rows_f32_ref`), `test/htp/host/moe_layer_host_check.c`, `test/htp/host/run_host_checks.sh` | new case: flag on with host-folded weights ≈ flag off with unfolded weights; a `grep -c qf32 hvx_fwht_f32.c == 0` line |
| converter | `Applications/CausalLM/quantize_stream.cpp:376-400` (`parseDType`/`dtypeName`), `:469-475` (`quantizedSize`, same bytes as WH), `:550`, `:608` (refusals), `:776` (fold point: `fwht_rows_f32_ref(source.data(), rows /*N*/, columns /*K*/)` **immediately before** `quant_qs4cx_f32`, only for the `_down` tensor), `:842` (`flushQs4cxScales`), `:1183-1189` (`gate_up` is written as plain `QS4CX_WH` bytes under `_HAD`), `:1275` (`moe_layer_dtype` tag), `:1308-1310` (usage) | `--moe_dtype QS4CX_WH_HAD`; reject `columns % 256 != 0` |
| dtype enum | `api/ccapi/include/tensor_dim.h:63-67`, `nntrainer/utils/base_properties.h:665-671`, `nntrainer/tensor/tensor_dim.cpp:182,425`, `nntrainer/tensor/tensor.cpp:139,191,247,298` (factory → `QS4CX_WH_Tensor`), `nntrainer/tensor/qs4cx_tensor.h:312` (`getStringDataType` must follow the dim's dtype), `nntrainer/tensor/float_tensor.cpp:765` (same throw as WH), `nntrainer/models/neuralnet.cpp:975` | `QS4CX_WH_HAD`: identical byte layout to `QS4CX_WH`, "expert down_proj K-axis rotated" |
| loader | `Applications/CausalLM/models/lfm2_moe/lfm2_moe_causallm.cpp:45,67` (dtype string → `weight_dtype`), `Applications/CausalLM/models/transformer.cpp:491-497,547`, `Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp:482-490,528-547` | accept `_HAD` as WH layout; pass `down_hadamard` |
| ComputeOps | `nntrainer/tensor/cpu_backend/compute_ops.h:290-320` (`gemm_qs4cx_moe_layer_fp32`, `register_qs4cx_weight`) | add `bool down_hadamard` after `weights_wh` (named core edit, contract §6) |
| HtpComputeOps | `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:890-925` (`gemm_qs4cx_moe_layer_fp32`), `:931-941`, `:1551-1560` (`invokeMoeLayer`), `:595-600` (`l2DiffEnabled`) | `moe_set_opts` on flag change (cached `moe_flags_applied_`, refuse `plain + had`); new `l2DiffMoe` (§3.4) |
| device gtests | `test/unittest/unittest_hvx_softmax.cpp:176` (add `HvxFwht.MatchesScalarBitExact`), `test/unittest/unittest_hvx_mm_u8i4.cpp:1148` (add `MoeLayerHadamardMatchesTwoCallReference`), `test/jni/Android.mk` (no new sources: the stub is regenerated) | |
| host gtest | `test/unittest/unittest_nntrainer_cpu_backend.cpp:635-700` (beside `wh_pack_unpacks_like_the_load_path`) | `hadamard_fold_preserves_dot`, `fwht_ref_is_involutive_up_to_scale` |
| profile / report | none: the FWHT sits inside the existing `REQUANT` probe window (`:1067-1076`) and the tail's `SWIGLU` column; `MOE_T_*` (`nntr_hvx_mm_u8i4.c:865-888`), `HTP_MOE_T_*` (`htp_compute_ops.cpp:19`) and `tools/htp_fc_report.py` are untouched | |

## 3. Design

### 3.1 The FWHT-256 specification (one spec, two implementations, bit-identical)

For each row and each 256-float block `v[0..255]` (k is a multiple of 256;
1792 = 7 blocks, no inter-block mixing):

```
for s in 1, 2, 4, 8, 16, 32, 64, 128:          # ascending, fixed
  for i in 0..255 with (i & s) == 0:
    a = v[i]; c = v[i + s]
    v[i]     = a + c                            # one IEEE-754 f32 add, RNE
    v[i + s] = a - c                            # one IEEE-754 f32 sub, RNE
for i in 0..255: v[i] = v[i] * 0.0625f          # exact (power of two)
```

Every stage output is exactly one f32 add or sub of two stage inputs and
the stage order is fixed, so **lane arrangement cannot change the bits**:
the HVX version is free to compute stages 1–16 inside a vector (`vror` by
±s lanes + `vmux` on the lane-index bit s) and stages 32/64/128 across the
8 vectors of a block, and still matches the scalar loop bit for bit. No
FMA, no qf32 (`Q6_Vqf32_*` is forbidden — v75/v79 qf32→sf differ, LEDGER
rule), no reassociation. Denormals: HVX sf arithmetic flushes them; the
scalar ref applies `ftz()` to every loaded input and every result so the
two agree there too. The sign of a flushed zero is the one thing the host
cannot decide; the device gtest feeds ±subnormal inputs and fixes the
ref's convention in the same PR (§4 step 5).

Normalisation: **1/16 on both sides**, as decided. Activation side = the
final multiply of `hvx_fwht_rows_f32` / `fwht_rows_f32_ref`; weight side =
the same function applied to the weight rows (§3.2), so `(hH/16)(HᵀW/16) =
hW` because `HHᵀ = 256·I`. Note for the record: a power-of-two scale on a
row is byte-neutral for `hvx_quant_rows_u8_params` (`hvx_quant_u8.c:80-92`:
`s` scales exactly, `zp` and every `x/s` are unchanged), so the activation
side's 1/16 costs one `vmpy` per vector and changes no u8 byte; it is kept
because the issue decided it and it keeps `mid`'s magnitude readable in
the diagnostics. Rejected alternative: 1/256 on the weight side only.

### 3.2 The fold axis, and the check that catches the wrong one

`writeQuantized` (`quantize_stream.cpp:776`) holds `source` as **N rows of
K** (`is_nxk=true`; for down N = 2048, K = 1792). `Hᵀ·W` on the K×N
matrix is, column by column, an FWHT along K; in N×K storage that is an
FWHT along each row. So the fold is literally
`fwht_rows_f32_ref(source.data(), rows, columns)` — the same function as
the activation side, no transpose, no second implementation — followed by
the untouched `quant_qs4cx_f32` → unpack → colsum → `whPack` sequence. The
scales and column sums are computed from the folded values, which is what
the epilogue needs (`hvx_dequant_acc_tile_to_f32` sees only `W'`).

Identity check (host gtest): random `W` (K = 512, N = 64) and `h`;
`y0 = h·W` in double, `y1 = fwht(h)·fwht_rows(W)` in f32; require
`max|y1 − y0| ≤ 1e-4 · max|y0|`. Folding along N instead of K fails this
by O(1). A second gtest checks `fwht_rows(fwht_rows(x)) == x` up to f32
rounding (the transform is its own inverse under the 1/16 scale).

### 3.3 Flag plumbing (chosen) vs per-weight flag (rejected)

Chosen: the rotation is a **layer** property carried by the model file's
`moe_layer_dtype = QS4CX_WH_HAD` (`quantize_stream.cpp:1275` writes it,
`lfm2_moe_causallm.cpp:45,67` turns it into every expert weight's dtype).
The layer sees it (`lfm2_moe_layer.cpp:482-490`: `_HAD` is accepted as WH
layout, `down_hadamard = dtype == QS4CX_WH_HAD`), passes
`down_hadamard` through `gemm_qs4cx_moe_layer_fp32`, and `HtpComputeOps`
sets the session option `moe_set_opts(HEXKL_MOE_FLAG_DOWN_HADAMARD)` once
(cached; re-sent only when the value changes, so a model mixing rotated
and unrotated MoE layers still works). The skel stores it in the session
and hands `flags` to `hexkl_mm_u8i4_moe_layer_run`. `gate_up` under
`_HAD` is byte-identical to `QS4CX_WH` (never rotated) — the dtype tag on
it means "WH layout, this layer's down input is rotated".

Rejected: a per-weight flag in `weight_register_u8i4_arena`. The requant
happens between gate_up and down and the kernel would have to read the
*down* handle's flag while quantizing gate_up's output — workable, but it
changes an existing IDL signature (every caller, both gtests) instead of
adding one entry, and the split-call path would still need a layer flag.
Also rejected: a `flags` parameter on `mm_u8i4_moe_layer` itself (two IDL
signatures, four callers). `moe_set_opts` is what htp_moe already has.

Contract §2 / doc 45 §3 compliance: no weight bytes move (DMA path,
arena budget and 32-bit address space unchanged — 0 new DSP heap), the
FWHT runs on the pool inside the requant window that is already exposed
(down's HMX needs all of `mid`), so nothing that was hidden becomes
visible; the op before the quantizer is `_det` (§3.1); `QS4CX_WH_HAD` has
no CPU kernel and no fallback (same throw as WH, `float_tensor.cpp:765`);
the CPU `QS4CX`/`q40` runs stay the accuracy reference.

### 3.4 `NNTR_L2_DIFF` on the MoE-layer path (metric 1)

In `gemm_qs4cx_moe_layer_fp32`, when `l2DiffEnabled()`, after the real
call and per expert `e` with `row_count[e] > 0`:

1. gather `act_e` (M_e × K) from `row_index`;
2. `gu = nntr_hvx_mm_u8i4_layer(h_gu[e], act_e)` (device, same handle);
3. `mid = swiglu_det_one(gu)` (host, `swiglu_det.h:179`); if `down_hadamard`, `fwht_rows_f32_ref(mid, M_e, inter)`;
4. `ref_e = mid · dequant(W_dn[e])` in f32 on the host, **no activation quantization**; `W_dn[e]` is unpacked from the registered bytes with `whSlot` (`htp_wh_layout.h`) × `w_scale` — read from the arena chunk (`ArenaEntry` chunk/off, host-mapped), not from `down_data[e]`, whose pages `releaseArmSource` already gave back;
5. `dev_e = nntr_hvx_mm_u8i4_moe_layer(M_e, n_experts=1, identity rows, weight 1.0)`;
6. print `[L2-DIFF-MOE] expert=e M=M_e had=0/1 snr=.. dB max_abs_err=..`.

Both sides start from the same device gate_up output and the same int4
down weight, so the only difference is the u8 requant of `mid` (plus f32
rounding) — exactly what the rotation targets. The int4-weight error that
the fold also changes is deliberately outside this metric; PPL (metric 0)
carries it. Cost: one host f32 GEMM per expert call (≈ 0.7 GFLOP for
M_e = 100) — a diagnostic run, never a TPS run.

## 4. Steps

Each step ends at a `hexagon-gates` rung; `source tools/htp/env.sh` first.

1. **Spec + fold + host gtests** — `fwht_det.h`; converter dtype
   `QS4CX_WH_HAD` and the core enum plumbing (§2.2 rows "converter",
   "dtype enum"); gtests `hadamard_fold_preserves_dot`,
   `fwht_ref_is_involutive_up_to_scale`, and a `QS4CX_WH_HAD` variant of
   `wh_pack_unpacks_like_the_load_path` proving the gate_up bytes equal
   the `QS4CX_WH` bytes. Gate **1**: `ninja -C build`,
   `unittest_nntrainer_cpu_backend --gtest_filter='*qs4cx*:*fwht*:*hadamard*'`,
   `unittest_causallm_models --gtest_filter='*Lfm2Moe*'` (6 passed, none
   skipped), `tools/htp_syntax_check.sh`.
2. **Convert the model** on the workstation:
   `nntr_quantize_stream $NNTR_MODEL_DIR/fp32 -o $NNTR_MODEL_DIR/q40-qs4cx-wh-had --fc_dtype Q4_0 --embd_dtype Q4_0 --lmhead_dtype Q4_0 --moe_dtype QS4CX_WH_HAD --isa ARM`;
   `nntr_config.json`: `"moe_engine": "htp"`, `"moe_htp_layers": ""`,
   `"moe_layer_dtype": "QS4CX_WH_HAD"` (written by the tool; verify);
   record the `_ARM.bin` md5 and size (must equal `q40-qs4cx-wh`'s). Gate:
   the file loads in the host `nntrainer_causallm`? — no, the host has no
   HTP; gate = size equality + md5 recorded + the converter's log line
   `MoE dtype: QS4CX_WH_HAD`.
3. **DSP kernel + flag + host check** — `hvx_fwht_f32.{h,c}`, the
   `flags` parameter and the two inserts (§2.1), `moe_set_opts` and the
   fwht debug entry in the IDL and wrappers, `build.sh` `SRCS`, scalar
   stub, `moe_layer_host_check.c` case "hadamard fold: flag on with
   host-folded weights vs flag off, SNR ≥ 60 dB at the harness shape
   (K=64, inter=256 — the harness inter must become a multiple of 256 for
   this case), and the `MOE_TAIL_MAX_ROWS=16u` build exercises the tail
   site", plus the `qf32` grep line. Gate **1** (`run_host_checks.sh` →
   `ALL CHECKS PASS`, `WORKER POOL LANES OK`) then gate **2**
   (`test/htp/build.sh`, `-Wall -Werror`, md5 of `libnntr_hvx_skel.so`;
   confirm `nm` shows `hvx_fwht_rows_f32` defined — rule 17).
4. **Host side** — `compute_ops.h` `down_hadamard`, `HtpComputeOps`
   (option cache + `l2DiffMoe`), `transformer.cpp`, `lfm2_moe_layer.cpp`;
   device gtests `HvxFwht.MatchesScalarBitExact` (8192 floats: a
   real-SwiGLU-like spread, exact cancellations, ±subnormals, ±0, the
   largest normals — bit compare, per-stage counts on failure like
   `HvxSwigluDet`) and `MoeLayerHadamardMatchesTwoCallReference` (K=2048,
   I=1792, N=2048, **8 experts** — 32 heap-baked experts would be 176 MB
   against the ≈ 182 MiB DSP heap, doc 46 §41; the reference is the
   existing two-call pattern with `fwht_rows_f32_ref` between
   `swiglu_det` and `htp_quant_pack_u8_ah`, expected bit-identical; plus
   a folded-vs-unfolded SNR ≥ 40 dB check and `applied == flags` from
   `moe_set_opts`). Gate **1** again (`tools/htp_syntax_check.sh`, host
   gtests) then gate **3**: `build_android.sh --htp` (`readelf` shows
   `libsdkl.so`, `libcdsprpc.so`), `ndk-build unittest_hvx_mm_u8i4
   unittest_hvx_softmax`, md5s.
5. **Device (user) — unavoidable; handoff `docs/measurements/95-down-hadamard.md`**
   (`hexagon-handoff` template, ≈ 90 min). Order inside the sitting:
   gtests first (`unittest_hvx_softmax --gtest_filter='HvxFwht*'`,
   `unittest_hvx_mm_u8i4 --gtest_filter='*Moe*'`) — if `MatchesScalarBitExact`
   fails only on subnormal inputs, the ref's flushed-zero sign is
   flipped and the step repeats (one extra round, no other change).
   Then full-model E2E, prompt 512, gen 64 / 512 / 1024, `NNTR_NUM_THREADS=8`,
   each twice:

   | variant | binary / skel | model | purpose |
   |---|---|---|---|
   | A | `htp_hadamard` @ `1ac18c39` unchanged, skel `.A.so` | `q40-qs4cx-wh` | control, run first |
   | B | new binary + new skel, flag off | `q40-qs4cx-wh` | code change alone: text bit-identical to A, PPL == A, TPS within noise |
   | C | new binary + new skel, flag on (set by the model) | `q40-qs4cx-wh-had` | the measurement |
   | D | CPU, `moe_engine=cpu` | `q40` | text/PPL reference (once per gen length is enough) |

   Extra one-off runs per variant (not for tok/s): `NNTR_PPL=1`,
   `NNTR_HTP_PROFILE=2` (paste `[HTP-PROFILE]`, read `min`), and for B and
   C `NNTR_L2_DIFF=1` (paste the `[L2-DIFF-MOE]` summary: min / median /
   count < 100 dB). Expected log lines: `MoE HTP kernel warmed up at load`,
   `[HTP-MOE] opts applied=2` (new, printed once), `prefill: 512 tokens`,
   `generation: G tokens`, `[PPL] prompt tokens=512 ... ppl=`.
6. PR into `htp_hadamard` after the handoff is filled (`state:measured`
   → supervisor), upstream-shaped commits: kernel/IDL, converter+dtype,
   host wiring+diagnostic, tests, docs — separately.

## 5. Risks

* **Flushed-zero sign** (§3.1): the only host-vs-device unknown of the
  kernel; surfaced by the gtest's ±subnormal vectors, bounded to one
  extra device round.
* **Stale skel / stub**: the IDL gains two entries, so an old skel with the
  new app fails with `AEE_EBADPARM (0x8000040E)` (doc 46 §48.7). The
  handoff table carries the skel md5 as read on the device and the
  `applied=` echo of `moe_set_opts`; a run without that line is void.
* **Thermal / DVFS drift between sittings**: A runs first, everything is
  read as A/B/C inside the sitting; two runs per cell; `min` from profiles.
* **DMA rate**: unaffected by design (same bytes, same descriptors); the
  `drain` columns of B and C vs A make an accidental change visible.
* **Address space**: none on the model path (0 new DSP heap, arena
  unchanged); the gtest is sized to 8 experts for the heap-bake path.
* **Text identity for C is not a gate** (§1 metric 3); if the user wants a
  same-weights CPU reference for the rotated model, that is a CPU
  `QS4CX_HAD` dequant path — out of scope, would be a new issue.
* **The metric-1 redefinition** may not be what the user meant; it is
  flagged in the issue comment (§7).
* Converter time: the 3.7 GiB expert set re-quantizes in one pass on the
  workstation (minutes); the fold adds 8 add/sub passes per weight row —
  negligible.

## 6. Docs to update

`docs/htp_moe/BENCHMARK.md` and `docs/htp_moe/LEDGER.md` live on `htp_moe`,
not on this branch. On this branch the results go into the filled
`docs/measurements/95-down-hadamard.md` and the issue; the supervisor
copies them across:

* BENCHMARK.md: three new rows (B, C at gen 64/512/1024 with the `ppl`
  and `requant SNR` columns) beside the A control of this sitting;
  artifact section: `q40-qs4cx-wh-had/*_ARM.bin` md5, the new skel and
  app md5s.
* LEDGER.md: item ⑱ closes with the verdict (effect / no effect, the
  numbers); rule candidates: "a power-of-two row scale is byte-neutral
  for the u8 quantizer", "HVX sf flushes denormals — det refs carry
  `ftz()`" (if the gtest confirms), and the requant-site count on this
  tree (two, tail off by default).
* `docs/htp_attention/*` are read-only.

## 7. Open question for the user (non-blocking)

Metric 1 is redefined from "device-vs-reference SNR + `total_flips`" (already
999 dB / 0 on this tree, doc 44 §566/§614) to "requant SNR against an
unquantized-`mid` f32 reference on the MoE-layer path" (§3.4). If the
literal metric is wanted anyway it needs a plain-layout rotated dtype
(`QS4CX_HAD`) and a way to force the split-call path for it — roughly the
dtype plumbing of §2.2 a second time. Metric 0 (PPL on/off) decides the
issue either way, so the plan proceeds unless told otherwise.
