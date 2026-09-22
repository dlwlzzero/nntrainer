# htp_hadamard — status (parked 2026-09-22)

**Parked by user decision (2026-09-22).** The Hadamard work stops here and
resumes on `htp_moe` only after the HVX decode work there is finished; the
user will ask for it then. The resume issue is **#110** (port to
`htp_moe`). Nothing on this branch is merged anywhere.

## What this branch is

| ref | commit | content |
|---|---|---|
| upstream PR nnstreamer/nntrainer#4327 head | `3006d255` | base code, no Hadamard |
| `htp_hadamard` | `63167235` | base + agent tooling (`1ac18c39`) + plan `docs/plans/95-down-hadamard.md`; **variant A** of the sitting |
| `htp/95-down-hadamard` | `22ab4b3e..6c89a912` | the Hadamard code, six commits; **variants B and C** |
| `htp/95-down-hadamard` | `3a566761` | filled handoff `docs/measurements/95-down-hadamard.md` |
| draft PR #106 | into `htp_hadamard` | kept open as the reviewable record |

Code commits, oldest first:

1. `22ab4b3e` converter: `--moe_dtype QS4CX_WH_HAD` folds `H/16` (block
   256, FWHT along K) into every expert `down` weight before
   `quant_qs4cx_f32`; same bytes and layout as `QS4CX_WH`.
2. `457ffd7d` host gtests: fold axis (`hadamard_fold_preserves_dot`),
   reference involution.
3. `84f3eefb` DSP: `hvx_fwht_rows_f32` (IEEE sf add/sub only, no qf32)
   before the u8 requant at both requant sites of this tree (HMX block
   loop, HVX tail); `moe_set_opts` IDL entry carries
   `HEXKL_MOE_FLAG_DOWN_HADAMARD = 2u`; host check.
4. `7b61773f` host: the MoE layer sets the flag from the weight dtype;
   `NNTR_L2_DIFF` prints `[L2-DIFF-MOE]` requant SNR per expert.
5. `84c5f71a` device gtests `HvxFwht.*`, `MoeLayerHadamardMatchesTwoCallReference`,
   `MoeSetOptsEchoesKnownBits`.
6. `6c89a912` dense FFN through the MoE kernel resets the options to 0.

## Result (one sitting, `R3CY205ZMND`, 2026-09-22)

| variant | PPL | requant SNR min / p10 / median (dB) | decode tok/s G 64 / 512 / 1024 |
|---|---|---|---|
| D CPU `q40` | 109.759 | — | — |
| A `63167235`, `q40-qs4cx-wh` | 115.095 | — | 25.08 / 24.40 / 23.83 |
| B new code, flag off | 115.095 (= A) | 25.01 / 31.27 / 34.68 | 24.82 / 24.12 / 23.75 |
| C new code, `q40-qs4cx-wh-had` | **100.497** | **39.68 / 41.67 / 42.67** | 24.49 / 23.65 / 23.53 |

* **Accuracy:** PPL −12.7 % vs A and −8.4 % vs the CPU; requant SNR median
  +8.0 dB.
* **Cost:** `requant` stage +446 µs on the M>1 call, +4.6 µs on the M==1
  call = **+2.7 % / +0.3 % of whole-call `dsp=`**. Decode deltas are inside
  the sitting's −2.6 % thermal drift. Prefill tok/s is not readable from
  this sitting (A's own re-run fell 524 → 405).
* **Gates:** B byte-identical to A at every G, PPL equal to the last digit.
* **Verdict (LEDGER ⑱ on `htp_moe`): carry it over.**

## Known defects (fix during the port, #110 steps 4–5)

1. `HvxFwht.MatchesScalarBitExact` fails in the subnormal row only
   (`bad_gated = bad_subnormal_row = 2`). Cause, per LEDGER rule 24: v79 HVX
   IEEE sf **keeps** subnormals, while `fwht_det.h` flushes them
   (`fwht_det_ftz`). Fix: drop the FTZ from the reference, then check the
   converted model's md5 (`a2829cd9…`) and that `-ffast-math` does not set
   aarch64 FPCR.FZ for the reference.
2. `MoeLayerHadamardMatchesTwoCallReference` is bit-exact
   (`bad_elems=0 of 409600`) but fails its SNR-improvement assertion on
   synthetic data (17.5 vs 23.3 dB), while the real model improves.
   LEDGER rule 25: keep bit-exactness as the gate, print the SNRs.

## Resume checklist (for the port, issue #110)

* Base: `htp_moe` after the HVX decode work, and after PRs #107 / #108 if
  still open. Source commits to port: `22ab4b3e..6c89a912` (not the docs).
* `htp_moe` has a **third requant site** this tree lacks: the M=1 HVX GEMV
  path in `hexkl_mm_u8i4_moe.c`. At m = 1 the row is 1792 = 7 × 256, so the
  FWHT applies unchanged.
* The HAD opts bit must not clear the GEMV default-on bit; `6c89a912`'s
  `setMoeOpts(session, 0u)` must become "clear bit 1 only".
* `htp_moe` has no `NNTR_PPL`: cherry-pick upstream `32b46e32` as its own
  commit (user decision at merge).
* Acceptance is #110's handoff: B ≡ A (text and PPL), C PPL ≤ A, SNR median
  ≥ B + 5 dB, M==1 `dsp=` ≤ +2 %, prefill −5 % gate.

## Reproducing the sitting

* Models: `/local/mnt/workspace/models/lfm2.5-8b-a1b/` — `q40`
  (`d28f55c5…`), `q40-qs4cx-wh` (`7b7867fa…`), `q40-qs4cx-wh-had`
  (`a2829cd9…`, differs from `q40-qs4cx-wh` only in the 704 `down`
  tensors). The converter is reproducible: re-converting gave the same md5.
* Build order when switching between `htp_hadamard` and this branch (the IDL
  differs): `nntrainer/tensor/htp_backend/generate_stub.sh`,
  `test/htp/build.sh`, `(cd builddir && ninja install)`, then
  `Applications/CausalLM/build_android.sh --htp --cache`. `--cache` alone
  does not rebuild `libnntrainer.so`.
* This tree's `test/htp/build.sh` lacks `htp_moe`'s #97 undefined-symbol
  guard; run it by hand (44 runtime imports, no project symbol).
