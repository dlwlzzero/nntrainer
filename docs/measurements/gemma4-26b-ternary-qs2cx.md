# Measurement: Gemma4-26B-A4B ternary QS2CX_WH — device decode + accuracy

Branch `htp/gemma4-ternary-qs2cx` @ `c60bb34df` — estimated device time: ~20 min
(no GitHub issue; branch-local working effort).

> **Note on template:** the standard hexagon-handoff template targets the
> LFM2.5-8B-A1B reference. This task is **Gemma4-26B-A4B**, so the artifacts
> and model dir below are Gemma4's. The structure (A/B in one sitting, text +
> PPL approval) is kept.

## Why
We converted the bart `gemma4-26b-pure-ternary` checkpoint to a `QS2CX_WH`
(2-bit) nntrainer file. On the host it is **bit-identical** (codes AND
per-column scales, all 7680 experts, `mismatches=0`) to the `QS4CX_WH` file of
the same model — the one doc 55 §10.8 measured on an S25 at **nll 4.5562**.
This run confirms the QS2CX_WH file **loads and decodes on the device** and
reproduces that accuracy at **half the expert bytes** (file 7.64 GB vs 13.35 GB).

## Variants (one sitting, A first) — experts vs FC dtype
All three share MoE experts; A uses 4-bit experts, B/C use 2-bit experts. A/B
use Q4_0 FC (CPU/ARM); C uses QS4CX FC (on the HTP, #260).

- **A = QS4CX_WH experts + Q4_0 FC** — the device-validated reference (doc 55
  §10.8, nll 4.5562). Reuse an existing on-phone build if present, else push
  from `~/Downloads/gemma4_ternary_build/out_qs4cx/`.
- **B = QS2CX_WH experts + Q4_0 FC** — the primary deliverable,
  `~/Downloads/gemma4_26b_ternary/`. Host-proven **bit-identical** to A's
  experts (codes+scales), so expect B's text == A's and B's PPL == A's, with
  the expert arena **halved**.
- **C = QS2CX_WH experts + QS4CX FC** — the FC-on-NPU variant,
  `~/Downloads/gemma4_26b_ternary_fcqs4cx/`. NOT bit-identical to A; its FC
  accuracy (per-channel QS4CX vs Q4_0's per-32-block) is decided here. Gate:
  text approved + decode PPL ≤ +2 % of A.

## Artifacts (built on the workstation; v79 / S25 Ultra)
| file | md5 | built with |
|---|---|---|
| Applications/CausalLM/jni/libs/arm64-v8a/nntrainer_causallm | | `build_android.sh --htp` (hexagon-gates rung 3) |
| Applications/CausalLM/jni/libs/arm64-v8a/libnntrainer.so | | same |
| test/htp/build/libnntr_hvx_skel.so | | hexagon-gates rung 2 |
| B: ~/Downloads/gemma4_26b_ternary/nntr_gemma4_26b_a4b_qs2cx_wh.bin (7,641,378,936 B) | `md5sum` on push | quantizer @ c60bb34df, `--fc_dtype Q4_0 --moe_dtype QS2CX_WH --moe_palette_refit off` |
| A: ~/Downloads/gemma4_ternary_build/out_qs4cx/nntr_gemma4_26b_a4b_qs4cx_wh.bin (13,350,844,536 B) | | quantizer @ c60bb34df, `--fc_dtype Q4_0 --moe_dtype QS4CX_WH` |
| C: ~/Downloads/gemma4_26b_ternary_fcqs4cx/nntr_gemma4_26b_a4b_qs2cx_fcqs4cx.bin (7,540,724,856 B) | | quantizer @ 1dd6c1ff3, `--fc_dtype QS4CX --moe_dtype QS2CX_WH --moe_palette_refit off` |

**Prerequisite:** the Android app (`nntrainer_causallm` + `libnntrainer.so`,
arm64-v8a) and the DSP skel must be built for this branch — hexagon-gates
rungs 2–3 (`source tools/htp/env.sh`; ~10 min). The QS4CX_WH Gemma4-26B run in
doc 55 used the same app; if those binaries at a compatible revision are still
on the phone, A can reuse them. The **only new weight file** this task
produces is B's `.bin`.

## Steps (workstation, S25 Ultra on USB)
1. `source tools/htp/env.sh`. Build/confirm the app + skel (rungs 2–3) if not
   already present for this branch. `adb devices` lists exactly one S25 Ultra;
   record its serial, battery %, warm/cold under Notes.
2. Push the model dirs to the phone (each is a folder with the `.bin`,
   `nntr_config.json`, `config.json`, `tokenizer.json`):
   - B: `adb push ~/Downloads/gemma4_26b_ternary /data/local/tmp/nntrainer/causallm/models/gemma4-26b-ternary-qs2cx`
   - A: the QS4CX_WH dir (reuse on-device, or push from `out_qs4cx/` with its configs).
   Record the `.bin` md5 the device's `md5sum` shows for each.
3. For each generation length G in 64 / 512 / 1024, and each variant (A first):
   set `"num_to_generate": G` in that model's on-device `nntr_config.json`, then
   ```
   adb shell "cd /data/local/tmp/nntrainer/causallm && NNTR_NUM_THREADS=8 NNTR_MOE_CACHE_EXPERTS=16 \
     LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/<model> '<512-token prompt>'"
   ```
   Expected lines: `prefill: 512 tokens, <ms> ms, <tps> TPS`, `generation: <G> tokens, <ms> ms, <tps> TPS`, the generated text, and the `[HTP] arena chunk` lines (B's expert arena should be ~half A's).
   Repeat twice; keep both.
4. Accuracy (per-token entry): run A at G=512 to write A's continuation, then
   force every variant (and A again as a null check) onto it:
   `NNTR_HTP_FORWARD=1 NNTR_PPL_DECODE=cont.ids ...`. Record decode PPL per variant.
5. Paste the summary lines below; commit this file on the branch.

## Results (fill in)
| variant | gen | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS | expert arena (MiB) | bin md5 (device) |
|---|---|---|---|---|---|---|---|---|
| A (QS4CX_WH, Q4_0 FC) | 64 | 1 | | | | | | |
| B (QS2CX_WH, Q4_0 FC) | 64 | 1 | | | | | | |
| C (QS2CX_WH, QS4CX FC) | 64 | 1 | | | | | | |
| A (QS4CX_WH, Q4_0 FC) | 512 | 1 | | | | | | |
| B (QS2CX_WH, Q4_0 FC) | 512 | 1 | | | | | | |
| C (QS2CX_WH, QS4CX FC) | 512 | 1 | | | | | | |

Accuracy anchor: doc 55 §10.8 QS4CX_WH nll **4.5562**. Gate: B's decode PPL ==
A's (bit-identical expectation), B's text == A's.

## Text + PPL approval
| variant | decode PPL (G=512, forced on A) | generated text (G=64, run 1) | text approved (y/n) |
|---|---|---|---|
| A (QS4CX_WH, Q4_0 FC) | | <paste> | (reference) |
| B (QS2CX_WH, Q4_0 FC) | | <paste> | |
| C (QS2CX_WH, QS4CX FC) | | <paste> | |

## Notes from the run
<serial, battery, thermal; any AEE/FARF errors; whether A reused existing on-device binaries; B's arena vs A's>
