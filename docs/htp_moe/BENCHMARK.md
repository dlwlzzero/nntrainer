# HTP MoE decode benchmark — LFM2.5-8B-A1B on the Galaxy S25 Ultra

Contract: `docs/plans/0001-htp-moe-decode-agent-system.md`. The supervisor
edits this file from filled `docs/measurements/*.md`; nobody else writes
numbers here.

## Method

`nntrainer_causallm` on the phone, `NNTR_NUM_THREADS=8`, prompt 512 tokens,
generation 64 / 512 / 1024 tokens, two runs per cell, non-profile binary.
Decode tok/s is reported over the whole generation and over the last 64
tokens. Every sitting starts with variant A (the unchanged reference
binary) and every other variant is read against it inside that sitting.
Accuracy: generated text identical to the CPU `q40` run of the same
prompt (y/n), `NNTR_L2_DIFF` when DSP arithmetic changed.

Unit: `R3CY10WM83Y` unless a row says otherwise. Rows from the PR author's
device are labelled `(PR)` and are provisional. Rows labelled
`R3CY205ZMND` (the second S25 Ultra, SM-S938N; #77 ran there because it
was the only unit reachable from that workstation) are ~7 % apart in DSP
clock from ours and do not replace the "now" (LEDGER rule 13); the anchor
sitting on `R3CY10WM83Y` is #94 (sitting 2, variant A; #91 folded in).

## Goals

| goal | now | target | ceiling | status |
|---|---|---|---|---|
| decode tok/s, NPU, gen 64 / 512 / 1024 | 20.8 (PR, gen 512, prompt 444); `R3CY205ZMND` #77: **19.7 / 21.3 · 18.4 / 18.2 · 19.0 / 17.0** (two runs per length) | **≥ 50** at each length | 48–52 (730 MB/token ÷ 34–38 GB/s) | provisional until the anchor sitting #94 (variant A) on `R3CY10WM83Y`; distance on `R3CY205ZMND`: **2.7×** at G=512 |
| decode tok/s, CPU control (`q40`, 8 threads) | 48 (PR); `R3CY205ZMND` #77: 54.1 / 52.9 · 52.7 / 52.0 · 46.4 / 48.5 | — (the floor the NPU must beat) | same ceiling | provisional; the CPU sits on the ceiling and clears 50 at G=64/512, not at G=1024 |
| prefill tok/s, NPU, prompt 512 | 523–532 (PR, prompt 444); `R3CY205ZMND` #77: 403–541 (median of 6: 490); CPU 231–336 | **≥ −5 % of variant A** in every handoff | — | gate, not a goal; the denominator is variant A of each sitting, never this cell |
| accuracy | text identical to CPU | identical | — | gate |

## Results

| date | issue | variant / what changed | model | gen | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | text = CPU | file |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-16/17 | (PR) | PR head as-is, MoE FFN on HTP | `q40-qs4cx-wh` | 512 (prompt 444) | 523 | 20.8 | — | n/a (different weights from `q40`) | `docs/htp_attention/49` §1 |
| 2026-09-16/17 | (PR) | CPU, all Q4_0, 8 threads | `q40` | 512 (prompt 444) | 334 | 48 | — | reference | `docs/htp_attention/49` §6.1.1 |
| 2026-09-21 | (PR) | + conv in_proj on HTP (18 layers) | `q40-qs4cx-wh` | 512 (prompt 444) | 532 | 16.5 (unexplained drop, no control run) | — | identical to the previous run | `docs/htp_attention/50` §3.4 |
| 2026-09-21 | #77 A run 1, unit `R3CY205ZMND` | CPU control, all Q4_0, 8 threads (`htp/77-first-handoff` @ `81fd3ec9`, forward path = PR head) | `q40` | 64 | 336.4 | 54.10 | n/a (line not printed, #89) | reference; run1 = run2 | `77-first-handoff.md` §A |
| 2026-09-21 | #77 A run 2, `R3CY205ZMND` | same | `q40` | 64 | 253.1 | 52.94 | n/a | reference; run1 = run2 | same |
| 2026-09-21 | #77 A run 1, `R3CY205ZMND` | same | `q40` | 512 | 297.5 | 52.67 | n/a | reference; run1 = run2 | same |
| 2026-09-21 | #77 A run 2, `R3CY205ZMND` | same | `q40` | 512 | 231.5 | 51.97 | n/a | reference; run1 = run2 | same |
| 2026-09-21 | #77 A run 1, `R3CY205ZMND` | same | `q40` | 1024 | 287.3 | 46.44 | n/a | reference; run1 = run2 | same |
| 2026-09-21 | #77 A run 2, `R3CY205ZMND` | same | `q40` | 1024 | 288.0 | 48.53 | n/a | reference; run1 = run2 | same |
| 2026-09-21 | #77 A run 1, `R3CY205ZMND` | PR head as-is, MoE FFN on HTP (device skel `47c14253…`) | `q40-qs4cx-wh` | 64 | 403.5 | 19.68 | n/a | n/a (different weights from `q40`); run1 = run2 | same |
| 2026-09-21 | #77 A run 2, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 64 | 475.4 | 21.27 | n/a | run1 = run2 | same |
| 2026-09-21 | #77 A run 1, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 512 | 485.8 | 18.38 | n/a | run1 = run2 | same |
| 2026-09-21 | #77 A run 2, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 512 | 494.7 | 18.24 | n/a | run1 = run2 | same |
| 2026-09-21 | #77 A run 1, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 1024 | 541.2 | 19.04 | n/a | run1 = run2 | same |
| 2026-09-21 | #77 A run 2, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 1024 | 455.1 | 16.96 | n/a | run1 = run2 | same |

#77 side tables (unit `R3CY205ZMND`, same sitting; verdicts in LEDGER §2):

* **Per-type decode cost, ms/token, `--profile` build (never tok/s), (sum at G=512 − sum at G=64) / 448:** `lfm2_moe` NPU **42.12** vs CPU **22.32**; `fully_connected` **28.16** vs **10.29**; `output_of_causallm` 2.85 vs 3.40; `mha_core` 2.30 vs 2.81; `addition` 0.21 vs 0.14; `rms_norm` 0.19 vs 0.12; `custom_multiply` 0.15 vs 0.09; `causal_conv1d` 0.06 vs 0.03; `split` 0.05 vs 0.05; `reshaped_rms_norm` 0.06 vs 0.03; `swiglu` 0.04 vs 0.02; total 76.2 vs 39.3 (profile-inflated; TPS binary gives 54.4 vs 19.0).
* **Transport floor (`[HTP-PROFILE]`, M==1 row, 1408 calls, `qos_mode` 2):** level 2 host 1948.6 / dsp 1360.9 / transport **587.7** µs/call, weight DMA 16.2 GB/s; level 3 host 1750.4 / dsp 1222.7 / transport **527.7**, 18.0 GB/s. M>1 row (23 calls, prefill): transport 2231 / 2123 µs/call, DMA 10.0 / 10.1 GB/s.
* **Arena DMA probe (GB/s, vote=1 / vote=0, workers 1 → 4):** contiguous 4 KiB × 256: 79.6 / 75.2 → 72.0 / 72.2; contiguous 1 MiB × 1: 78.4 / 76.6 → 72.4 / 72.3; 2D 16 KiB × 64 @ 56 KiB: 116.5 / 113.9 → 109.7 / 110.2; 2D 8 KiB × 64 @ 56 KiB: 111.0 / 109.5 → 108.6 / 108.1. 32/32 `checksum_ok=y`.
* **Two-reader DDR:** CPU alone 67.90 GB/s, CPU with DSP 39.73 (−41 %); DSP side 3265 / 3113 GB/s = cache/VTCM-resident, invalid (#90).

## Artifacts

| artifact | md5 | built from | note |
|---|---|---|---|
| `test/htp/build/libnntr_hvx_skel.so` (v79, SDK 6.4.0.1, HexKL 6.4.0.1) | `0a3d4b81541799e5a25bbf234c515812` | `htp_moe` @ `2ce38d65` | first workstation build, 2026-09-21 |
| `Applications/CausalLM/jni/libs/arm64-v8a/nntrainer_causallm` (`--htp`, NDK r30) | `53814a39abad75c045dc6d8ae16e67df` | `htp_moe` @ `2ce38d65` | `libnntrainer.so` NEEDED lists `libsdkl.so`, `libcdsprpc.so` |
| `Applications/CausalLM/jni/libs/arm64-v8a/libcausallm_core.so` | `f735315882a0ba1769b5e901d9cb122f` | same | |
| `Applications/CausalLM/jni/obj/local/arm64-v8a/libnntrainer.so` | `7c42228a6706dded899bf05c1cb6ef14` | same | r30 leaves it in `obj/local` |
| `Applications/CausalLM/jni/obj/local/arm64-v8a/libccapi-nntrainer.so` | `65c9034c6341384443166b89de66191d` | same | |
| `test/jni/obj/local/arm64-v8a/unittest_hvx_mm_u8i4` / `_softmax` / `_attn` / `_fc` | `08a99d8f…` / `17ef87d5…` / `63fa6a4b…` / `7bdc2e9d…` | same | device gtests |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40/nntr_lfm2_8b_a1b_q40_arm.bin` (4,768,855,808 B) | `d28f55c5bd7adeb8bf73b02de582eb88` | `nntr_quantize_stream` from `htp_moe` @ `2ce38d65`, `--isa ARM`, all Q4_0 | CPU control (#78) |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (4,316,133,120 B) | `7b7867fab51845664c0050c0a837073e` | same plus `--moe_dtype QS4CX_WH`; `moe_engine: htp`, `moe_htp_layers: ""` | NPU model (#78) |
| #77 set, `htp/77-first-handoff` @ `3c6e8397` (table) / `81fd3ec9` (built and pushed, other build path): skel A / novote, `tps/nntrainer_causallm`, `libcausallm_core.so`, `libnntrainer.so`, `profile/nntrainer_causallm`, `unittest_hvx_dma_probe` | table `0dd2c303…` / `40007503…` / `53814a39…` / `83955c80…` / `326094f4…` / `ed4e73fc…` / `cf18de2c…`; **device** `47c14253…` / `99556663…` / `170af7a6…` / — / — / `3806b904…` / `9de5715b…` | `77-first-handoff.md` Artifacts + Notes | compiled files differ by build path (LEDGER rule 14); non-compiled files (`libc++_shared.so` `b1586b9b…`, `libsdkl.so` `0ad4e22a…`, `77-prompt512.txt` `fc65c158…`, `tokenizer.json` `7b8067a5…`) and both model `.bin`s rebuilt there match bit-for-bit |

**Next sitting (#94) rebuild note, 2026-09-22:** every artifact above was
built from `htp_moe` @ `2ce38d65` (or the #77 branch). PR #93 (`b6ebc2b7`)
changed `test/htp/nntr_hvx.idl` (`dma_probe`, `moe_dma_trace_read`,
`dma_replay`; `mm_u8i4_moe_layer_timed`'s `stage_us` grew) and the probe
slots, so the reference set for sitting 2 — skel, `nntrainer_causallm`,
`libcausallm_core.so`, `libnntrainer.so`, `libccapi-nntrainer.so`,
`unittest_hvx_dma_probe`, `unittest_hvx_mm_u8i4` — **must be rebuilt from
`htp_moe` @ `750428e8`** and their md5s recorded from the pushed build
(LEDGER rules 3, 14). The two model `.bin`s are unchanged (quantizer
format untouched) and keep the md5s above. Pushing an old app with the
new skel, or the reverse, fails with `AEE_EBADPARM`.

## Log

| date | what | rows / goals touched |
|---|---|---|
| 2026-09-21 | File created from PR #4327 docs 49 and 50; goals from the contract | all |
| 2026-09-21 | Weights prepared (#78); first handoff written (#77, `docs/measurements/77-first-handoff.md` on `htp/77-first-handoff`) | artifacts |
| 2026-09-21 | #77 filled (`e8b930ad`) on unit **`R3CY205ZMND`**, not ours: 12 A rows added, unit-tagged readings next to the provisional "now" (not replaced; anchor sitting #91), side tables A-profile / B / C / ④, artifact md5 note; #77 closed | results, goals (now column annotated, status), artifacts |
| 2026-09-22 | Cycle 3: #87 closed (PR #93 merged `b6ebc2b7`), #91 folded into #94 (sitting 2 = anchor cells + #87 trace/replay, variant C only if PR #86 merges first); rebuild note for the #94 artifact set; PR #92 guide merged (LEDGER §3a); verdict ① corrected (no FC on the HTP in the NPU config → LEDGER ⑰) | goals (status column), artifacts |
