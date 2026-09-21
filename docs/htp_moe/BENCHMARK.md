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
device are labelled `(PR)` and are provisional.

## Goals

| goal | now | target | ceiling | status |
|---|---|---|---|---|
| decode tok/s, NPU, gen 64 / 512 / 1024 | 20.8 (PR, gen 512, prompt 444) | **≥ 50** at each length | 48–52 (730 MB/token ÷ 34–38 GB/s) | provisional until the first handoff |
| decode tok/s, CPU control (`q40`, 8 threads) | 48 (PR) | — (the floor the NPU must beat) | same ceiling | provisional |
| prefill tok/s, NPU, prompt 512 | 523–532 (PR, prompt 444) | **≥ −5 % of variant A** in every handoff | — | gate, not a goal |
| accuracy | text identical to CPU | identical | — | gate |

## Results

| date | issue | variant / what changed | model | gen | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | text = CPU | file |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-16/17 | (PR) | PR head as-is, MoE FFN on HTP | `q40-qs4cx-wh` | 512 (prompt 444) | 523 | 20.8 | — | n/a (different weights from `q40`) | `docs/htp_attention/49` §1 |
| 2026-09-16/17 | (PR) | CPU, all Q4_0, 8 threads | `q40` | 512 (prompt 444) | 334 | 48 | — | reference | `docs/htp_attention/49` §6.1.1 |
| 2026-09-21 | (PR) | + conv in_proj on HTP (18 layers) | `q40-qs4cx-wh` | 512 (prompt 444) | 532 | 16.5 (unexplained drop, no control run) | — | identical to the previous run | `docs/htp_attention/50` §3.4 |

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

## Log

| date | what | rows / goals touched |
|---|---|---|
| 2026-09-21 | File created from PR #4327 docs 49 and 50; goals from the contract | all |
| 2026-09-21 | Weights prepared (#78); first handoff written (#77, `docs/measurements/77-first-handoff.md` on `htp/77-first-handoff`) | artifacts |
