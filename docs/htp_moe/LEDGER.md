# HTP MoE decode ledger — rules learned on silicon, verdicts, open items

Contract: `docs/plans/0001-htp-moe-decode-agent-system.md`. The supervisor
appends here; the planner and implementer read it before touching a kernel.
The tracker issue (#76) carries `hexagon` + `prio:*` and **no `state:*`
label**: state belongs to its children, so the tracker never occupies the
single `state:in-progress` slot.

## Upstream

PR nntrainer/nntrainer#4327, branch `claude/htp-lfm2-moe-ffn` on
`Seunghui98/nntrainer`. `htp_moe` is frozen at head **`2ce38d65`**
(2026-09-21 06:45 UTC, "[CausalLM] Route conv out_proj and the dense FFN to
the HTP by config").

**New commits since (seen 2026-09-21, cycle 1; PR head `7f81560b`,
updated 08:10 UTC, fast-forward from `2ce38d65`, 31 files, +3490/−636).
Merging them into `htp_moe` is the user's decision (contract §5, Q16);
not merged.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `95caa139` | [docs] 50: the fifth run — every FC group on, per-shape calls measured | doc 50 |
| `fb0f02b9` | [HTP] Stage each call through an ION buffer of its own size class | `htp_compute_ops.cpp` (transport-side, relevant to ⑦ / #83) |
| `7497cfd6` | [docs] 50: size-class staging measured — decode back to 20.5 TPS | doc 50 |
| `d2f0bf47` | [CausalLM] Run the dense FFN as one HTP call through the MoE layer kernel | app, `hexkl_mm_u8i4_moe.{c,h}` |
| `35b6124b` | [HTP] Give the dense FFN's MoE-kernel calls their own profile row | profile |
| `20d18ab4` | [docs] 50: the no-switch run — today's baseline is 921–1013, not 848 | doc 50 §3.8: prefill **ms** on the author's unit, 921 ms = 482 tok/s / 1013 ms = 438 tok/s (was 848 ms = 523), decode 21.2 / 21.1; "the device is +73..+165 ms slower today, two consecutive runs 10 % apart" — rule 9 again |
| `80bf92ad` | [docs] 51: third C run — the dense row needs a rebuild to appear | new doc 51 |
| `39ce49d7` | [docs] 51: the dense-only run — 10.4 ms per fused call, as predicted | doc 51 |
| `7f81560b` | [HTP] Run the LFM2 conv block as one call: in_proj, gates, conv1d, out_proj | new `hmx/hexkl_conv_block.{c,h}`, **`hvx/hvx_conv_gate_f32.{c,h}`** (causal depthwise conv1d L=3 + gate, HVX, prefill shape), `test/htp/host/conv_block_host_check.c`, stand-ins moved to `hvx_scalar_stubs.{c,h}`, two IDL entries `mm_u8i4_conv_block[_timed]`, `conv_block_layer.cpp`, `compute_ops.h`; "not yet run on a device" |

**Cycle 2 (seen 2026-09-21 09:44 UTC): PR head moved `7f81560b` →
`72d9b140`, six more commits, all on the conv-block DIFF/compare path
(`Applications/CausalLM/layers/conv_block_layer.cpp`) and doc 51 — no
kernel, IDL or ring change. Still not merged; user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `5d190100` | [docs] 51: the conv block on device — 4.9 ms a call, prefill 732–741 | doc 51, `00_START_HERE.md` (the fused conv block measured on the author's unit: prefill 732–741 **ms**) |
| `81298c35` | [CausalLM] conv_block: NNTR_CONV_BLOCK_DIFF / _SHADOW discriminators | `conv_block_layer.cpp` |
| `9681f465` | [CausalLM] conv_block compare: reference through the CPU Q4_0 GEMM directly | `conv_block_layer.cpp`, doc 51 |
| `4b8b8f6f` | [CausalLM] conv_block DIFF: an activations-only reference and outlier ratios | same |
| `fe74b6cb` | [CausalLM] conv_block DIFF: emulate four weight recipes in f32 | same |
| `72d9b140` | [CausalLM] conv_block DIFF: emulate row-flattening and K-grouped int4 | same |

Reading: the author is chasing an accuracy discrepancy in the fused conv
block (four DIFF commits in 80 minutes); `7f81560b`'s "not yet run on a
device" is now run (5d190100) and under investigation. Nothing here
touches decode; a merge decision can wait until that line settles.

What this changes for us: (1) the "net-new" causal conv1d + gating of §4
now exists upstream in prefill shape — #82 lifts it instead of deriving
it; (2) the ARM decode path of the conv block is still the CPU one there
(the fused call runs only at prefill); (3) doc 50 §3.8 re-measured the
author's unit 10–16 % slower on the same binary (prefill 482–438 tok/s,
not 523), one more reason the provisional "now" is replaced only by #77's
control run on ours; (4) the IDL changed, so a merge forces a
skel + stub rebuild and new md5s in BENCHMARK.md.

## 1. Rules (device disagreed with reasoning; do not re-derive)

Inherited from the PR's device work, with their sources:

1. **A `--profile` build is never the tok/s binary** — it inflates prefill
   by 83 % (doc 46 §43.4).
2. **Read `min`, not `avg`** — profile averages include the first
   (registration) call (doc 46 §48.7).
3. **Rebuild the skel whenever the IDL or DSP sources change**;
   `build_android.sh` never does. Symptom of a stale skel:
   `AEE_EBADPARM (0x8000040E)` (doc 46 §48.7).
4. **`build_android.sh --clean` drops `-Denable-htp` silently**; check
   `readelf -d libnntrainer.so` for `libsdkl.so` (mobile guide §4).
5. **Never push the link-time stub `libcdsprpc.so`** to the phone (mobile
   guide §5).
6. **`QS4CX_WH` has no CPU fallback**; a wrong layout gives plausible wrong
   text, not an error; `moe_htp_layers` must be empty (doc 46 §35, §48.7).
7. **Moving a matmul to the HTP does not pay when the round trip exceeds
   the matmul** — conv in_proj: −13 ms of an expected −65..−100 (doc 50
   §3.4); the same rule closed doc 45 §0.
8. **The DSP address space, not host RAM, bounds registration**: 3840 MiB
   arena + ≈ 182 MiB heap in one 4 GiB space; 256 MiB chunks avoid the
   alignment waste that made it look like a 3 GiB limit (doc 46 §39–§41).
9. **Decode drifts between sittings without a code cause** (20.8 → 16.5,
   doc 50 §3.4; ±5 % in `hvx_impl` #53): every sitting starts with the
   control binary, and results are read as A/B inside the sitting.
10. **Cross-session decode numbers from another unit are provisional**
    (two S25 Ultra units: 6–8.6 % apart on one binary, `hvx_impl` #23/#53).

Learned in this project (#77, unit `R3CY205ZMND`, 2026-09-21):

11. **Isolated DMA probes overstate the in-situ rate by 4–7×.** The same
    engine on the same arena and skel: 72–80 GB/s contiguous, 108–117
    GB/s strided 2D in a probe; **16.2–18.0 GB/s** in the MoE layer call's
    `weight DMA` line of the same sitting (the PR author's 38 GB/s probe
    sat in between). Workers 1 → 4 are flat to negative, the bus vote is
    ≤ 4 % (noise). Wall 2 is therefore decided only by instrumentation
    *inside* the layer call (#87), never by a probe.
12. **A DSP-side bandwidth loop must defeat VTCM/L2 residency or it
    measures nothing**: the two-reader probe's `dsp_alone` read 3265 GB/s.
    Bound the result inside the test (≤ 150 GB/s or `INVALID`) (#90).
13. **Unit `R3CY205ZMND` (SM-S938N) ≠ unit `R3CY10WM83Y`**, ~7 % apart in
    DSP clock: numbers from one are unit-tagged and never replace the
    other's "now"; only a same-sitting A/B on either unit is a verdict.
    Ratio per cell to be measured by #91.
14. **Compiled artifacts are not byte-reproducible across build paths.**
    #77's skel, app and gtest md5s all differed from the table after a
    rebuild of identical sources under another path, while every
    non-compiled file and both model `.bin`s matched bit-for-bit. A
    handoff's md5 table is filled from the build that is pushed, or the
    device `md5sum` lines are recorded against the commit; a mismatch
    with that provenance intact is a note, not a void, in a control-only
    sitting — with a B variant it stays a void.
15. **`NNTR_HTP_PROFILE=3` inflates `[M0-PROF] ffn=` ~5×** (16–20 ms →
    88–90 ms per MoE layer): only the `[HTP-PROFILE]` `transport` /
    `host` / `dsp` columns are comparable across levels 2 and 3.
16. **The `generation(last 64)` line lives in the base report block or
    not at all**: `Lfm2MoeCausalLM` reports from `causal_lm.cpp:699`, not
    from `lfm2_causallm.cpp`'s block (#89).

## 2. Verdicts (measured, closed)

| item | verdict | source |
|---|---|---|
| Tier-1 per-expert FastRPC (176 calls/token) | regression by construction (57 ms transport) | doc 41 §3 |
| Grouped gate_up at decode | built; transport 326 → 68 µs/mm | doc 34 §4E |
| In-place tile dequant, weight residency, cross-matmul prefetch, HVX worker pool, poll QoS + ION | built and verified | doc 34 §4A–F |
| GEMV instead of HMX at M=1 (prefill FC context) | rejected there (padding tax ~40 µs) — **re-opened for decode** where the 64-row tile is wall 1 (1.03 of 1.35 ms) | doc 34 §5.1 vs doc 48 §3 |
| Phase A W1/Q1/DQ1 micro-optimisations | one of four survived (D1: overlap weight DMA with activation quant) | doc 44 Phase A |
| Fused SwiGLU MoE path (A1) | passed, 32/32 `max_abs_err=0` | doc 44 |
| conv in_proj on HTP | runs, text identical, −13 ms prefill, not a decode lever | doc 50 §3.4 |
| Whole-model residency (doc 45) | plan exists, prefill-ordered; we take its Phase E (one call per token) first, M=1 shape | contract §3.1 |
| ① ARM remainder / per-type decode cost (ms/token, NPU vs CPU, unit `R3CY205ZMND`) | `lfm2_moe` **42.1 vs 22.3**, `fully_connected` **28.2 vs 10.3**, `output_of_causallm` 2.85 vs 3.40, `mha_core` 2.30 vs 2.81, all others ≤ 0.2; MoE + dense FC = 92 % of the NPU token, 83 % of the CPU's. The MoE FFN on the NPU costs ~2× the CPU path; the dense FC routed through the HTP 2.7×. lm_head is the one place the NPU already wins, so ⑧ (lm_head twin) is not first. FC thread over-splitting (doc 46 §45) is not the lead: the FC cost is the HTP round trip | #77 A (profile) |
| ② Transport floor (wall 3) | **marshalling**: 527.7 µs/call at level 3 vs 587.7 at level 2 (−10 %, inside ±15 %; wake/clock would have been ≤ 300), `qos_mode` 2 both. Fix = prebound per-layer arguments + persistent ION staging on plain FastRPC (#88, design input #83 narrowed); dspqueue / resident worker deferred to one paragraph | #77 B |
| ③ Arena DMA probe (wall 2) | **shape** is the only axis: strided 2D 108–117 GB/s vs contiguous 72–80; workers flat-to-negative; vote ≤ 4 %. **But isolated rates are 4–7× the in-situ 16–18 GB/s**, so the wall is the layer call's use of the ring, not the engine (rule 11) — ⑥ rewritten, #87 | #77 C |
| ④ Two-reader DDR | **inconclusive on the DSP side** (probe defect, rule 12, #90); CPU side clean: 67.9 → 39.7 GB/s under DSP contention (−41 %). Q11 stays open | #77 ④ |
| Control on `R3CY205ZMND` (12 cells, prompt 512) | NPU decode 18.2–21.3 tok/s, CPU 46.4–54.1, NPU prefill 403–541: the NPU is **2.7× short** of 50 at G=512; the CPU clears 50 at G=64/512 and not at G=1024. Provisional for our unit until #91 | #77 A |

## 3. Open items (candidates for issues; the supervisor promotes them)

| # | item | expected | depends on |
|---|---|---|---|
| ① | ~~Measurement A~~ **measured (#77) → §2.** Follow-on: the dense FC's 28.2 ms/token on the NPU path is the second-largest lever after MoE; it is a round-trip cost, so it folds into ⑨ (one call per token), not into an FC kernel | — | — |
| ② | ~~Measurement B~~ **measured (#77) → marshalling → ⑦ / #88** | — | — |
| ③ | ~~Measurement C~~ **measured (#77) → shape only; in-situ gap is the wall → ⑥ / #87** | — | — |
| ④ | Two-reader DDR probe — **DSP side invalid in #77 (rule 12); refiled as #90** (stream > VTCM+L2 or use the DMA ring, bounded result). CPU side: −41 % under contention, so any split pays less than the sum | decides whether the CPU+NPU split (contract §3.2) can ever pay | #90, ride-along step in a later sitting |
| ⑤ | **Filed as #80.** M=1 MoE path on the existing HVX GEMV (wall 1). The kernel already exists: `hvx/hvx_gemm_u8i4_wh.c` (u8×i4 over WH tiles, int32 bit-identical to HMX, m ≤ 16) is used only by the prefill "tail" path in `hexkl_mm_u8i4_moe.c` (off by default, net −0.5 ms there). Decode needs a dispatch that sends all four experts through it at M=1 with no 64-row block, plus the weight feed (arena read vs DMA into VTCM) that ③ decides | MoE DSP 1.35 → ≈ 0.3 ms/call if DMA ≥ 30 GB/s | ③ |
| ⑥ | **Wall 2 rewritten (#77 C): not descriptor / engine / vote but "why does the MoE call see a quarter of the isolated rate".** Step 1 = #87, instrument the ring use inside `hexkl_mm_u8i4_moe.c` (per-descriptor issue/complete pcycles, wait time in `hexkl_dma_ring_wait`, outstanding depth, actual chunk shapes at M=1 and M>1) + a device gtest that reproduces the in-situ pattern, gate = a table attributing the 4–7× to named causes. Step 2 = the fix that table names (separate issue). Interacts with #80/#86: the M=1 GEMV path reads the arena directly, so its A/B also tells what the ring costs | 16–18 → ≥ 40 GB/s in situ; 1.19 → ≤ 0.57 ms per call | #87 |
| ⑦ | Transport (wall 3) **resolved by #77 B to prebound handles + per-call buffer/marshalling cleanup on plain FastRPC — #88**; #83 narrowed to the document that says what "prebound" concretely means (inventory of per-call bytes, bind/run IDL pair, persistent ION staging; upstream `fb0f02b9` size-class staging is the author's step in the same area). dspqueue / resident worker: one paragraph, revisited only with ⑨ | 0.53 → ≤ 0.1 ms per call (−10 ms/token) | #83 → #88 |
| ⑧ | lm_head blocked Q4_0 twin on device (doc 46 §46): confirm 25.7 → ≈ 3.4 ms | ARM remainder | ① |
| ⑨ | One FastRPC call per token: M=1 RMSNorm, conv1d + gating, RoPE, attention, dense FFN, lm_head on the DSP; per-token entry in the IDL. **Filed as #85 (skeleton entry + op table), #81 (m=1 attention), #82 (RMSNorm, q/k norm, RoPE, conv1d + gating)**; host harness for all of them #84 | removes 22 round trips and the ARM remainder | ⑤ ⑥ ⑦ |
| ⑩ | Registration at load time, bake cache on disk (doc 45 Phase D "P4") | load time, not speed | — |
| ⑪ | Prefill residency (doc 45 B/C/D) | prefill 523 → 700+ | after decode goal |
| ⑫ | CPU+NPU expert split | raises the ceiling only if ④ > 45 GB/s | ④, user decision Q11 |
| ⑬ | (withdrawn 2026-09-21: no simulator in this project, user decision) | — | — |
| ⑭ | `generation(last 64)` in the base report block (**#89**) | fills the contract §1.1 column from the next handoff on | — |
| ⑮ | Anchor sitting on `R3CY10WM83Y` (**#91**, #77 §2 only, ≈ 30 min) | replaces the provisional "now" in BENCHMARK.md and contract §1; measures rule 13's ratio | user's phone time |
| ⑯ | Device A/B of #80 / PR #86 (M=1 GEMV switch on vs off): the first lever against `lfm2_moe` 42.1 ms/token; either bundled with #87's instrumented sitting or run first as a plain A/B — user's choice | MoE DSP 1.35 → ≈ 0.3 ms/call if the arena read keeps up; if not, the number says what ⑥ must deliver | PR #86 merge decision |

## 4. Reusable code on `hvx_impl` (survey 2026-09-21; read with `git show hvx_impl:<path>`)

`hvx_impl` (Qwen3-0.6B, W8A8 HVX-only) is frozen but its kernels and
harness are device-validated. What lifts, in value order:

| lift | from | serves | adaptation |
|---|---|---|---|
| m=1 decode attention with a DSP-resident KV cache (K stored transposed `[head_dim][max_seq]` so one vector covers 64 positions; workers split by kv head) | `nntrainer/tensor/hexagon/htp/ops/hvx-attn.c` (152 LOC) | ⑨ attention | drop the token loop (m=1); per-worker score scratch from the orchestrator; KV dtype vs `hexkl_kv_quant.c` (fp16 there, u8 here → dequant in the score loop if u8 stays); `wp_run` → `hvx_worker_pool_run` |
| "DSP owns the graph": validate the op list once at init, then `forward(tokens, pos → logits)` runs `for op in ops: table[kind]()` with per-op pcycles and one FARF line per call | `htp/htp_graph.{h,c}` (~200 of 546 LOC), `htp/nntr_htp.idl` (3 methods incl. `forward_debug`), `htp/executor.c` (fd mmap + `AEE_EALREADY` handling) | ⑨ one call per token | add a `forward` entry beside the per-op IDL; rebuild `next_mm[]` over the LFM2.5 op sequence; repopulate the op table with MoE/conv1d/dense kinds |
| cross-op weight prefetch: two op-independent VTCM half-slabs so the last chunk of op N kicks chunk 0 of op N+1 while norm/RoPE/attention run in between | `htp/ops/hvx-matmul.c` `mm_slab`/`mm_worker_vtcm`/`mm_pf_kick` (~120 LOC) | ⑥ ⑨ | hvx_impl's per-worker push/pop DMA FIFO vs our global index ring (`hexkl_dma_ring_push2d` + `next_idx`/`wait`): rewrite the pipeline loop; keep the "drain, never abandon" rule for a prefetch left by another op |
| host E2E harness without the CausalLM app: `hexagon_e2e_test` (`--tokens/--chunk/--steps/--eval/--dump-*`, `E2E` lines), `HexagonRunner`, `RpcmemBuffer`, md5-gated `run_e2e_test.sh`, `summ_farf_prof.py`, `find_divergence.py` (needs `forward_debug`), `make_tokens.py` | `test/hexagon/hexagon_e2e_test.cpp`, `nntrainer/tensor/hexagon/host/*`, `tools/hexagon/*` (~650 LOC) | measurement | replace the qwen3 lowering/config with LFM2.5's; `NNTR_HAVE_FASTRPC_MAP_STATIC` probe into `test/htp/build.sh` |
| small ops: RMSNorm (also per-head q/k norm via `FLAG_PER_HEAD`), RoPE (`rope_rotate`, 20 lines, head_dim 128 hard-coded — LFM2.5 is 64), the shared fp32 cos/sin row `nntr_htp_rope.h`, and the quantizer's integer-only rounding recipe (one qf32 product, then sign/exp/significand split, ties-to-even) that made results v75/v79-portable | `htp/ops/hvx-rmsnorm.c`, `hvx-rope.c`, `nntr_htp_rope.h`, `hvx/hvx-quant.h` (~295 LOC) | ⑨ | activation dtype (fp16 there, f32 residual here); our quantizer is asymmetric u8 with zero point — port the rounding only if `hvx_quant_u8.c` still reads sf bits after a qf32 op |

Not lifted: the W8A8 tiled `vrmpy` matmul (wrong layout for int4; our `hvx_gemm_u8i4_wh_col` is the math), `dma-queue.c` (we have `hexkl_dma_ring.c`), the worker pool (ours is richer; note the opposite HVX-context convention: hvx_impl locks a unit per worker and leaves none for the caller, ours runs index 0 inline), the qwen3 lowering/packer/app glue, the simulator harness (no simulator here). Causal depthwise conv1d (L=3) + gating has no counterpart on either branch: net-new.

Silicon rules from `hvx_impl`'s HEXAGON.md §7 that bind here too: compute in fp32 inside an op and narrow to fp16 once (`Vhf_equals_Vqf16` after a qf16 multiply rounds badly, 2.7 % PPL); clamp the SiLU exp argument (already 85 here, doc 44); `-mhvx-ieee-fp` is required for fp16 intrinsics on toolchain 19; int32 `vrmpy` sums are exact in any order, divergence enters only in the epilogue; qf32→sf conversion differs between v75 and v79, so quantizers decode integers, never sf bits; HVX code runs only on threads that own an HVX context; validate `k` bounds for exact int accumulation; never compare wall-clock tok/s across units, report pcycles and `pcycles_per_us`.
