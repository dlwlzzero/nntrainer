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

## 3. Open items (candidates for issues; the supervisor promotes them)

| # | item | expected | depends on |
|---|---|---|---|
| ① | Measurement A: `--profile` decode breakdown of the ARM remainder (FC, mha, lm_head, other) | tells whether FC thread over-splitting (doc 46 §45) is first | first handoff |
| ② | Measurement B: `NNTR_HTP_PROFILE=3` transport floor — wake/clock (→ resident worker / dspqueue) vs marshalling (→ prebound handles + buffer cleanup) | picks the transport fix | first handoff |
| ③ | Measurement C: arena DMA probe — linear vs 2D descriptors, 1–4 engines each `dmstart`, bus vote on/off | explains 18 vs 38 GB/s | first handoff |
| ④ | Two-reader DDR probe: CPU and DSP streaming disjoint buffers concurrently, aggregate GB/s | decides whether the CPU+NPU split (contract §3.2) can ever pay | first handoff |
| ⑤ | **Filed as #80.** M=1 MoE path on the existing HVX GEMV (wall 1). The kernel already exists: `hvx/hvx_gemm_u8i4_wh.c` (u8×i4 over WH tiles, int32 bit-identical to HMX, m ≤ 16) is used only by the prefill "tail" path in `hexkl_mm_u8i4_moe.c` (off by default, net −0.5 ms there). Decode needs a dispatch that sends all four experts through it at M=1 with no 64-row block, plus the weight feed (arena read vs DMA into VTCM) that ③ decides | MoE DSP 1.35 → ≈ 0.3 ms/call if DMA ≥ 30 GB/s | ③ |
| ⑥ | DMA path fix (wall 2) | 1.19 → 0.57 ms floor per call | ③ |
| ⑦ | Transport (wall 3): dspqueue or resident worker. **Options document filed as #83** (dspqueue confirmed present in SDK 6.4.0.1: `incs/dspqueue.h`, `examples/dspqueue`) | 0.57 → ≤ 0.1 ms per call | ② |
| ⑧ | lm_head blocked Q4_0 twin on device (doc 46 §46): confirm 25.7 → ≈ 3.4 ms | ARM remainder | ① |
| ⑨ | One FastRPC call per token: M=1 RMSNorm, conv1d + gating, RoPE, attention, dense FFN, lm_head on the DSP; per-token entry in the IDL. **Filed as #85 (skeleton entry + op table), #81 (m=1 attention), #82 (RMSNorm, q/k norm, RoPE, conv1d + gating)**; host harness for all of them #84 | removes 22 round trips and the ARM remainder | ⑤ ⑥ ⑦ |
| ⑩ | Registration at load time, bake cache on disk (doc 45 Phase D "P4") | load time, not speed | — |
| ⑪ | Prefill residency (doc 45 B/C/D) | prefill 523 → 700+ | after decode goal |
| ⑫ | CPU+NPU expert split | raises the ceiling only if ④ > 45 GB/s | ④, user decision Q11 |
| ⑬ | (withdrawn 2026-09-21: no simulator in this project, user decision) | — | — |

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
