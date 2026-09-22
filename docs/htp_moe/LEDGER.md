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
the HTP by config"), plus the three cherry-picks `fb0f02b9` / `04a2fcc4` /
`b0a384d6` that PR #103 carried (merged by the user as `ad714de7`, cycle 6).
Upstream watch sha: **`5622c743`** (cycle 6).

**New commits since (seen 2026-09-21, cycle 1; PR head `7f81560b`,
updated 08:10 UTC, fast-forward from `2ce38d65`, 31 files, +3490/−636).
Merging them into `htp_moe` is the user's decision (contract §5, Q16);
not merged.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `95caa139` | [docs] 50: the fifth run — every FC group on, per-shape calls measured | doc 50 |
| `fb0f02b9` | [HTP] Stage each call through an ION buffer of its own size class | `htp_compute_ops.cpp` (transport-side, relevant to ⑦ / #83). **Cherry-picked into `htp_moe` by PR #103 (#88)**, `docs/htp_attention` hunk dropped |
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

**Cycle 3 (seen 2026-09-22 ~01:00 UTC): PR head moved `72d9b140` →
`0a0c0402` (updated 2026-09-21 10:27 UTC), eight more commits. Files:
`conv_block_layer.cpp`, `tie_word_embedding.{cpp,h}`, `causal_lm.cpp`,
`hmx/hexkl_conv_block.{c,h}`, `htp_backend.cpp`, docs 00/51. No IDL, ring
or MoE-kernel change; one new runtime knob (`NNTR_HTP_POLL_US`) touches
the FastRPC poll window that #88 (wall 3) also works on. Still not merged;
user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `8d361580` | [CausalLM] conv_block DIFF: K-grouped int4 with the symmetric rule, groups 32-256 | `conv_block_layer.cpp` |
| `2e69c732` | [CausalLM] conv_block DIFF: the grouped-int4 recipes the previous commit described | same |
| `32b46e32` | [CausalLM] NNTR_PPL: the prompt's teacher-forced perplexity at prefill | `causal_lm.cpp`, `tie_word_embedding.{cpp,h}` — a PPL readout we could reuse as an accuracy column |
| `e2554a24` | [docs] 51: perplexity closes the conv block accuracy question | doc 51 |
| `04a2fcc4` | [HTP] NNTR_HTP_POLL_US: the FastRPC poll window as a knob | `htp_backend.cpp` — overlaps #88's area (transport). **Cherry-picked into `htp_moe` by PR #103 (#88)** |
| `07a9bec5` | [CausalLM] conv_block: drop the recipe emulation, keep DIFF/SHADOW and PPL | `conv_block_layer.cpp` |
| `e699e1bd` | [HTP] conv block: pipeline phase 2 one block ahead | `hexkl_conv_block.{c,h}` (prefill-shape fused conv block) |
| `0a0c0402` | [docs] 51: the perplexity split -- the dense FFN costs 4%, the conv block nothing | doc 51: the author's accuracy verdict — the fused conv block is PPL-neutral, the dense FFN through the MoE kernel costs 4 % PPL |

Reading: the conv-block accuracy line has settled (PPL-neutral); the
dense-FFN-through-MoE-kernel path (`d2f0bf47`) costs 4 % PPL on the
author's measure, which would fail our gate (c) if we ever routed the dense
FFN that way. `04a2fcc4` is the one commit with a bearing on our decode
work (wall 3); the rest is prefill/accuracy. A merge decision can still
wait; if merged, the IDL from `7f81560b` forces a skel + app rebuild.

**Cycle 4 (seen 2026-09-22 ~02:30 UTC): PR head moved `0a0c0402` →
`b0a384d6` (updated 2026-09-22 01:54 UTC), three more commits. Files:
`htp_compute_ops.cpp` (profile accounting), `htp_backend.cpp` (FastRPC
poll default), docs 00/51. No IDL, ring, weight-layout or MoE-kernel
change. Still not merged; user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `6f8f7791` | [HTP] Keep the conv row's hidden worker time out of the mm residual | `htp_compute_ops.cpp` — `[HTP-PROFILE]` residual arithmetic for the conv row; prefill-only row, but the same table #94 B reads |
| `faba38d1` | [docs] 51: correct why the dense FFN is expensive -- down's depth, not the chunking | doc 51 |
| `b0a384d6` | [HTP] Poll the FastRPC reply for 5 ms by default | `htp_backend.cpp` — makes `04a2fcc4`'s `NNTR_HTP_POLL_US` knob default to 5 ms. Transport-side (wall 3); the #88 plan read it (author's unit: decode call transport 158 → 83 µs, doc 51 §2.20). **Cherry-picked into `htp_moe` by PR #103 (#88)**, `docs/htp_attention` hunks dropped |

Reading: `b0a384d6` is the second upstream commit in #88's area (after
`04a2fcc4`). Two consequences: (1) the #88 planner reads doc 51's poll
numbers before writing the design; (2) if the user merges before #88's
handoff, variant A of that sitting already carries the 5 ms poll and #88
is read against it — cleaner than merging in between. Nothing here touches
#94's artifact set.

**Cycle 5a (seen 2026-09-22 ~04:30 UTC): PR head unchanged at `b0a384d6`.**

**Cycle 5b (seen 2026-09-22 ~05:30 UTC): PR head moved `b0a384d6` →
`80fa1a1a` (updated 2026-09-22 05:14 UTC), four more commits. Files:
`hmx/hexkl_mm_u8i4_moe.c` (+370/−226), `htp_compute_ops.cpp`,
`Applications/CausalLM/layers/qkv_layer.{cpp,h}`, `lfm2_causallm.cpp`,
`transformer.cpp`, two LFM2 unit tests, doc 51. One commit
(`5731b6e5`) rewrites the MoE kernel's down-matmul pipeline — the same
file and the same DMA schedule that #94's trace measured and that the
wall-2 fix issue (#100) targets. Still not merged; user decision.**
Oldest first:

| sha | subject | touches |
|---|---|---|
| `3006d255` | [HTP] Give plain FC calls their own profile row | `htp_compute_ops.cpp` (profile rows; #94 B reads this table) |
| `5731b6e5` | [HTP] Issue the MoE kernel's down matmul one block behind | **`hexkl_mm_u8i4_moe.c`** — pipelining of the down matmul behind gate/up in the HMX block loop; changes the 46-descriptor issue order #94 traced (rows f/g/h). Prefill-shaped by intent (doc 51's "all-on run"); its M=1 effect is unmeasured |
| `1f538c20` | [CausalLM] Fuse LFM2's q/k/v projections and their norms into qkv_layer | `qkv_layer.{cpp,h}`, `lfm2_causallm.cpp`, `transformer.cpp`, tests — ARM-side attention projections (the ARM remainder, ⑨ / #81 area) |
| `80fa1a1a` | [Docs] Record the 585 ms all-on run: pipeline and qkv sum to −33 | doc 51: the author's prefill at 585 ms with everything on (−33 ms from the two commits above) |

Reading: `5731b6e5` is the first upstream commit inside the MoE kernel
since `htp_moe` froze. If merged, #94's trace / attribution table
(descriptor order, `first expert ready`, rows f/g/h) must be re-taken as
variant A of the next sitting before #100 is read against it; if not
merged, #100 works on the frozen list and a later merge re-opens the
question. Either way the planner of #100 reads that diff first. The GEMV
path (C) bypasses this loop entirely, so ⑯'s verdict is unaffected.

**User merge decision (cycle 6, 2026-09-22): PR #103 (#88) merged by the
user as `ad714de7`.** That merge is the user accepting the three upstream
cherry-picks `fb0f02b9` (size-class ION staging), `04a2fcc4`
(`NNTR_HTP_POLL_US`) and `b0a384d6` (5 ms poll default) into `htp_moe`,
with their `docs/htp_attention` hunks dropped. Nothing else from #4327 is
merged; the rest of the tables above and below stays a user decision.

**Cycle 6 (seen 2026-09-22 ~06:30 UTC): PR head moved `80fa1a1a` →
`5622c743` (updated 05:53 UTC), four more commits. Net files:
`hmx/hexkl_mm_u8i4_dma.c` (+104/−37), `hmx/hexkl_mm_u8i4_moe.c`
(+8/−3, a comment and one moved line), `htp_compute_ops.cpp` (+6, a
comment), `test/htp/host/fc_layer_host_check.c` (new), host stubs, docs
00/51/52. No IDL change; the M=1 GEMV path (#105) is not touched. Not
merged; user decision.** Oldest first:

| sha | subject | touches |
|---|---|---|
| `7e0b0e71` | [HTP] Three small exposures: early gate_up chunks, pooled FC epilogue, MoE args in rpcmem | moe, dma, compute_ops — two of the three reverted by the next commit |
| `aa371dd2` | [HTP] Revert the early gate_up chunks and the rpcmem MoE args; keep the pooled FC epilogue | net: `hexkl_mm_u8i4_dma.c` plain-FC epilogue double-buffered onto the pool (prefill FC calls, not on our decode config); two **measured negatives** left as comments: (a) pushing the next expert's gate_up chunks early put the activation behind them in the ring, GATHER 84 → 747 µs/call (388 at decode), "the DMA moves ~12 GB/s beside the HMX, not the idle 33" (doc 51 §2.26) — input for #100 / rows f–g; (b) the MoE call's five small sequences from one rpcmem buffer: transport 627 → 605 µs, noise, "the gap to the FC calls' ~400 is the call's length … a call longer than the poll window ends in the interrupt-driven wait" — input for reading handoff 88 |
| `7ada48b7` | [Docs] Record the pooled-FC-only run and the poll 9000 experiment | doc 51 (a longer poll window tried) |
| `5622c743` | [Docs] 52: the flash-experts (cached-slim on HTP) task, self-contained | new doc 52: the author's next task is **memory** (peak RSS 5.2 GB, 3840 MiB expert arena → LRU/flash-loaded experts on the HTP path, "DSP kernel untouched"); records the author's all-on state as prefill 585–616 ms / decode 24.2 TPS and "no HTP software lever left ≥ 5 ms" for prefill |

Reading: nothing here changes a decode kernel. Two of the comments are
free measurements for our open work: (a) a ~12 GB/s ring rate beside the
HMX, which bears on #100; (b) an rpcmem-args null, which bears on #88. Doc
52 says the author's line is now moving to memory residency. If that line
lands, it would change the arena our GEMV path reads directly (#105, ㉒),
so it is worth knowing about before any merge.

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
13. **Any unit; only a same-sitting A/B is a verdict** (user decision
    2026-09-22, contract §12; supersedes the earlier "unit `R3CY10WM83Y`
    is the anchor" form of this rule). Handoffs name no serial; the
    filled handoff records the serial that ran; every BENCHMARK row is
    unit-tagged; cross-sitting numbers are compared only as same-unit
    drift when the unit happens to be the same (rule 9). The second-unit
    anchor and the per-cell unit ratio (#91, ⑮) are withdrawn as goals.
    Two S25 Ultra units were 6–8.6 % apart in `hvx_impl`; that fact stays
    as the reason for the rule.
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

Learned in this project (#94 first attempt / #97, unit `R3CY205ZMND`,
2026-09-22):

17. **Gate 2 must include an undefined-symbol check of the skel.**
    `-Wall -Werror` clean is not "links": a Hexagon shared object links
    with unresolved symbols by default, and the device loader then fails
    `dlopen` with `0x80000406` / `dlerror RX VA 0xFFF00000 outside ELF
    segment` — a message that reads like a segment-layout bug and is not
    one. #94's skel from `2a75f7d9` had seven `U hexkl_dma_trace_*`
    (`hexkl_dma_trace.c` from PR #93 never entered `test/htp/build.sh`
    `SRCS`; the host check compiled it on its own, so gate 1 passed).
    From now on gate 2 = `-Wall -Werror` **and** `hexagon-nm -u -D
    build/libnntr_hvx_skel.so` shows no `U` outside the runtime set
    (`HAP_*`, `compute_resource_*`, `qurt*`, libc, compiler-rt); #97 puts
    that check (or `-Wl,--no-undefined`) into `build.sh`. Second artifact
    found broken only on the device (rule 14 was the first): every new
    `.c` under `htp_backend/` is checked against `build.sh` at review.
    Distinguish the three loader errors: `0x8000040E` = stale skel vs IDL
    (rule 3), `htp Context is not registered` = app-side, `0x80000406` =
    the skel itself does not load.

Learned in this project (#94 sitting 2, unit `R3CY205ZMND`, 2026-09-22):

18. **The isolated probe rate drifts 3–19 % with the phone's temperature
    inside one sitting** (`DMA_PROBE` shape iii, 1 worker: 106.9 GB/s at
    29 °C, 88.8 at ~51 °C) while the tok/s cells of the same sitting did
    not collapse (CPU 47–53). A probe reading is quoted with its
    `thermal_zone0`; the gap analysis uses the cool reading as the
    achievable ceiling and the hot one as the run of record.
19. **The real chunk list is slow on its own** (provisional on #99): the
    traced 46-descriptor / 32-region M=1 list replayed unpaced, one
    worker, no compute, no competitor runs at 40.2 GB/s where a probe of
    the identical 8 KiB × 64 @ 56 KiB shape does 106.9 on the same unit
    and skel; paced to the in-situ schedule it reproduces the in-situ
    15.6–18.6 GB/s. Rule 11's 4–7× therefore splits into ≈ 2.7× in the
    list itself and ≈ 1.3–1.8× in the interleaving (rows f + g ≈ 20 % of
    `dsp_us`). More workers make the replay slower (40.2 → 39.1 → 36.5).
20. **NPU tok/s drifts −7..−16 % day to day on one unit while the DSP
    profile columns move ≤ 4 %** (#94 s2 vs #77, same unit and forward
    path): the tok/s cells carry host and thermal state, the
    `[HTP-PROFILE]` `dsp=` / `mm` columns carry the kernel. Cross-sitting
    kernel claims are read from the profile columns, tok/s only inside a
    sitting.
21. **Rule 14, refined by the cycle-5 decision on #94 Deviation 5:** a
    device set whose md5s differ from the staged table but whose
    provenance is intact (commit + build line + `md5sum` recorded, DSP
    sources shown identical by `git diff`) is **accepted as the sitting's
    provenance** when every variant of the sitting is the same binary set
    switched by an environment variable or config key (A and C of #94).
    "Void once a B variant is included" applies only when a variant is a
    *different binary* whose staged md5 the device does not match. Also
    from that sitting: `test/htp/build.sh` picks the newest directory
    under `$HEXKL_ROOT/lib` when `HEXKL_SDK_VER` is unset — a workstation
    with 6.6.0.0 installed silently links another HexKL; every handoff's
    build line carries `HEXKL_SDK_VER=6.4.0.1`.

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
| ① ARM remainder / per-type decode cost (ms/token, NPU vs CPU, unit `R3CY205ZMND`) | `lfm2_moe` **42.1 vs 22.3**, `fully_connected` **28.2 vs 10.3**, `output_of_causallm` 2.85 vs 3.40, `mha_core` 2.30 vs 2.81, all others ≤ 0.2; MoE + dense FC = 92 % of the NPU token, 83 % of the CPU's. The MoE FFN on the NPU costs ~2× the CPU path; the dense FC routed through the HTP 2.7×. lm_head is the one place the NPU already wins, so ⑧ (lm_head twin) is not first. **Correction (cycle 3, guide-writer's reading of the config):** the #78 `q40-qs4cx-wh` `nntr_config.json` routes **no FC to the HTP** — `lfm2_causallm.cpp:337–348` defaults `attn_proj_engine`, `conv_in_proj_engine`, `conv_out_proj_engine`, `dense_ffn_engine` to `cpu` and #78 set only `moe_engine`. So the NPU run's `fully_connected` 28.2 vs 10.3 ms/token is **not** an HTP round trip; the cause is open (candidates: doc 48 §2 ③ FC thread over-splitting when the 8 CPU threads contend with the FastRPC poll thread, or the CPU clock dropping while the DSP runs). Decidable with one extra profile cell: the NPU model at `NNTR_NUM_THREADS=4` (FC row moves → threading), or the non-WH `QS4CX` model with `NNTR_MOE_HTP_DECODE` off vs on (the `q40-qs4cx-wh` model has no CPU MoE fallback, rule 6, so it cannot be the control). Open item ⑰ | #77 A (profile); correction from the guide (PR #92) |
| ② Transport floor (wall 3) | **marshalling**: 527.7 µs/call at level 3 vs 587.7 at level 2 (−10 %, inside ±15 %; wake/clock would have been ≤ 300), `qos_mode` 2 both. **Refined by the #88 plan (host inventory, not yet measured):** cache maintenance on the over-sized staging pair (decode's 8 KiB rode a 4 MiB cached ION pair after prompt 512; 36–59 µs/MB, doc 50 §3.6) plus the 100 µs poll-window fallback to the interrupt path; the per-call arguments proper are 464 non-ION bytes (48 B primitive block + 5 sequences) and cannot carry 500 µs. Fix taken first = upstream `fb0f02b9` + `04a2fcc4` + `b0a384d6` (PR #103); prebind is conditional on that measurement (plan 88 §3.3 / §4 step 5); dspqueue / resident worker deferred to one paragraph (#83 §4) | #77 B; plan 88 §2.1 |
| ③ Arena DMA probe (wall 2) | **shape** is the only axis: strided 2D 108–117 GB/s vs contiguous 72–80; workers flat-to-negative; vote ≤ 4 %. **But isolated rates are 4–7× the in-situ 16–18 GB/s**, so the wall is the layer call's use of the ring, not the engine (rule 11) — ⑥ rewritten, #87 | #77 C |
| ④ Two-reader DDR | **inconclusive on the DSP side** (probe defect, rule 12, #90); CPU side clean: 67.9 → 39.7 GB/s under DSP contention (−41 %). Q11 stays open | #77 ④ |
| Control on `R3CY205ZMND` (12 cells, prompt 512) | NPU decode 18.2–21.3 tok/s, CPU 46.4–54.1, NPU prefill 403–541: the NPU is **2.7× short** of 50 at G=512; the CPU clears 50 at G=64/512 and not at G=1024. Provisional for our unit until #91 | #77 A |
| **⑯ M=1 HVX GEMV (PR #86, `NNTR_MOE_HTP_M1_GEMV=1`) — variant C of #94 sitting 2, same binary and sitting as A** | **Goal progress, accuracy gate passed.** Decode +6.2 % (G=64, 18.83 vs 17.72), +7.7 % (G=512, 18.33 vs 17.03), +1.7 % (G=1024, no change); text byte-identical to A at every G and run; gtest `MoeLayerM1GemvMatchesHmx` bit-identical (0 bad of 2048 / 8192). Profile M==1: dsp **1414 → 1044 µs/call (−26 %)**, `DMA ring:` 46 → 2 descriptors, wait 269 → 4.3 µs, gather 146.5 → 0 — the DMA wait is gone from the path — **but `mm` 783 → 975 µs (+24 %)**: the GEMV reads the arena directly at ≈ 22.6 GB/s effective (22 MB in 975 µs), slower than the HMX loop's staged DMA + compute. Transport 649 → 748 µs/call (+100 µs, unexplained; `moe_set_opts` is once per session, so it is per-call cost of the switched path — #88 reads it). Prefill untouched (M>1 dsp 0.13 % apart). **Next lever inside ⑯: the GEMV's weight feed** (prefetch the expert tiles into VTCM ahead of the `vrmpy` loop instead of the direct arena read) — it must not re-import row h's slow list, so it waits for #100 (open item ㉒). Provenance: device skel `d6568c8b…` from `08afbb10` on the user's workstation, accepted under rule 21 | #94 s2 §C, `dad0f476` |
| **Plan 87 §3.2 attribution a–h (wall 2): in-situ B levels 2/3 + `MoeChunkReplay` + `DMA_PROBE`, one sitting** | **a refuted** (wait 10.2 / 19.0 % of dsp, depth 9–11), **b refuted** (pushed shapes = probe iii), **c refuted, provisional #99** (workers 1/2/4 → 40.2 / 39.1 / 36.5 GB/s), **d sensitivity, not cause** (DDR competitor +1620 % wait, HVX competitor 0), **e refuted, sign reversed** (expert 0 fastest), **f confirmed 9.7 / 10.3 %** (the single blocked `site=act` wait = gather), **g real 11.0 / 7.1 %, unexplained** (`fresh` and `gap_us` controls both null), **h new, ≈ 60–70 % of the gap, provisional #99** (rule 19). **Wall-2 fix = row h → #100**; f (hide the first expert's 3.5 MiB behind gather / the previous layer) and g (cold-call penalty) are the remaining ~20 % and are filed after h. Plan 87 §0 expected f + g large: they are real but small; h was not foreseen | #94 s2 attribution table |
| Control, #94 sitting 2 (`R3CY205ZMND`, 13 A + 6 C cells, one binary set `2a75f7d9` + #98 skel) | NPU **17.72 / 17.03 / 17.30**, CPU **52.43 / 49.22 / 48.31**, NPU prefill 389–527: **the "now"** (BENCHMARK, contract §1). Distance to 50: 2.9× at G=512 (2.7× with C). Same-unit drift vs #77: CPU −9..+4 %, NPU −16..+5 % per cell with DSP profile columns ≤ 4 % apart (rule 20) | #94 s2 §A/§B |
| CPU control re-run on `R3CY205ZMND` (#94 first attempt, 6 cells, next day) | 53.0 / 53.5 · 51.1 / 51.2 · 49.7 / 48.4 tok/s vs #77's 54.1 / 52.9 · 52.7 / 52.0 · 46.4 / 48.5: **one unit drifts ≤ 7 % per cell day to day** (rule 9 quantified for the CPU path); G=1024 stays 46–50, i.e. the CPU does not reliably clear 50 there. No NPU cell (#97). Not the anchor; not a verdict on any lever | #94 `ec7ad296` |

## 3. Open items (candidates for issues; the supervisor promotes them)

| # | item | expected | depends on |
|---|---|---|---|
| ① | ~~Measurement A~~ **measured (#77) → §2.** Follow-on: the dense FC's 28.2 ms/token on the NPU path is the second-largest lever after MoE; **it is not a round-trip cost** (no FC is on the HTP in that config, §2 correction) — see ⑰ | — | ⑰ |
| ⑱ | **Accuracy, filed as #95 (p1):** Hadamard rotation on the MoE down_proj input — fold `Hᵀ·W_down` offline (new dtype `QS4CX_WH_HAD`, block 256, 1792 = 7 × 256, 1/16 both sides), FWHT-256 in IEEE `sf` add/sub on the DSP right before the existing u8 requant at all three requant sites of `hexkl_mm_u8i4_moe.c`. Targets the recorded gap (doc 43, 2026-09-09: 5 of 32 MoE calls at 67–80 dB SNR, each from exactly one of 1792 elements crossing a u8 level — boundary rounding, not row outliers; column-wise outliers never measured). **First accuracy item in this table**; it does not move tok/s by design and may show no effect — a "no effect" result closes it with the numbers. Gate = `NNTR_L2_DIFF` per-call SNR / `total_flips` on vs off on the 5 recorded calls + full-model handoff with prefill/decode TPS and text-identical-to-CPU; prefill gate applies (it touches the weight layout and the requant stage). Rules that bind: no qf32 (v75/v79), host scalar `fwht_rows_f32_ref` bit-identical to the HVX kernel, CPU `QS4CX` run stays the reference (no CPU kernel for `_HAD`, rule 6 applies to it too) | requant SNR on the 5 bad calls ↑; text-identical-to-CPU unchanged or better; TPS within noise | none; can run in parallel with the walls (touches quantizer + requant only) |
| ② | ~~Measurement B~~ **measured (#77) → marshalling → ⑦ / #88** | — | — |
| ③ | ~~Measurement C~~ **measured (#77) → shape only; in-situ gap is the wall → ⑥ / #87** | — | — |
| ④ | Two-reader DDR probe — **DSP side invalid in #77 (rule 12); refiled as #90** (stream > VTCM+L2 or use the DMA ring, bounded result). CPU side: −41 % under contention, so any split pays less than the sum | decides whether the CPU+NPU split (contract §3.2) can ever pay | #90, ride-along step in a later sitting |
| ⑤ | **Filed as #80.** M=1 MoE path on the existing HVX GEMV (wall 1). The kernel already exists: `hvx/hvx_gemm_u8i4_wh.c` (u8×i4 over WH tiles, int32 bit-identical to HMX, m ≤ 16) is used only by the prefill "tail" path in `hexkl_mm_u8i4_moe.c` (off by default, net −0.5 ms there). Decode needs a dispatch that sends all four experts through it at M=1 with no 64-row block, plus the weight feed (arena read vs DMA into VTCM) that ③ decides | MoE DSP 1.35 → ≈ 0.3 ms/call if DMA ≥ 30 GB/s | ③ |
| ⑥ | **Wall 2 — measured by #94 sitting 2 (§2 attribution): the 4–7× is ≈ 2.7× in the traced chunk list itself (row h, rule 19, provisional on #99) and ≈ 1.3–1.8× in the interleaving (f + g ≈ 20 % of `dsp_us`). Step 2 filed as #100 (row h: why the 46-descriptor / 32-region list replays at 40 GB/s where probe iii does 107; gate = replay ≥ 80 % of probe iii and in-situ `engine GB/s` ≥ 40; blocked by #99 until the replay's content check is explained). Rows f and g follow #100.** History: rewritten by #77 C — not descriptor / engine / vote but "why does the MoE call see a quarter of the isolated rate". Step 1 = ~~#87~~ **landed on `htp_moe` as `b6ebc2b7` (PR #93, 2026-09-22; #87 closed)**: `hmx/hexkl_dma_trace.{c,h}` (static tables, union-of-intervals busy / depth / blocked-wait arithmetic, host-checked by `dma_trace_host_check`), trace hooks in `hexkl_mm_u8i4_moe.c` behind `hexkl_probe_on` (byte-identical output on vs off, 19 descriptors traced at the fixture shape, 46 at the LFM2 M=1 shape), IDL entries `dma_probe` / `moe_dma_trace_read` / `dma_replay` (`test/htp/nntr_hvx_dma_probe.c`), the header-only in-situ descriptor plan `test/htp/nntr_moe_dma_plan.h`, the second `weight DMA:` line and the `[HTP-DMA]` per-descriptor dump in `[HTP-PROFILE]` (`htp_compute_ops.cpp`, first `NNTR_HTP_DMA_TRACE` calls, default 3), and `TEST_F(HvxDmaProbe, MoeChunkReplay)` (workers 1/2/4 × HVX load × fresh/gap) in `unittest_hvx_dma_probe`. **The device sitting that fills the attribution table (plan 87 §1, hypotheses a–g) is #94 (sitting 2); its first attempt never reached the NPU because the trace's own `hexkl_dma_trace.c` was missing from the skel build (#97, rule 17).** Original scope for reference: instrument the ring use inside `hexkl_mm_u8i4_moe.c` (per-descriptor issue/complete pcycles, wait time in `hexkl_dma_ring_wait`, outstanding depth, actual chunk shapes at M=1 and M>1) + a device gtest that reproduces the in-situ pattern, gate = a table attributing the 4–7× to named causes. Step 2 = the fix that table names (separate issue). Interacts with #80/#86: the M=1 GEMV path reads the arena directly, so its A/B also tells what the ring costs | 16–18 → ≥ 40 GB/s in situ; 1.19 → ≤ 0.57 ms per call | #87 |
| ⑦ | Transport (wall 3) **resolved by #77 B to prebound handles + per-call buffer/marshalling cleanup on plain FastRPC — #88**. **PR #103 (`htp/88-moe-call-marshalling`, merged by the user as `ad714de7`, cycle 6): size-class staging + 5 ms poll cherry-picked, `staging:` inventory line added; handoff `88-moe-call-marshalling.md` being measured now (A = `08afbb10`, B = PR, C = B + `NNTR_HTP_POLL_US=100`)**; #83 narrowed to the document that says what "prebound" concretely means (inventory of per-call bytes, bind/run IDL pair, persistent ION staging; upstream `fb0f02b9` size-class staging is the author's step in the same area). dspqueue / resident worker: one paragraph, revisited only with ⑨ | 0.53 → ≤ 0.1 ms per call (−10 ms/token) | #83 → #88 |
| ⑧ | lm_head blocked Q4_0 twin on device (doc 46 §46): confirm 25.7 → ≈ 3.4 ms | ARM remainder | ① |
| ⑨ | One FastRPC call per token: M=1 RMSNorm, conv1d + gating, RoPE, attention, dense FFN, lm_head on the DSP; per-token entry in the IDL. **Filed as #85 (skeleton entry + op table), #81 (m=1 attention), #82 (RMSNorm, q/k norm, RoPE, conv1d + gating)**; host harness for all of them #84 | removes 22 round trips and the ARM remainder | ⑤ ⑥ ⑦ |
| ⑩ | Registration at load time, bake cache on disk (doc 45 Phase D "P4") | load time, not speed | — |
| ⑪ | Prefill residency (doc 45 B/C/D) | prefill 523 → 700+ | after decode goal |
| ⑫ | CPU+NPU expert split | raises the ceiling only if ④ > 45 GB/s | ④, user decision Q11 |
| ⑬ | (withdrawn 2026-09-21: no simulator in this project, user decision) | — | — |
| ⑭ | `generation(last 64)` in the base report block (**#89**) | fills the contract §1.1 column from the next handoff on | — |
| ⑮ | ~~Anchor sitting on a named unit~~ **withdrawn (user decision 2026-09-22, rule 13): any unit, same-sitting A/B only.** #94 sitting 2 (`dad0f476`, `R3CY205ZMND`) is the "now" in BENCHMARK.md and contract §1; no second-unit issue is opened | — | — |
| ⑯ | **Measured → §2 (goal progress +6.2 / +7.7 / +1.7 %, text identical).** Follow-ups: make the GEMV path the default (**#101**, so every later variant A carries it); its compute side (**#105**, ㉒) and weight feed (㉒ remainder). History: Device A/B of #80 / PR #86 (M=1 GEMV switch on vs off): the first lever against `lfm2_moe` 42.1 ms/token. **Cycle 4: PR #86 merged as `2a75f7d9` (2026-09-22 02:00 UTC) before #94's set was pushed, so it rides as variant C (`NNTR_MOE_HTP_M1_GEMV=1`, 6 NPU cells + one level-2 run + `*MoeLayerM1GemvMatchesHmx*`) of sitting 2 — handoff rebuilt from `2a75f7d9` (`htp/94-sitting2-anchor-trace` @ `8029b76e`). A on that head = the same binary with the switch off (`[HTP] moe m1 gemv: off (applied=0x0)`, `blocks=5632 m1_gemv=0/1408`); a C log that prints `on` in an A cell voids the run.** | MoE DSP 1.35 → ≈ 0.3 ms/call if the arena read keeps up; if not, the number says what ⑥ must deliver | #97 (skel loads) → #94 (user's phone time); first attempt blocked before any NPU cell |
| ⑲ | ~~#97~~ **closed (PR #98 merged `14f65120`; confirmed on silicon by #94 s2's `DmaProbeShapes` PASSED as the first device command).** Skel from `2a75f7d9` did not load (`0x80000406`): `hexkl_dma_trace.c` absent from `test/htp/build.sh` `SRCS` → seven undefined `hexkl_dma_trace_*`. Fix = add the file + an undefined-symbol guard in `build.sh` (rule 17); then #94's artifact set is rebuilt and the handoff re-issued (same document, new md5s) | unblocks every NPU cell of #94 (⑥ ⑮ ⑯ ⑰) | none |
| ⑳ | **Filed as #99 (p1).** `MoeChunkReplay` fails its content check on all 11 cells (`res[6]=1467840` vs `want_sum=9461760`, `bytes_per_call` 22560768 vs the in-situ 22020096, `regions=32`) while `DmaProbeShapes` passes in the same binary. Decides whether the replay timings (rows c, d, g, h; rule 19) stand or must be re-measured; **#100 is blocked on it** | rows c/d/g/h confirmed or re-measured | none |
| ㉑ | **Filed as #102 (p2).** Level-2 `[HTP-PROFILE]` M==1 row with the GEMV path on: `swiglu 5601.2`, `rest<=-5588.2 (-311.8% of host)`, no `weight DMA:` line — the GEMV path's stage slots are mis-attributed in the print (`htp_compute_ops.cpp` / the 30-slot stage table of #86). Cosmetic, no tok/s cell affected, but the B row of every sitting after #101 is unreadable until fixed | a readable M==1 row under `NNTR_MOE_HTP_M1_GEMV=1` | #101 makes it urgent |
| ㉒ | **Split (user direction, cycle 6, 2026-09-22). Compute side filed as #105 (p0, `state:needs-plan`):** one-row `gemm_rows1` at `m = 1` (drops `gemm_rows4`'s three dead accumulators; accumulation order unchanged, so int32 stays bit-identical to the HMX), `l2fetch` lead / distance, and the lane split over the pool. It needs no ring, so it does not wait for #100. The handoff sets `NNTR_MOE_HTP_M1_GEMV=1` explicitly in A and B, which keeps it independent of #101. **Feed side (remainder, not yet an issue):** VTCM staging of each expert's `wh_bytes` by DMA (two linear descriptors per expert); it must not re-import row h's slow list, so it is filed once #100 names h's cause. Evidence: C's `mm` 783 → 975 µs, a direct arena read at ≈ 22.6 GB/s effective, lane-bound (`swiglu / mm = 5.75`; plan 101 §3.4). Supervisor's estimate, not measured: the dead `vrmpy` are ~50–100 µs of the 975, and `hvx_impl` #59 measured direct DDR vector reads at 21–25 GB/s against 37 through the ring. So #105 alone may stop above 600; its ride-along microbench (arena vs VTCM copy × 4-acc vs 1-acc) sizes the feed half. Combined gate for ㉒ = `mm` ≤ 600 µs at M==1 with text identical | MoE dsp 1044 → ≈ 700 µs/call (−7 ms/token) | #105 (compute); feed: #100 |
| ⑰ | **`fully_connected` 28.2 vs 10.3 ms/token on the NPU run with no FC on the HTP** (§2 ① correction). Not a round trip; candidates: CPU thread over-splitting / contention with the FastRPC poll thread (doc 48 §2 ③), CPU DVFS while the DSP runs. 18 ms/token is the second-largest single lever after MoE and needs no DSP code if it is a threading matter. Decide with one extra profile cell (`NNTR_NUM_THREADS=4` on the NPU model, or the non-WH `QS4CX` model with `NNTR_MOE_HTP_DECODE` off/on) — candidate ride-along for #94 or its own issue — **#94 s2 skipped the ride-along (budget)**, so this still needs its cell | up to −18 ms/token on the NPU path | next sitting's ride-along |

## 3a. Guide and tooling notes

* **`docs/htp_moe/guide/`** exists since PR #92 (`cffcec3e`, merged
  `750428e8`, 2026-09-22): five self-contained English pages
  (`index`, `01-run-it`, `02-decode-path`, `03-performance`,
  `04-glossary`), decode only. The guide writer refreshes it on every
  filled handoff and every merged kernel/app PR (contract §8 step 4); its
  numbers are copied from BENCHMARK.md, never measured.
* Guide-writer findings folded into this ledger: (a) the FC 28.2 vs 10.3
  ms/token question (§2 ① correction, ⑰); (b)
  `Applications/CausalLM/install_android.sh` takes **no `--model`
  argument** in this tree — it pushes binaries and prints the `adb push`
  of a model dir as a hint (`install_android.sh:268–279`); handoffs must
  spell out the model push themselves (#77's did).

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
