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
Upstream watch sha: **`a996b4bf`** (cycle 7; unchanged in cycles 8 and 9).

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

**Cycle 7 (seen 2026-09-22 ~07:30 UTC): PR head moved `5622c743` →
`a996b4bf` (updated 06:55 UTC), one commit, docs only
(`docs/htp_attention/00_START_HERE.md` +1, new
`docs/htp_attention/53_int_requant_task.md` +114). No code, IDL or kernel
change. Not merged; user decision.**

| sha | subject | touches |
|---|---|---|
| `a996b4bf` | [Docs] 53: the int32-to-u8 requantization task (arXiv 2511.11248), self-contained | new doc 53 (Korean): a task note for replacing the MoE kernel's int32 → f32 → u8 epilogue by a direct int32 → u8 requant; the paper itself not yet read. Its own reading: at prefill the epilogue is hidden under the HMX (0.35 ms/call exposed, 2.4 %), the gain would be structural (VTCM `gate_off` 448 → 112 KB, the row-scan barrier) and the gate is PPL (`NNTR_PPL=1`, 62.09 ± 0.5 %); **at decode (M=1) "not applicable"** — the author's M=1 call is 1.44 ms with dequant 1.1 µs, gather 128, acc 259, mm 764 |

Reading: nothing for the decode goal. It shares the requant stage with #95
(⑱, Hadamard before the u8 requant, PR #106), so whoever reads #95's
verdict should know an int-requant rewrite of the same stage is the
author's next prefill topic; a merge would conflict there, not in the M=1
GEMV path.

**Cycle 8 (seen 2026-09-22 ~08:30 UTC): PR head unchanged at `a996b4bf`
(updated 06:55 UTC).** One relevant fact from the #95 sitting: the
side tree `htp_hadamard` (upstream `3006d255` + tooling) reads decode
**25.08 / 24.40 / 23.83 tok/s** on its own control A (`R3CY205ZMND`,
same day). `htp_moe`'s #88 B read 24.50 / 23.80 / 23.52 in another
sitting. Rule 9 forbids reading this as a verdict. It is only
consistent with both trees carrying the same transport fix (`fb0f02b9` /
`b0a384d6`) and neither having the GEMV on.

What this changes for us: (1) the "net-new" causal conv1d + gating of §4
now exists upstream in prefill shape — #82 lifts it instead of deriving
it; (2) the ARM decode path of the conv block is still the CPU one there
(the fused call runs only at prefill); (3) doc 50 §3.8 re-measured the
author's unit 10–16 % slower on the same binary (prefill 482–438 tok/s,
not 523), one more reason the provisional "now" is replaced only by #77's
control run on ours; (4) the IDL changed, so a merge forces a
skel + stub rebuild and new md5s in BENCHMARK.md.

**Cycle 9 (seen 2026-09-22): PR head unchanged at `a996b4bf` (updated
06:55 UTC).** Nothing to decide.

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

Learned in this project (#88, unit `R3CY205ZMND`, 2026-09-22):

22. **Handoff artifacts must be reproducible from the commit on any
    workstation; the staging path is not a deliverable.** Twice now (#94
    s2's skel, #88's whole A/B set) the sitting ran on a workstation where
    `/local/mnt/workspace/htp_moe/<n>/` and the implementer's worktrees did
    not exist, and the user rebuilt from the named commits. A handoff
    therefore carries, per variant: the commit, the exact build line
    (`build_android.sh --htp` then `--htp --cache`, `test/htp/build.sh`
    with `HEXKL_SDK_VER=6.4.0.1`, env var names, no absolute worktree
    paths), and the md5s of the non-compiled inputs (model, tokenizer,
    prompt, `libc++_shared.so`, `libsdkl.so`), which must match anywhere.
    **Rule 21 extended to binary B variants:** a sitting whose variants are
    *different binaries* rebuilt from their named commits is accepted
    when (a) every results row carries the device md5 of the binary that
    differs between variants, (b) the source diff between the variant
    commits is exactly the PR under test (`git diff A B -- nntrainer
    Applications test`), and (c) shared DSP artifacts are one file for
    all variants. It stays void when a device md5 matches neither the
    staged table nor a recorded rebuild. #88 met (a)–(c) (device
    `libnntrainer.so` `5db71a2e…` A / `3cf06450…` B in every row; diff
    `08afbb10..f3e99176` = `htp_backend/{htp_backend.cpp,
    htp_compute_ops.cpp, htp_rpcmem.h}` only; one skel `2bd7311f…`).
23. **The FastRPC `transport` column drifts across sittings like tok/s;
    only `dsp=` is a cross-sitting column.** Identical sources
    (`2a75f7d9` = `08afbb10` outside `build.sh`), same unit, same day,
    ≈ 2 h apart: #94 s2 A level 2 M==1 host / dsp / transport 2062.5 /
    1414.0 / 648.6 µs, #88 A 1812.1 / 1410.8 / 401.3 — dsp −0.2 %,
    transport −38 %. Rule 20 extends to `host` and `transport`: a
    transport claim is an A/B inside one sitting (as #88's was), never a
    comparison against an earlier sitting's row. Corollary for budgets:
    the time **outside** the MoE call (TPS token minus 22 × host) moved
    with the poll/staging variant too — 13.5 ms (A) / 11.4 (C) / 8.4 (B)
    at G=64 — so ≈ 5 of #88's 12.6 ms/token gain landed outside the
    call's own timer (candidates: the 5 ms poll keeps the calling core
    and its cluster clock up between calls; cache maintenance around the
    call; not decided).
24. **In an `m1_gemv=calls/calls` profile row, `swiglu` is Σ lane-time
    over the worker pool (6 lanes on v79), not a stage;** `swiglu / mm`
    is lane utilisation (#94 C: 5601.2 / 974.7 = 5.75). Since PR #108 the
    row's `rest` leaves it out and the `weight DMA: n/a` line prints the
    ratio; before that PR a GEMV row's `rest` is meaningless (#102).

Learned in this project (#95, side tree `htp_hadamard`, unit
`R3CY205ZMND`, 2026-09-22):

24. **v79 HVX IEEE `sf` arithmetic keeps subnormals. It does not flush
    them to zero.** `HvxFwht.MatchesScalarBitExact` fed a row of ±1e-39 /
    ±1e-40 (all subnormal). The DSP (`Q6_Vsf_vadd/vsub_VsfVsf`, then
    `Q6_Vsf_vmpy_VsfVsf` by 1/16, `-mhvx-ieee-fp`) returned
    `0x1.7f4b4p-127` at index 1. That is exactly 8·(1e-39 + 1e-40), the
    unflushed IEEE result, and itself subnormal. The scalar spec
    (`fwht_det_ftz` on every input and result) returned `+0`. Only the two
    non-zero outputs of that row differed (`bad_gated = bad_subnormal_row
    = 2`). The spec's premise ("HVX sf flushes them") was host reasoning,
    and the device refutes it: this is not a flushed-zero *sign*. A `_det`
    scalar spec for HVX `sf` code is plain IEEE RNE with no FTZ, and the host
    side must not run under a flush either (`-ffast-math` /
    `crtfastmath` can set aarch64 FPCR.FZ). Fix carried by #110. The
    same test's overflow row (±3e38, `bad_overflow_row=127 of 256`) is
    ungated and not yet explained (inf/NaN encoding is the candidate).
25. **A synthetic single-shape gtest cannot give an accuracy verdict. It
    gates bit-exactness only.** `MoeLayerHadamardMatchesTwoCallReference`
    matched its two-call host reference bit for bit (`bad_elems=0 of
    409600`, `max_ulp=0`). It still failed its own SNR assertion: rotated
    17.51 dB vs unrotated 23.32 dB, −5.8 dB on its uniform synthetic input.
    The full model in the same sitting improved +8.0 dB median requant SNR
    over about 6.2 k real expert calls, and PPL dropped −12.7 %. The two
    disagree in sign. Accuracy verdicts come from the full-model columns
    (`NNTR_PPL`, `[L2-DIFF-MOE]` over a run, text vs A). A gtest prints
    its SNR as a field and asserts only the bit-identity to its scalar
    reference (gate (a)).

Learned in this project (#105, unit `R3CY205ZMND`, 2026-09-22):

26. **At m = 1 the HVX GEMV over the arena is bound by DDR read latency,
    not by `vrmpy` issue. Cutting compute there makes it slower.** The plan
    expected that dropping `gemm_rows4`'s three dead accumulators would
    save 50–100 µs. On the device the one-row loop (B1) raised the M==1
    `mm` from 973.0 to 1044.6 µs (+7.4 %) and cut decode by 4–8 %. Under
    the same loop, the L2-hot microbench cell ran 2.26× faster (70.77 →
    31.25 ns/tile). With 2 `vrmpy` per quarter-tile instead of 8, fewer
    loads are in flight to cover the DDR latency. The `l2fetch` lead wins
    it back monotonically (0 / 64 / 192 KB: 145.2 / 137.5 / 120.2 ns/tile
    arena) but still does not beat the four-row loop's 111.7 at 192 KB.
    A compute-side change to the M=1 GEMV is read on the `hot` cell, or
    after the feed is in VTCM. On the arena it shows the latency, not the
    compute.
27. **ION arena and cached heap read alike from HVX (within 11 %, no
    consistent sign), 21–27 GB/s.** The ION mapping is not why the M=1
    GEMV feed is slow. That is the plain direct-vector DDR band `hvx_impl`
    #59 measured on a sibling SoC (21–25 GB/s vs 37 through the ring). Only
    a DMA feed into VTCM (#100 → ㉒) gets past it. Also from this sitting,
    confirming rule 20: two level-2 profile passes in opposite order, from
    30 °C and 50–58 °C starts, agreed ≤ 0.4 % on every `mm`. A's `mm`
    973.0 reproduces #94 s2 C's 974.7 on the same DSP path, while decode
    tok/s differs by +47 % (PR #103's transport). The `mm` / `dsp` columns
    carry the kernel; E2E prefill tok/s carries the position in the
    block (A's own pairs −10..−24 %).

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
| ② Transport floor (wall 3) | **marshalling**: 527.7 µs/call at level 3 vs 587.7 at level 2 (−10 %, inside ±15 %; wake/clock would have been ≤ 300), `qos_mode` 2 both. **Refined by the #88 plan (host inventory, not yet measured):** cache maintenance on the over-sized staging pair (decode's 8 KiB rode a 4 MiB cached ION pair after prompt 512; 36–59 µs/MB, doc 50 §3.6) plus the 100 µs poll-window fallback to the interrupt path; the per-call arguments proper are 464 non-ION bytes (48 B primitive block + 5 sequences) and cannot carry 500 µs. Fix taken first = upstream `fb0f02b9` + `04a2fcc4` + `b0a384d6` (PR #103); prebind is conditional on that measurement (plan 88 §3.3 / §4 step 5); dspqueue / resident worker deferred to one paragraph (#83 §4). **Confirmed and closed by #88's sitting: staging 245.8 µs + poll 67.6 µs of A's 401.3; see ⑦ below** | #77 B; plan 88 §2.1; #88 |
| ③ Arena DMA probe (wall 2) | **shape** is the only axis: strided 2D 108–117 GB/s vs contiguous 72–80; workers flat-to-negative; vote ≤ 4 %. **But isolated rates are 4–7× the in-situ 16–18 GB/s**, so the wall is the layer call's use of the ring, not the engine (rule 11) — ⑥ rewritten, #87 | #77 C |
| ④ Two-reader DDR | **inconclusive on the DSP side** (probe defect, rule 12, #90); CPU side clean: 67.9 → 39.7 GB/s under DSP contention (−41 %). Q11 stays open | #77 ④ |
| Control on `R3CY205ZMND` (12 cells, prompt 512) | NPU decode 18.2–21.3 tok/s, CPU 46.4–54.1, NPU prefill 403–541: the NPU is **2.7× short** of 50 at G=512; the CPU clears 50 at G=64/512 and not at G=1024. Provisional for our unit until #91 | #77 A |
| **⑯ M=1 HVX GEMV (PR #86, `NNTR_MOE_HTP_M1_GEMV=1`) — variant C of #94 sitting 2, same binary and sitting as A** | **Goal progress, accuracy gate passed.** Decode +6.2 % (G=64, 18.83 vs 17.72), +7.7 % (G=512, 18.33 vs 17.03), +1.7 % (G=1024, no change); text byte-identical to A at every G and run; gtest `MoeLayerM1GemvMatchesHmx` bit-identical (0 bad of 2048 / 8192). Profile M==1: dsp **1414 → 1044 µs/call (−26 %)**, `DMA ring:` 46 → 2 descriptors, wait 269 → 4.3 µs, gather 146.5 → 0 — the DMA wait is gone from the path — **but `mm` 783 → 975 µs (+24 %)**: the GEMV reads the arena directly at ≈ 22.6 GB/s effective (22 MB in 975 µs), slower than the HMX loop's staged DMA + compute. Transport 649 → 748 µs/call (+100 µs, unexplained; `moe_set_opts` is once per session, so it is per-call cost of the switched path — #88 reads it). Prefill untouched (M>1 dsp 0.13 % apart). **Next lever inside ⑯: the GEMV's weight feed** (prefetch the expert tiles into VTCM ahead of the `vrmpy` loop instead of the direct arena read) — it must not re-import row h's slow list, so it waits for #100 (open item ㉒). Provenance: device skel `d6568c8b…` from `08afbb10` on the user's workstation, accepted under rule 21 | #94 s2 §C, `dad0f476` |
| **⑦ Wall 3, transport (PR #103: upstream size-class ION staging `fb0f02b9` + 5 ms poll `04a2fcc4`/`b0a384d6` + `staging:` line) — #88, one sitting A/B/C, `R3CY205ZMND`** | **Closed; the largest single-issue gain so far.** M==1 transport **401.3 → 87.9 µs/call (−78.1 %)** at level 2 (82.3 at level 3), `qos_mode=2` in all four profiles, gate ≤ 0.1 ms met → plan 88 §3.3 prebind **not built**. C (B + `NNTR_HTP_POLL_US=100`) 155.5 → the 313.4 µs cut = **staging 245.8 (78 %) + poll 67.6 (22 %)**, the plan's order and size. `staging:` M==1 `act 65536 out 65536 ion=y rpc allocs=19 non-ION 6/464 B`, `allocs` flat at level 3 (no per-call allocation). Decode 18.72 / 16.83 / 16.46 → **24.50 / 23.80 / 23.52 tok/s (+30.9 / +41.4 / +42.9 %)**, C at G=64 21.82 (+16.6 %); text bit-identical A = B = C at every G, run 1 = run 2; prefill −0.9 % (six-cell means 486.4 → 481.9), M>1 transport 2494.6 → 1770.3 with M>1 `dsp=` −0.07 %; peak RSS +0.3 % (both size classes stay alive). Thermal: B ran hotter (48.8 → 56.6 °C) and still won every cell; A re-run after B 17.82 (inside A's own spread). Only ≈ 6.9 of the 12.6 ms/token gain is in the transport column; ≈ 5 ms landed outside the MoE call (rule 23). Provenance: A/B/skel rebuilt on the user's workstation from `08afbb10` / `f3e99176` — accepted under rule 22 | `88-moe-call-marshalling.md` @ `3f9fa38d` |
| **Per-token budget at `htp_moe` head (cycle 7 estimate, not a measurement: TPS token time minus 22 × the level-2 M==1 `host` of the same sitting's G=64 profile)** | #88 B, G=512: token **42.0 ms** = MoE dsp 22 × 1.383 = **30.4 (72 %)** + transport 22 × 0.088 = **1.9 (5 %)** + outside the MoE call **≈ 9.6 (23 %)** (8.4 at G=64: FC, conv, attention, norms, lm_head, sampling). Floors at 38 GB/s: MoE 484 MB/token (22 × 22.02 MB) → **12.7 ms**, the rest 246 MB (lm_head Q4_0 alone 147 MB) → **≈ 6.5 ms** — so the MoE dsp column is **2.4× its floor (17.7 ms of excess)**, the outside-the-call time **≈ 1.5× (≈ 3 ms of excess)**, transport 1.9. **Biggest lever = MoE dsp (walls 1–2).** Ladder (projections): + #101 (GEMV default, #94 C: dsp −370 µs/call) ≈ −8 ms → ≈ 34 ms (≈ 29 tok/s, if C's +100 µs transport does not return under the new staging); + #105's gate (`mm` 975 → 600) ≈ −8 ms → ≈ 26 ms (≈ 39 tok/s); MoE at its floor → ≈ 24 ms (≈ 41 tok/s). 50 tok/s = 20 ms also needs transport + outside-the-call ≤ ≈ 7 ms → ⑨ (#85) and ⑧. ⑰ in TPS terms is ≈ 2–3 ms, not the profile's 18 (the `--profile` build adds ≈ 20 ms/token to both paths, #77 76.2 vs 54.4 and 39.3 vs 19.0), so it is not the biggest lever and gets no issue of its own | #88 B / A / C profiles + TPS; #94 s2 A / C; #77 profile |
| **Plan 87 §3.2 attribution a–h (wall 2): in-situ B levels 2/3 + `MoeChunkReplay` + `DMA_PROBE`, one sitting** | **a refuted** (wait 10.2 / 19.0 % of dsp, depth 9–11), **b refuted** (pushed shapes = probe iii), **c refuted, provisional #99** (workers 1/2/4 → 40.2 / 39.1 / 36.5 GB/s), **d sensitivity, not cause** (DDR competitor +1620 % wait, HVX competitor 0), **e refuted, sign reversed** (expert 0 fastest), **f confirmed 9.7 / 10.3 %** (the single blocked `site=act` wait = gather), **g real 11.0 / 7.1 %, unexplained** (`fresh` and `gap_us` controls both null), **h new, ≈ 60–70 % of the gap, provisional #99** (rule 19). **Wall-2 fix = row h → #100**; f (hide the first expert's 3.5 MiB behind gather / the previous layer) and g (cold-call penalty) are the remaining ~20 % and are filed after h. Plan 87 §0 expected f + g large: they are real but small; h was not foreseen | #94 s2 attribution table |
| **⑱ Hadamard rotation on the MoE down_proj input (`QS4CX_WH_HAD`, FWHT-256 before the u8 requant), #95. One sitting A/B/C/D on the side tree `htp_hadamard` (upstream `3006d255` + tooling: HMX block loop only, no M=1 GEMV, no DMA trace, 5 ms poll), `R3CY205ZMND`. Off-tree, so the numbers are not in BENCHMARK.md's `htp_moe` rows** | **Carry it over. Accuracy win, cost within noise.** **PPL (`NNTR_PPL`, 511 prompt tokens):** D CPU `q40` **109.759**, A **115.095**, B **115.095** (= A to the last digit), C **100.497**. C is −12.7 % vs A and −8.4 % vs the CPU, the first NPU variant below the CPU. **Requant SNR** (`[L2-DIFF-MOE]`, min / p10 / median dB, about 6.2 k calls): B 25.01 / 31.27 / 34.68 → C **39.68 / 41.67 / 42.67** (+14.7 / +10.4 / +8.0). `had=0` on all 6246 B lines, `had=1` on all 6274 C lines. **Gates:** B text byte-identical to A at G 64 / 512 / 1024 (after stripping the `[HTP-MOE] opts` line). C diverges from A at generated word 52, and both NPU variants diverge from D at word 43 at every G (a property of the weights). **Cost** (`[HTP-PROFILE]` level 2, G=64, mean µs/call): `requant` M>1 1097.3 / 1121.2 / **1567.4** (A / B / C, +446 µs), M==1 41.9 / 42.0 / **46.6**. Whole-call `dsp=`: M>1 15915.6 / 16061.7 / **16491.0** (C vs B **+2.7 %**), M==1 1333.5 / 1334.3 / **1338.5** (**+0.3 %**). **Decode tok/s:** A 25.08 / 24.40 / 23.83, B 24.82 / 24.12 / 23.75, C 24.49 / 23.65 / 23.53. A's own G=64 cell drifted −2.6 % from the sitting's start to its end (29 → 58 °C), so every B and C delta is inside that drift. C vs A's end re-run is +0.3 %. **Prefill tok/s cannot be read** (A 466.8 → B 433.8 → C 403.2, but A's re-run at the end read 404.7: order and heat, not code). The M>1 `dsp=` +2.7 % is the prefill reading, inside the −5 % gate. **Gtests:** `RejectsPartialBlock` and `MoeSetOptsEchoesKnownBits` OK. `MatchesScalarBitExact` failed only in the subnormal row (rule 24). `MoeLayerHadamardMatchesTwoCallReference` was bit-exact but failed its SNR assertion (rule 25). **Provenance:** everything was rebuilt on the user's workstation (Deviation 1). Device md5s match the rebuilt set. `git diff 63167235 6c89a912` is exactly the #95 commits. The model `a2829cd9…` equals the predicted hash and differs from `q40-qs4cx-wh` only in the 704 `down` tensors. The verdict is B vs C, one binary set switched by the model's dtype (rule 21). **Side reading on the accuracy gate:** A's PPL is +4.9 % over the CPU `q40`, and its text leaves the CPU's at word 43. That is the size of the "n/a (different weights)" that every NPU row in BENCHMARK.md carries, measured for the first time. Remainder (spec fix, gtest assertion, port incl. the M=1 GEMV requant site, `NNTR_PPL`): **#110** | `95-down-hadamard.md` @ `3a566761` (`origin/htp/95-down-hadamard`) |
| **㉒ compute half: M=1 GEMV one-row loop + `l2fetch` lead (PR #107 @ `a0ae8b29`), #105. One sitting A/B1/B2/B3, one app set, four skels, `R3CY205ZMND`, `NNTR_MOE_HTP_M1_GEMV=1` in every cell** | **Gate failed, accuracy passed. PR #107: hold, do not merge as-is.** M==1 `mm` (level 2, mean of two passes ≤ 0.4 % apart): A **973.0**, B1 1044.6 (+7.4 %), B2 1004.9 (+3.3 %, the branch default), B3 **930.9 (−4.3 %, inside ±5 %)**. Decode vs A 27.743 / 26.879 / 24.826: B1 −4.1 / −4.0 / −8.4 %, B2 −1.9 / −4.3 / −5.8 %, B3 **+2.6 / +2.2 / −1.7 %** (A's own G=1024 pair spreads 7 %). No variant reaches `mm` ≤ 600. **Accuracy:** `bit_identical yes`, `bad_elems` 0 under all four skels; text = A in 18/18 B cells. **Prefill:** M>1 `dsp` within 2.5 % of A in all eight profiles, `m1_gemv=0/23`. The E2E prefill column drifts with position (rule 27), so the gate is not failed. **Split (`MoeM1GemvFeedVsCompute`):** `arena − hot` is **≈ 74 % of `mm` under B3** (≈ 690 µs feed / ≈ 240 compute) and ≈ 37 % under A's four-row loop (≈ 590 compute). So the feed half is the lever. The one-row loop is what lets a VTCM feed hide the compute (240 ≪ 600 rather than 590 ≈ 600) (rule 26). **Merge verdict (cycle 9): hold.** As-is, the default lead is 64 KB = B2, a measured decode regression at all three G. Flipped to 192 KB (B3) it is neutral, not a gain, and the flip is a code change (no implementer this cycle). Once the weights stream from VTCM the `l2fetch` lead has nothing to prefetch, so the lead question dissolves if #100 certifies a feed shape. The PR's lasting content is the one-row loop, the x86 HVX-emulation check (`gemv_native_check.c`) and the microbench. The ㉒ feed issue takes them over by rebasing onto #107 and measuring feed + one-row together. If #100's feed verdict fails (no shape ≥ 37 GB/s beside HVX), the fallback is to land #107 with the lead at ≥ 192 KB after a sweep (384 KB + L2 budget). That fallback is filed only then. **Also read:** the GEMV path's M==1 transport is 182–184 µs here vs #88 B's 87.9 on the GEMV-off path. That is cross-sitting (rule 23), but it is the same +≈ 95 µs #94 C showed pre-#103 (649 → 748). #108's sitting reads it as an A/B (⑦ note). Provenance: rebuilt set accepted under rule 22 (BENCHMARK artifact row) | `105-m1-gemv-compute.md` @ `da41340b` (`origin/htp/105-m1-gemv-compute`) |
| **Per-token budget, cycle 9 (estimate: TPS token minus 22 × #105 A's level-2 M==1 `host` 1215.7 µs)** | #105 A (GEMV on, post-#103), G=512: token **37.2 ms** = MoE dsp 22 × 1.033 = **22.7 (61 %)**, of which `mm` 21.4 at ≈ 22.6 GB/s vs the 12.7 ms DDR floor, + transport 22 × 0.183 = **4.0 (11 %)** + outside the call **≈ 10.5 (28 %)** (9.3 at G=64). Ladder (projections): VTCM feed + one-row loop, `mm` 973 → ≈ 600 → ≈ −8.2 ms → ≈ 29 ms (≈ 34 tok/s). GEMV transport back to the GEMV-off 0.09 (if #108's A/B confirms it is path cost) → ≈ −2 ms. Then 50 tok/s = 20 ms needs outside-the-call ≤ ≈ 3.5 ms (MoE dsp ≈ 14.5 + transport ≈ 2) → ⑨ (#85), ⑧ | #105 A profile + TPS |
| Control, #94 sitting 2 (`R3CY205ZMND`, 13 A + 6 C cells, one binary set `2a75f7d9` + #98 skel) | NPU **17.72 / 17.03 / 17.30**, CPU **52.43 / 49.22 / 48.31**, NPU prefill 389–527: **the "now"** (BENCHMARK, contract §1). Distance to 50: 2.9× at G=512 (2.7× with C). Same-unit drift vs #77: CPU −9..+4 %, NPU −16..+5 % per cell with DSP profile columns ≤ 4 % apart (rule 20) | #94 s2 §A/§B |
| CPU control re-run on `R3CY205ZMND` (#94 first attempt, 6 cells, next day) | 53.0 / 53.5 · 51.1 / 51.2 · 49.7 / 48.4 tok/s vs #77's 54.1 / 52.9 · 52.7 / 52.0 · 46.4 / 48.5: **one unit drifts ≤ 7 % per cell day to day** (rule 9 quantified for the CPU path); G=1024 stays 46–50, i.e. the CPU does not reliably clear 50 there. No NPU cell (#97). Not the anchor; not a verdict on any lever | #94 `ec7ad296` |

## 3. Open items (candidates for issues; the supervisor promotes them)

| # | item | expected | depends on |
|---|---|---|---|
| ① | ~~Measurement A~~ **measured (#77) → §2.** Follow-on: the dense FC's 28.2 ms/token on the NPU path is the second-largest lever after MoE; **it is not a round-trip cost** (no FC is on the HTP in that config, §2 correction) — see ⑰ | — | ⑰ |
| ⑱ | **Measured by #95 on the side tree `htp_hadamard` → §2 (cycle 8): carry it over. PPL −12.7 % (100.5 vs A 115.1, CPU 109.8), requant SNR median +8.0 dB, cost +2.7 % / +0.3 % of the M>1 / M==1 `dsp=`, decode inside drift. #95 closed. Remainder filed as #110 (p2, `state:needs-plan`):** (a) `fwht_det.h` drops its FTZ (rule 24); (b) the synthetic SNR assertion becomes a printed field (rule 25); (c) port to `htp_moe` with the M=1 GEMV requant site (about `:509`), plus the HMX block loop (about `:1415`) and the tail (about `:370`); the HAD opts bit must not clear PR #108's default-on GEMV bit; (d) `NNTR_PPL` as a separate upstream cherry-pick (`32b46e32`, user decision at merge). Gate = an `htp_moe` handoff: B ≡ A (text + PPL), C PPL ≤ A, SNR median ≥ B + 5 dB, M==1 `dsp=` ≤ +2 %, prefill −5 %. p2 because it does not move decode tok/s; it becomes p1 if the user adopts PPL as the accuracy column for `QS4CX_WH*` models. History: **Accuracy, filed as #95 (p1):** Hadamard rotation on the MoE down_proj input — fold `Hᵀ·W_down` offline (new dtype `QS4CX_WH_HAD`, block 256, 1792 = 7 × 256, 1/16 both sides), FWHT-256 in IEEE `sf` add/sub on the DSP right before the existing u8 requant at all three requant sites of `hexkl_mm_u8i4_moe.c`. Targets the recorded gap (doc 43, 2026-09-09: 5 of 32 MoE calls at 67–80 dB SNR, each from exactly one of 1792 elements crossing a u8 level — boundary rounding, not row outliers; column-wise outliers never measured). **First accuracy item in this table**; it does not move tok/s by design and may show no effect — a "no effect" result closes it with the numbers. Gate = `NNTR_L2_DIFF` per-call SNR / `total_flips` on vs off on the 5 recorded calls + full-model handoff with prefill/decode TPS and text-identical-to-CPU; prefill gate applies (it touches the weight layout and the requant stage). Rules that bind: no qf32 (v75/v79), host scalar `fwht_rows_f32_ref` bit-identical to the HVX kernel, CPU `QS4CX` run stays the reference (no CPU kernel for `_HAD`, rule 6 applies to it too) | requant SNR on the 5 bad calls ↑; text-identical-to-CPU unchanged or better; TPS within noise | none; can run in parallel with the walls (touches quantizer + requant only) |
| ② | ~~Measurement B~~ **measured (#77) → marshalling → ⑦ / #88 → closed by #88's sitting 2026-09-22** | — | — |
| ③ | ~~Measurement C~~ **measured (#77) → shape only; in-situ gap is the wall → ⑥ / #87** | — | — |
| ④ | Two-reader DDR probe — **DSP side invalid in #77 (rule 12); refiled as #90** (stream > VTCM+L2 or use the DMA ring, bounded result). CPU side: −41 % under contention, so any split pays less than the sum | decides whether the CPU+NPU split (contract §3.2) can ever pay | #90, ride-along step in a later sitting |
| ⑤ | **Filed as #80.** M=1 MoE path on the existing HVX GEMV (wall 1). The kernel already exists: `hvx/hvx_gemm_u8i4_wh.c` (u8×i4 over WH tiles, int32 bit-identical to HMX, m ≤ 16) is used only by the prefill "tail" path in `hexkl_mm_u8i4_moe.c` (off by default, net −0.5 ms there). Decode needs a dispatch that sends all four experts through it at M=1 with no 64-row block, plus the weight feed (arena read vs DMA into VTCM) that ③ decides | MoE DSP 1.35 → ≈ 0.3 ms/call if DMA ≥ 30 GB/s | ③ |
| ⑥ | **Wall 2 — measured by #94 sitting 2 (§2 attribution): the 4–7× is ≈ 2.7× in the traced chunk list itself (row h, rule 19, provisional on #99) and ≈ 1.3–1.8× in the interleaving (f + g ≈ 20 % of `dsp_us`). Step 2 filed as #100 (row h: why the 46-descriptor / 32-region list replays at 40 GB/s where probe iii does 107; gate = replay ≥ 80 % of probe iii and in-situ `engine GB/s` ≥ 40; blocked by #99 until the replay's content check is explained). Rows f and g follow #100.** **Cycle 9: #100 re-scoped to plan 100 §0 (row h's cause + a DDR → VTCM feed shape certified ≥ 37 GB/s beside HVX for ㉒, same-run `c_star` denominator), raised to p0 after #105, no longer blocked by #99 (new cells set their dst mode explicitly).** History: rewritten by #77 C — not descriptor / engine / vote but "why does the MoE call see a quarter of the isolated rate". Step 1 = ~~#87~~ **landed on `htp_moe` as `b6ebc2b7` (PR #93, 2026-09-22; #87 closed)**: `hmx/hexkl_dma_trace.{c,h}` (static tables, union-of-intervals busy / depth / blocked-wait arithmetic, host-checked by `dma_trace_host_check`), trace hooks in `hexkl_mm_u8i4_moe.c` behind `hexkl_probe_on` (byte-identical output on vs off, 19 descriptors traced at the fixture shape, 46 at the LFM2 M=1 shape), IDL entries `dma_probe` / `moe_dma_trace_read` / `dma_replay` (`test/htp/nntr_hvx_dma_probe.c`), the header-only in-situ descriptor plan `test/htp/nntr_moe_dma_plan.h`, the second `weight DMA:` line and the `[HTP-DMA]` per-descriptor dump in `[HTP-PROFILE]` (`htp_compute_ops.cpp`, first `NNTR_HTP_DMA_TRACE` calls, default 3), and `TEST_F(HvxDmaProbe, MoeChunkReplay)` (workers 1/2/4 × HVX load × fresh/gap) in `unittest_hvx_dma_probe`. **The device sitting that fills the attribution table (plan 87 §1, hypotheses a–g) is #94 (sitting 2); its first attempt never reached the NPU because the trace's own `hexkl_dma_trace.c` was missing from the skel build (#97, rule 17).** Original scope for reference: instrument the ring use inside `hexkl_mm_u8i4_moe.c` (per-descriptor issue/complete pcycles, wait time in `hexkl_dma_ring_wait`, outstanding depth, actual chunk shapes at M=1 and M>1) + a device gtest that reproduces the in-situ pattern, gate = a table attributing the 4–7× to named causes. Step 2 = the fix that table names (separate issue). Interacts with #80/#86: the M=1 GEMV path reads the arena directly, so its A/B also tells what the ring costs | 16–18 → ≥ 40 GB/s in situ; 1.19 → ≤ 0.57 ms per call | #87 |
| ⑦ | **CLOSED 2026-09-22 by #88's sitting** (`88-moe-call-marshalling.md` @ `3f9fa38d`, `R3CY205ZMND`, one sitting A/B/C; §2 row): PR #103's size-class ION staging + 5 ms poll take the M==1 transport **401.3 → 87.9 µs/call (−78.1 %)**, past the ≤ 0.1 ms gate, so plan §3.3's prebind IDL pair is **not built**; the cut splits **staging 245.8 µs / poll 67.6 µs** (C = B + `NNTR_HTP_POLL_US=100` reads 155.5). Decode **+30.9 / +41.4 / +42.9 %** at G 64 / 512 / 1024, prefill −0.9 % (gate ≥ −5 %), M>1 transport 2494.6 → 1770.3 with `dsp=` unmoved, text bit-identical at all three G. Wall 3 is no longer the lever (1.9 ms/token left); the MoE call is 94 % DSP, so walls 1–2 own most of the remaining 2.1× to 50 tok/s. #88 and #83 (the transport-options document, consumed by plan 88) closed in cycle 7. **Left open, not an issue:** whether the GEMV path's +100 µs transport of #94 C (649 → 748) survives the new staging — #101's ride-along reads it as the same-sitting A (GEMV default) − A0 (`NNTR_MOE_HTP_M1_GEMV=0`) level-2 M==1 transport (`101-ride-along.md` R2/R4, rule 23). **Cycle 9:** #105's A (GEMV on, post-#103) reads M==1 transport 182–184 µs against #88 B's 87.9 (GEMV off), which is cross-sitting (rule 23) and consistent with it surviving. At ≈ 2 ms/token it is worth an A/B line in #108's sitting. dspqueue / resident worker: one paragraph (plan 83 §4), revisited only with ⑨. History: resolved by #77 B to prebound handles + per-call buffer/marshalling cleanup on plain FastRPC — #88; #83 narrowed to the document that says what "prebound" concretely means | 0.53 → ≤ 0.1 ms per call — **met: 0.088** | — |
| ⑧ | lm_head blocked Q4_0 twin on device (doc 46 §46): confirm 25.7 → ≈ 3.4 ms | ARM remainder | ① |
| ⑨ | One FastRPC call per token: M=1 RMSNorm, conv1d + gating, RoPE, attention, dense FFN, lm_head on the DSP; per-token entry in the IDL. **Filed as #85 (skeleton entry + op table), #81 (m=1 attention), #82 (RMSNorm, q/k norm, RoPE, conv1d + gating)**; host harness for all of them #84 | removes 22 round trips (**1.9 ms/token since #88**, was 12.5) and the time outside the MoE call (≈ 9.6 ms/token at G=512, floor ≈ 6.5); needed for the last ≈ 20 % to 50 tok/s once the MoE dsp column nears its floor (§2 budget row) | ⑤ ⑥ ⑦ |
| ⑩ | Registration at load time, bake cache on disk (doc 45 Phase D "P4") | load time, not speed | — |
| ⑪ | Prefill residency (doc 45 B/C/D) | prefill 523 → 700+ | after decode goal |
| ⑫ | CPU+NPU expert split | raises the ceiling only if ④ > 45 GB/s | ④, user decision Q11 |
| ⑬ | (withdrawn 2026-09-21: no simulator in this project, user decision) | — | — |
| ⑭ | `generation(last 64)` in the base report block (**#89**) | fills the contract §1.1 column from the next handoff on | — |
| ⑮ | ~~Anchor sitting on a named unit~~ **withdrawn (user decision 2026-09-22, rule 13): any unit, same-sitting A/B only.** #94 sitting 2 (`dad0f476`, `R3CY205ZMND`) is the "now" in BENCHMARK.md and contract §1; no second-unit issue is opened | — | — |
| ⑯ | **Default since PR #108 (#101):** env unset = GEMV on, `NNTR_MOE_HTP_M1_GEMV=0` = opt-out; proof line `[HTP] moe m1 gemv: on (applied=0x1) source=default`. From that PR on, an A log with `off` or without `source=default` is the HMX path and is not the reference (the sitting-2 rule below is inverted). **Measured → §2 (goal progress +6.2 / +7.7 / +1.7 %, text identical).** Follow-ups: make the GEMV path the default (**#101**, PR #108 merged, so every later variant A carries it); its compute side (**#105**, measured cycle 9: gate failed, PR #107 held, §2) and weight feed (㉒ remainder, via #100). History: Device A/B of #80 / PR #86 (M=1 GEMV switch on vs off): the first lever against `lfm2_moe` 42.1 ms/token. **Cycle 4: PR #86 merged as `2a75f7d9` (2026-09-22 02:00 UTC) before #94's set was pushed, so it rides as variant C (`NNTR_MOE_HTP_M1_GEMV=1`, 6 NPU cells + one level-2 run + `*MoeLayerM1GemvMatchesHmx*`) of sitting 2 — handoff rebuilt from `2a75f7d9` (`htp/94-sitting2-anchor-trace` @ `8029b76e`). A on that head = the same binary with the switch off (`[HTP] moe m1 gemv: off (applied=0x0)`, `blocks=5632 m1_gemv=0/1408`); a C log that prints `on` in an A cell voids the run.** | MoE DSP 1.35 → ≈ 0.3 ms/call if the arena read keeps up; if not, the number says what ⑥ must deliver | #97 (skel loads) → #94 (user's phone time); first attempt blocked before any NPU cell |
| ⑲ | ~~#97~~ **closed (PR #98 merged `14f65120`; confirmed on silicon by #94 s2's `DmaProbeShapes` PASSED as the first device command).** Skel from `2a75f7d9` did not load (`0x80000406`): `hexkl_dma_trace.c` absent from `test/htp/build.sh` `SRCS` → seven undefined `hexkl_dma_trace_*`. Fix = add the file + an undefined-symbol guard in `build.sh` (rule 17); then #94's artifact set is rebuilt and the handoff re-issued (same document, new md5s) | unblocks every NPU cell of #94 (⑥ ⑮ ⑯ ⑰) | none |
| ⑳ | **Filed as #99 (p1).** `MoeChunkReplay` fails its content check on all 11 cells (`res[6]=1467840` vs `want_sum=9461760`, `bytes_per_call` 22560768 vs the in-situ 22020096, `regions=32`) while `DmaProbeShapes` passes in the same binary. Decides whether the replay timings (rows c, d, g, h; rule 19) stand or must be re-measured; ~~#100 is blocked on it~~ (cycle 9: only #100's old 11 cells and `traced*` depend on it) | rows c/d/g/h confirmed or re-measured | none |
| ㉑ | **Fixed by PR #108 (print only, #102); row not yet seen on a device (#101 ride-along R2).** Cause: on the GEMV path the `swiglu` slot is Σ lane-time over the pool (`moe_tail_probe_add`, by design), and the print subtracted it as a stage; the 30-slot table is not misaligned. `rest` now leaves it out, and a `weight DMA: n/a (direct arena read inside mm …, N.NN lanes busy over mm)` line replaces the missing one. **Filed as #102 (p2).** Level-2 `[HTP-PROFILE]` M==1 row with the GEMV path on: `swiglu 5601.2`, `rest<=-5588.2 (-311.8% of host)`, no `weight DMA:` line — the GEMV path's stage slots are mis-attributed in the print (`htp_compute_ops.cpp` / the 30-slot stage table of #86). Cosmetic, no tok/s cell affected, but the B row of every sitting after #101 is unreadable until fixed | a readable M==1 row under `NNTR_MOE_HTP_M1_GEMV=1` | #101 makes it urgent |
| ㉒ | **Cycle 9: compute half measured (#105 → §2): gate failed. The best loop (one-row + 192 KB lead) gives `mm` 930.9 vs 973.0 (−4.3 %). The feed is ≈ 74 % of `mm` under that loop and ≈ 37 % under the four-row one. The feed half is the lever. It needs #107's loop to hide the compute: ≈ 240 µs vs the ≈ 590 the four-row loop leaves. PR #107 is held, and the feed issue rebases onto it. Order: #100 certifies a DDR → VTCM descriptor shape (≥ 37 GB/s beside HVX). Then the ㉒ feed issue is filed with that shape, gated `mm` ≤ 600 at M==1, text = A, prefill −5 %, and its A/B measures feed + one-row loop against A. If #100 certifies nothing, land #107 with a lead sweep (≥ 192 KB) and close ㉒ at what the arena allows.** Earlier: **Split (user direction, cycle 6, 2026-09-22). Compute side filed as #105 (p0, `state:needs-plan`):** one-row `gemm_rows1` at `m = 1` (drops `gemm_rows4`'s three dead accumulators; accumulation order unchanged, so int32 stays bit-identical to the HMX), `l2fetch` lead / distance, and the lane split over the pool. It needs no ring, so it does not wait for #100. The handoff sets `NNTR_MOE_HTP_M1_GEMV=1` explicitly in A and B, which keeps it independent of #101. **Feed side (remainder, not yet an issue):** VTCM staging of each expert's `wh_bytes` by DMA (two linear descriptors per expert); it must not re-import row h's slow list, so it is filed once #100 names h's cause. Evidence: C's `mm` 783 → 975 µs, a direct arena read at ≈ 22.6 GB/s effective, lane-bound (`swiglu / mm = 5.75`; plan 101 §3.4). Supervisor's estimate, not measured: the dead `vrmpy` are ~50–100 µs of the 975, and `hvx_impl` #59 measured direct DDR vector reads at 21–25 GB/s against 37 through the ring. So #105 alone may stop above 600; its ride-along microbench (arena vs VTCM copy × 4-acc vs 1-acc) sizes the feed half. Combined gate for ㉒ = `mm` ≤ 600 µs at M==1 with text identical | MoE dsp 1033 → ≈ 660 µs/call (`mm` 973 → ≈ 600; ≈ −8 ms/token) | feed: #100 → feed issue; compute: PR #107 (held) |
| ⑰ | **`fully_connected` 28.2 vs 10.3 ms/token on the NPU run with no FC on the HTP** (§2 ① correction). Not a round trip; candidates: CPU thread over-splitting / contention with the FastRPC poll thread (doc 48 §2 ③), CPU DVFS while the DSP runs. 18 ms/token is the second-largest single lever after MoE and needs no DSP code if it is a threading matter. Decide with one extra profile cell (`NNTR_NUM_THREADS=4` on the NPU model, or the non-WH `QS4CX` model with `NNTR_MOE_HTP_DECODE` off/on) — candidate ride-along for #94 or its own issue — **#94 s2 skipped the ride-along (budget)**, so this still needs its cell | **Resized in cycle 7 (§2 budget row):** in TPS terms the whole outside-the-MoE-call time is 11.1–13.7 ms/token on the pre-#103 binaries (#94 s2 A and C, #88 A) and **8.4 (G=64) / 9.6 (G=512) on `htp_moe` head** (#88 B — the 5 ms poll / staging moved ≈ 5 ms of it, rule 23), against a ≈ 6.5 ms byte floor; the profile's 28.2 vs 10.3 is mostly the `--profile` build's own ≈ 20 ms/token. So ⑰ is worth ≈ 2–3 ms/token now, not 18: no issue; still a cheap ride-along (`NNTR_NUM_THREADS=4` on the NPU model, TPS binary) in the next sitting that has room | ≈ −2..−3 ms/token on the NPU path | next sitting's ride-along |

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
* **Rebuilding a handoff set on a fresh checkout (#105 run notes, cycle 9;
  rule 22):** four gaps the recipes did not state, now in the
  `hexagon-gates` skill rung 3. (1) A new `git worktree` has **empty
  `subprojects/`**. `build_android.sh` then dies on `iniparser.h` after
  ≈ 15 min. Run `git submodule update --init --depth 1` (or copy the
  populated dirs) first. (2) `Applications/CausalLM/lib/libtokenizers_android_c.a`
  is per-checkout. Copy it or rebuild it with `build_tokenizer_android.sh`.
  (3) `libc++_shared.so` may not appear under `jni/obj/local/arm64-v8a/`.
  Take it from the NDK r30 sysroot
  (`toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/`,
  md5 `b1586b9b…`). (4) `HEXKL_ROOT` differs between workstations. Export
  it explicitly next to `HEXKL_SDK_VER=6.4.0.1`, and never rely on
  `env.sh`'s default path.

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
