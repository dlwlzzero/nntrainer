# Hexagon backend — benchmark and goals (Qwen3-0.6B, Snapdragon 8 Elite)

The single place where "where are we" and "where are we going" live. The
supervisor agent (`.claude/agents/hexagon-supervisor.md`) updates the
`nntrainer` rows from filled measurement handoffs
(`docs/measurements/`); HEXAGON.md §8 keeps the detailed tables. A snapshot
is published on the author's blog
(`dlwlzzero.github.io/_study/2026-09-16-WTD.md`).

Method for the `nntrainer` rows: `hexagon_e2e_test --chunk 128 --steps 64`,
decode median over steps 2–64, host wall time around each RPC
(HEXAGON.md §8.2). Two S25 Ultra units have been used and their cDSP
decode clocks differ (1.91 vs 2.05 GHz, HEXAGON.md §7 rule 9): rows name
the unit, tok/s is compared only within a unit, and DSP Mcycles per step
is the cross-unit number. GENIEX rows are from Qualcomm AI Hub
(https://aihub.qualcomm.com/models/qwen3_0_6b).

## Comparison table

| Runtime | Precision | Device | Backend | Context | Prefill tok/s | Decode tok/s | Notes |
|---|---|---|---|---|---|---|---|
| GENIEX_LLAMACPP | q4_0 | 8 Elite QPN | NPU | 512 | 3,409 | 70.3 | |
| GENIEX_LLAMACPP | q4_0 | 8 Elite QPN | NPU | 4096 | 2,550 | 27.5 | |
| GENIEX_LLAMACPP | q4_0 | 8 Elite QPN | CPU | 512 | 945 | 89.4 | |
| GENIEX_LLAMACPP | q4_0 | 8 Elite QPN | CPU | 4096 | 237 | 18.9 | |
| GENIEX_QAIRT | w4a16 | 8 Elite (Galaxy S25) | NPU | 512 | 8,207 | 121 | |
| GENIEX_QAIRT | w4a16 | 8 Elite (Galaxy S25) | NPU | 1024 | 7,503 | 112 | |
| GENIEX_QAIRT | w4a16 | 8 Elite (Galaxy S25) | NPU | 4096 | 3,546 | 57.6 | |
| GENIEX_QAIRT | w4a16 | 8 Elite QPN | NPU | 4096 | 7,574 | 115 | |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX only | 512 | 192.1 | 27.7 | int8 per-channel weights, per-token int8 act; down_proj int16 act (M6 P4, 2026-09-16, SDK 6.0.0.2, v75 skel) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX only | 1024 | 118.2 | 22.9 | same |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX only | 4096 | 27.3 | 9.4 | same |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX only | 512 | 189.0 | 31.9 | same kernels, v75 skel rebuilt with SDK 6.4.0.1 / hexagon-clang 19.0.04 (#23, 2026-09-17, `docs/measurements/23-sdk64-baseline.md`); no regression vs P4; the 512 decode figure is ±5 % by construction (HEXAGON.md §8.2) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX only | 1024 | 117.1 | 23.2 | same |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX only | 4096 | 27.1 | 9.6 | same |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra, unit `R3CY205ZMND`) | NPU, HVX only, v79-native skel | 512 | 207.4 | 31.4 | `HEX_ARCH=v79` build of the same sources (#23, 2026-09-17); PPL / top-1 track the v75 skel at every context. Pre-#35 quantiser decode (the four ±1 LSB v79 simulator failures, fixed by #35); kept as the v79 reference on this unit |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra, unit `R3CY205ZMND`) | NPU, HVX only, v79-native skel | 1024 | 131.2 | 25.3 | same |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra, unit `R3CY205ZMND`) | NPU, HVX only, v79-native skel | 4096 | 34.4 | 11.4 | same |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra, unit `R3CY10WM83Y`) | NPU, HVX only | 512 | 185.5 | 29.4 | same kernels as the SDK 6.4 rows, v75 skel built from the #33 tree (md5 `cc0f1725…`; its DSP diff against #23's `81dcaab4…` is comments, clang-format and one per-call token-id check), host logits buffer in rpcmem + `FASTRPC_MAP_STATIC` (#24, 2026-09-18, `docs/measurements/24-host-logits.md`). This is the P3/P4 unit, whose cDSP holds 1.91 GHz in the decode loop where the #23 unit held 2.05 GHz: DSP cycles match #23 within 1–6 % (65.1 vs 64.3 Mcyc @512), so the tok/s delta is the unit's clock, not the change (HEXAGON.md §7 rule 9). malloc / rpcmem / static decode steps are within 70 µs of each other; `--eval` 33.0884 / 189 = #23 |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra, unit `R3CY10WM83Y`) | NPU, HVX only | 1024 | 114.5 | 23.0 | same (85.0 Mcyc vs #23 84.2) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra, unit `R3CY10WM83Y`) | NPU, HVX only | 4096 | 25.4 | 9.1 | same (225.9 Mcyc vs #23 213.0, +6 % after 168 s of sustained load, not repeated; the same +6 % recurs with the v79 skel in #35, issue #53) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra, unit `R3CY10WM83Y`) | NPU, HVX only, **v79 shipping skel** | 512 | 198.7 | 30.9 | `HEX_ARCH=v79` default build since #35 (integer quantiser decode, simulator 13/13 on both arches; `docs/measurements/35-v79-skel.md` B1, 2026-09-18, SDK 6.4.0.1 skel md5 `8d8cdb26…`). 61.2 Mcyc/step vs #23's 60.1 on the other unit (+1.8 %); prefill −4.2 %. `--eval` on the P4 prompt **32.4497 / 184** (x86 33.0195 / 184: −1.73 %, top-1 exact) — the v79 accuracy record, see HEXAGON.md §8.2; `t512_23` 41.4947 vs x86 41.5466 (−0.12 %). A `-DHTP_FORCE_QF_HELPERS` build (B2) reproduces every PPL / top-1 digit for digit and is within 0.5 % in speed (§7 rule 1 closed) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra, unit `R3CY10WM83Y`) | NPU, HVX only, **v79 shipping skel** | 1024 | 131.1 | 24.9 | same (77.8 Mcyc vs #23 76.5, +1.7 %; prefill −0.1 %); `--eval` 5.8595 / 692 (#23 v79 5.9509 / 692) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra, unit `R3CY10WM83Y`) | NPU, HVX only, **v79 shipping skel** | 4096 | 32.7 | 11.2 | same, cooled re-run (first run 32.1 / 10.8); 186.8 Mcyc (first run 188.0) vs #23 177.6 = +5.2 / +5.9 %, prefill −4.9 / −6.6 %: outside the ±5 % gate, but the v75 #24 row above misses #23 by the same +6 % on this unit with a DSP diff of comments and one token-id check, so the 4096 cell is read as a unit / session effect until issue #53 resolves it (HEXAGON.md §7 rule 9(e)). Still 17 % fewer cycles and +29 % prefill than v75 on the same unit and day (#24: 225.9 Mcyc / 25.4 tok/s); `--eval` 1.5623 / 3759 (#23 v79 1.5687 / 3758) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 512 | TODO | TODO | needs HexKL (stage 2) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 1024 | TODO | TODO | |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 4096 | TODO | TODO | |
| nntrainer | w4a8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX only | 512 / 4096 | TODO | TODO | 4-bit weight stream, issue #51 (HVX widens nibbles to int8 before vrmpy; u8i4 is the HMX format, stage 2) |
| nntrainer | — | 8 Elite (Galaxy S25 Ultra) | CPU | 512 / 4096 | skipped | skipped | not a current goal |

## Goals

| Stage | Metric | Now | Goal | Gap | Levers (follow-up ledger numbers) |
|---|---|---|---|---|---|
| 1 HVX only, W8A8 | decode @512 | 30.9 (v79 shipping skel, #35 2026-09-18, unit `R3CY10WM83Y` at 1.91 GHz, 61.2 Mcyc/step; the #23 unit `R3CY205ZMND` at 2.05 GHz read 31.4 on the pre-#35 v79 skel and 31.9 on v75; v75 on this unit 29.4, #24) | ≥ 60 (provisional: the W8 weight-stream bound, replaced by the ceiling the #25 handoff measures) | 1.9× | ① cross-op prefetch, ⑨ ⑩ chunk / `MM_TB` / `MM16_R` device sweep (#25); DSP clock during host idle gaps (#41: 1.17 GHz under `--eval` vs 1.91 in the generation loop, same unit). ⑫ host logits path is **closed** by #24: the 151,936-float return costs ≈ 2.5 ms in an empty call and < 0.1 ms inside a real 34 ms step, so it is not a lever. 70.3 needs 41.9 GB/s over the whole step with 596 MB of int8 weights (issue #49) |
| 1 HVX only, w4a8 | decode @512 / @4096 | — | ≥ 70.3 / ≥ 27.5 (the GENIEX_LLAMACPP q4_0 NPU rows) | | 4-bit weight stream (307 MB/step) on HVX, issue #51; after #25 measures the W8 ceiling |
| 1 HVX only | decode @4096 | 11.2 (v79 shipping skel, #35, `R3CY10WM83Y`, 186.8 Mcyc; #23 unit 11.4 / 177.6 Mcyc; v75 9.1–9.6) | ≥ 27.5 | 2.5× | same + P5 attention K^T reuse (only cost that grows with context); the 4096 cell carries a ±6 % cross-session spread until issue #53 settles it |
| 1 HVX only | prefill @512 | 198.7 (v79 shipping skel, #35, `R3CY10WM83Y`; #23 unit 207.4; v75 185.5–189.0) | 1000 (interim) | 5.0× | tiled kernel efficiency, P5 attention; 3,409 is out of HVX-only reach |
| 2 HVX + HMX | prefill @512 | — | ≥ 1000, then chase 3,409 | | HMX u8i8 matmul (specs/hexagon-hmx), needs HexKL |
| 2 HVX + HMX | decode @512 | — | ≥ 60 (W8) | | same weight-stream bound as stage 1; HMX does not change the bytes per step |

Decode and prefill do not move together: decode is weight-bandwidth plus
host-path bound, prefill is compute bound (HEXAGON.md §8.2). Reaching the
decode goal says nothing about prefill.

## Log

| Date | Change | Rows touched |
|---|---|---|
| 2026-09-16 | Table moved here from the blog draft; goals split into stage 1 / stage 2 (plan 0000) | all |
| 2026-09-18 | #24 host logits measurement (`docs/measurements/24-host-logits.md`): three rows on the P4 unit `R3CY10WM83Y` with the rpcmem + `FASTRPC_MAP_STATIC` logits return; ledger ⑫ closed as "not a lever"; unit clock caveat added to the method and the goal "Now" | nntrainer HVX-only rows, stage 1 W8A8 goal, method |
| 2026-09-18 | #35 v79 shipping-skel measurement (`docs/measurements/35-v79-skel.md`, unit `R3CY10WM83Y`): three v79 rows added, goal "Now" moved to them; #23's v79 rows re-labelled with their unit; the 4096 cell's +6 % cross-session spread filed as issue #53; HEXAGON.md §7 rule 1 closed (IEEE vs qf helpers identical on v79 silicon), rule 9 corrected and extended | nntrainer v79 rows, stage 1 goals, method |
| 2026-09-18 | Stage-1 W8A8 decode goal set to the weight-stream bound (≥ 60 provisional) after issue #49's arithmetic; 70.3 / 27.5 become the stage-1 w4a8 row (issue #51) | stage 1 goals, w4a8 row |
| 2026-09-17 | SDK 6.4 baseline from `docs/measurements/23-sdk64-baseline.md` (#23): v75 rows re-measured, no regression; v79-native skel rows added as ledger ④ data; goal "Now" moved to the v75 re-measurement | nntrainer HVX-only rows, stage 1 goals |
| 2026-09-17 | #35: quantiser decode made jam-independent, simulator 13/13 on both arches; user decision: v79 is the primary arch (S25 Ultra), v75 wrapped up. Goal "Now" moved to the v79 rows; device re-measurement of the fixed decode pending (`docs/measurements/35-v79-skel.md`, B1/B2, A optional) | v79 rows' note, stage 1 goals |
