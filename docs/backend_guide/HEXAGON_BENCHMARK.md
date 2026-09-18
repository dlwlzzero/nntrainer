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
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX only, v79-native skel | 512 | 207.4 | 31.4 | `HEX_ARCH=v79` build of the same sources (#23, 2026-09-17); PPL / top-1 track the v75 skel at every context; **not the shipping default** until ledger ④ decides (four v79 simulator tests still fail by ±1 LSB, HEXAGON.md §5.2 / §7 rule 1) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX only, v79-native skel | 1024 | 131.2 | 25.3 | same |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX only, v79-native skel | 4096 | 34.4 | 11.4 | same |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra, unit `R3CY10WM83Y`) | NPU, HVX only | 512 | 185.5 | 29.4 | same kernels and v75 skel (md5 `cc0f1725…`) as the SDK 6.4 rows, host logits buffer in rpcmem + `FASTRPC_MAP_STATIC` (#24, 2026-09-18, `docs/measurements/24-host-logits.md`). This is the P3/P4 unit, whose cDSP holds 1.91 GHz in the decode loop where the #23 unit held 2.05 GHz: DSP cycles match #23 within 1–6 % (65.1 vs 64.3 Mcyc @512), so the tok/s delta is the unit's clock, not the change (HEXAGON.md §7 rule 9). malloc / rpcmem / static decode steps are within 70 µs of each other; `--eval` 33.0884 / 189 = #23 |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra, unit `R3CY10WM83Y`) | NPU, HVX only | 1024 | 114.5 | 23.0 | same (85.0 Mcyc vs #23 84.2) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra, unit `R3CY10WM83Y`) | NPU, HVX only | 4096 | 25.4 | 9.1 | same (225.9 Mcyc vs #23 213.0, +6 % after 168 s of sustained load, not repeated) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 512 | TODO | TODO | needs HexKL (stage 2) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 1024 | TODO | TODO | |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 4096 | TODO | TODO | |
| nntrainer | w4a8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX only | 512 / 4096 | TODO | TODO | 4-bit weight stream, issue #51 (HVX widens nibbles to int8 before vrmpy; u8i4 is the HMX format, stage 2) |
| nntrainer | — | 8 Elite (Galaxy S25 Ultra) | CPU | 512 / 4096 | skipped | skipped | not a current goal |

## Goals

| Stage | Metric | Now | Goal | Gap | Levers (follow-up ledger numbers) |
|---|---|---|---|---|---|
| 1 HVX only, W8A8 | decode @512 | 31.9 (v75, 2026-09-17, unit `R3CY205ZMND` at 2.05 GHz; v79 skel 31.4; the same skel reads 29.4 on the 1.91 GHz P4 unit, #24 2026-09-18) | ≥ 60 (provisional: the W8 weight-stream bound, replaced by the ceiling the #25 handoff measures) | 1.9× | ① cross-op prefetch, ⑨ ⑩ chunk / `MM_TB` / `MM16_R` device sweep (#25); DSP clock during host idle gaps (#41: 1.17 GHz under `--eval` vs 1.91 in the generation loop, same unit). ⑫ host logits path is **closed** by #24: the 151,936-float return costs ≈ 2.5 ms in an empty call and < 0.1 ms inside a real 34 ms step, so it is not a lever. 70.3 needs 41.9 GB/s over the whole step with 596 MB of int8 weights (issue #49) |
| 1 HVX only, w4a8 | decode @512 / @4096 | — | ≥ 70.3 / ≥ 27.5 (the GENIEX_LLAMACPP q4_0 NPU rows) | | 4-bit weight stream (307 MB/step) on HVX, issue #51; after #25 measures the W8 ceiling |
| 1 HVX only | decode @4096 | 9.6 (v75; v79 skel 11.4) | ≥ 27.5 | 2.9× (v79: 2.4×) | same + P5 attention K^T reuse (only cost that grows with context) |
| 1 HVX only | prefill @512 | 189.0 (v75; v79 skel 207.4) | 1000 (interim) | 5.3× (v79: 4.8×) | tiled kernel efficiency, P5 attention; 3,409 is out of HVX-only reach |
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
| 2026-09-18 | Stage-1 W8A8 decode goal set to the weight-stream bound (≥ 60 provisional) after issue #49's arithmetic; 70.3 / 27.5 become the stage-1 w4a8 row (issue #51) | stage 1 goals, w4a8 row |
| 2026-09-17 | SDK 6.4 baseline from `docs/measurements/23-sdk64-baseline.md` (#23): v75 rows re-measured, no regression; v79-native skel rows added as ledger ④ data; goal "Now" moved to the v75 re-measurement | nntrainer HVX-only rows, stage 1 goals |
