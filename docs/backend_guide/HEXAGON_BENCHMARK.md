# Hexagon backend — benchmark and goals (Qwen3-0.6B, Snapdragon 8 Elite)

The single place where "where are we" and "where are we going" live. The
supervisor agent (`.claude/agents/hexagon-supervisor.md`) updates the
`nntrainer` rows from filled measurement handoffs
(`docs/measurements/`); HEXAGON.md §8 keeps the detailed tables. A snapshot
is published on the author's blog
(`dlwlzzero.github.io/_study/2026-09-16-WTD.md`).

Method for the `nntrainer` rows: `hexagon_e2e_test --chunk 128 --steps 64`,
decode median over steps 2–64, host wall time around each RPC
(HEXAGON.md §8.2). GENIEX rows are from Qualcomm AI Hub
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
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 512 | TODO | TODO | needs HexKL (stage 2) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 1024 | TODO | TODO | |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 4096 | TODO | TODO | |
| nntrainer | w4a8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 512 | TODO | TODO | after stage 2 (u8i4) |
| nntrainer | — | 8 Elite (Galaxy S25 Ultra) | CPU | 512 / 4096 | skipped | skipped | not a current goal |

## Goals

| Stage | Metric | Now | Goal | Gap | Levers (follow-up ledger numbers) |
|---|---|---|---|---|---|
| 1 HVX only | decode @512 | 31.9 (v75, 2026-09-17; v79 skel 31.4) | ≥ 70.3 | 2.2× | ⑫ host logits path (~40 ms/step), ① cross-op prefetch, ⑨ ⑩ chunk / `MM_TB` / `MM16_R` device sweep |
| 1 HVX only | decode @4096 | 9.6 (v75; v79 skel 11.4) | ≥ 27.5 | 2.9× (v79: 2.4×) | same + P5 attention K^T reuse (only cost that grows with context) |
| 1 HVX only | prefill @512 | 189.0 (v75; v79 skel 207.4) | 1000 (interim) | 5.3× (v79: 4.8×) | tiled kernel efficiency, P5 attention; 3,409 is out of HVX-only reach |
| 2 HVX + HMX | prefill @512 | — | ≥ 1000, then chase 3,409 | | HMX u8i8 matmul (specs/hexagon-hmx), needs HexKL |
| 2 HVX + HMX | decode @512 | — | ≥ 60 | | bandwidth bound ≈ 60 tok/s at 598 MB of int8 weights |

Decode and prefill do not move together: decode is weight-bandwidth plus
host-path bound, prefill is compute bound (HEXAGON.md §8.2). Reaching the
decode goal says nothing about prefill.

## Log

| Date | Change | Rows touched |
|---|---|---|
| 2026-09-16 | Table moved here from the blog draft; goals split into stage 1 / stage 2 (plan 0000) | all |
| 2026-09-17 | SDK 6.4 baseline from `docs/measurements/23-sdk64-baseline.md` (#23): v75 rows re-measured, no regression; v79-native skel rows added as ledger ④ data; goal "Now" moved to the v75 re-measurement | nntrainer HVX-only rows, stage 1 goals |
