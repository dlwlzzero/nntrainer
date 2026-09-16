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
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 512 | TODO | TODO | needs HexKL (stage 2) |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 1024 | TODO | TODO | |
| nntrainer | W8A8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 4096 | TODO | TODO | |
| nntrainer | w4a8 | 8 Elite (Galaxy S25 Ultra) | NPU, HVX + HMX | 512 | TODO | TODO | after stage 2 (u8i4) |
| nntrainer | — | 8 Elite (Galaxy S25 Ultra) | CPU | 512 / 4096 | skipped | skipped | not a current goal |

## Goals

| Stage | Metric | Now | Goal | Gap | Levers (follow-up ledger numbers) |
|---|---|---|---|---|---|
| 1 HVX only | decode @512 | 27.7 | ≥ 70.3 | 2.5× | ⑫ host logits path (~40 ms/step), ① cross-op prefetch, ⑨ ⑩ chunk / `MM_TB` / `MM16_R` device sweep |
| 1 HVX only | decode @4096 | 9.4 | ≥ 27.5 | 2.9× | same + P5 attention K^T reuse (only cost that grows with context) |
| 1 HVX only | prefill @512 | 192.1 | 1000 (interim) | 5.2× | tiled kernel efficiency, P5 attention; 3,409 is out of HVX-only reach |
| 2 HVX + HMX | prefill @512 | — | ≥ 1000, then chase 3,409 | | HMX u8i8 matmul (specs/hexagon-hmx), needs HexKL |
| 2 HVX + HMX | decode @512 | — | ≥ 60 | | bandwidth bound ≈ 60 tok/s at 598 MB of int8 weights |

Decode and prefill do not move together: decode is weight-bandwidth plus
host-path bound, prefill is compute bound (HEXAGON.md §8.2). Reaching the
decode goal says nothing about prefill.

## Log

| Date | Change | Rows touched |
|---|---|---|
| 2026-09-16 | Table moved here from the blog draft; goals split into stage 1 / stage 2 (plan 0000) | all |
