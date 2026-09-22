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

Unit: **any Galaxy S25 Ultra** (user decision 2026-09-22, contract §12).
A handoff never names a serial; the filled handoff records the serial
actually used, every row here carries it, and numbers are compared only
as an A/B inside one sitting. Where two sittings happen to share a unit,
the same-unit drift is recorded as a note (LEDGER rule 13). Rows from the
PR author's device are labelled `(PR)`. Every row of this project so far
(#77, #94) ran on `R3CY205ZMND` (SM-S938N); #94 sitting 2 (2026-09-22,
`dad0f476`) is the **"now"** — the first sitting with all 12 A cells on
one binary set plus the first lever (variant C, LEDGER ⑯).

## Goals

| goal | now | target | ceiling | status |
|---|---|---|---|---|
| decode tok/s, NPU, gen 64 / 512 / 1024 | **17.72 / 17.03 / 17.30** (#94 sitting 2 A, means of two runs, `R3CY205ZMND`, `htp_moe` @ `2a75f7d9` + #98 skel); best lever so far: variant C (`NNTR_MOE_HTP_M1_GEMV=1`, same sitting) **18.83 / 18.33 / 17.59** = +6.2 / +7.7 / +1.7 %. Earlier: #77 19.7 / 21.3 · 18.4 / 18.2 · 19.0 / 17.0 (same unit, previous day); 20.8 (PR, gen 512, prompt 444) | **≥ 50** at each length | 48–52 (730 MB/token ÷ 34–38 GB/s) | distance **2.9× at G=512** (2.7× with C on). Per token (22 MoE calls): host 2062 µs/call → **45 ms of the 56 ms token** in A, 1792 → 39 of 53 ms in C (dsp 1414 → 1044, transport 649 → 748); the three walls stand — C removed the DMA wait from wall 2's path, but its `mm` grew 783 → 975 µs (LEDGER ⑯) |
| decode tok/s, CPU control (`q40`, 8 threads) | **52.43 / 49.22 / 48.31** (#94 sitting 2 A, means, `R3CY205ZMND`); earlier same unit: #77 54.1 / 52.9 · 52.7 / 52.0 · 46.4 / 48.5, #94 first attempt 53.0 / 53.5 · 51.1 / 51.2 · 49.7 / 48.4; 48 (PR) | — (the floor the NPU must beat) | same ceiling | the CPU sits on the ceiling; it clears 50 at G=64 in every sitting, at G=512 in two of three (47.3–52.7 over six runs), never reliably at G=1024 (46.4–49.7). Same-unit drift across three sittings ≤ 9 % per cell (rule 9) |
| prefill tok/s, NPU, prompt 512 | **389–527 (mean 470)** (#94 sitting 2 A, `R3CY205ZMND`); #77 403–541 (median 490); 523–532 (PR, prompt 444); CPU 268–340 (#94), 231–336 (#77) | **≥ −5 % of variant A** in every handoff | — | gate, not a goal; the denominator is variant A of each sitting, never this cell. One sitting's own A spread is ±13 % (389–527), so a single-cell −5 % is noise — read the M>1 `dsp=` of the profile row as the tie-breaker (#94 C: 16554 vs 16576 µs = unchanged) |
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
| 2026-09-22 | #94 partial, A run 1, unit `R3CY205ZMND` (**not** the planned `R3CY10WM83Y`; via a remote adb bridge) | CPU control, all Q4_0, 8 threads (`htp/94-sitting2-anchor-trace` @ `8029b76e`, code = `htp_moe` @ `2a75f7d9`; CPU path unchanged since #77) | `q40` | 64 | 287.2 | 53.02 | n/a (#89) | reference; run1 = run2 | `94-sitting2-anchor-trace.md` @ `ec7ad296` §A |
| 2026-09-22 | #94 partial, A run 2, `R3CY205ZMND` | same | `q40` | 64 | 274.2 | 53.47 | n/a | reference; run1 = run2 | same |
| 2026-09-22 | #94 partial, A run 1, `R3CY205ZMND` | same | `q40` | 512 | 278.7 | 51.08 | n/a | reference; run1 = run2 | same |
| 2026-09-22 | #94 partial, A run 2, `R3CY205ZMND` | same | `q40` | 512 | 280.2 | 51.15 | n/a | reference; run1 = run2 | same |
| 2026-09-22 | #94 partial, A run 1, `R3CY205ZMND` | same | `q40` | 1024 | 274.7 | 49.68 | n/a | reference; run1 = run2 | same |
| 2026-09-22 | #94 partial, A run 2, `R3CY205ZMND` | same | `q40` | 1024 | 202.7 | 48.40 | n/a | reference; run1 = run2 | same |
| 2026-09-22 | #94 partial, A NPU (6 cells), B, gtests, C | **not measured** — the skel built from `2a75f7d9` fails `remote_handle_open_domain` with `0x80000406` (`dlerror RX VA 0xFFF00000 outside ELF segment`); cause = seven undefined `hexkl_dma_trace_*` symbols, `hexkl_dma_trace.c` missing from `test/htp/build.sh` (#97) | `q40-qs4cx-wh` | — | — | — | — | — | same, Deviation 1 |
| 2026-09-22 | #94 sitting 2, A run 1, unit `R3CY205ZMND` (remote adb bridge; skel `d6568c8b…` built on the user's workstation from `htp_moe` @ `08afbb10`, DSP sources = `2a75f7d9`; ARM binaries = `2a75f7d9`) | CPU control, all Q4_0, 8 threads | `q40` | 64 | 339.7 | 51.57 | n/a (#89) | reference; run1 = run2 | `94-sitting2-anchor-trace.md` @ `dad0f476` §A |
| 2026-09-22 | #94 s2, A run 2, `R3CY205ZMND` | same | `q40` | 64 | 315.3 | 53.29 | n/a | reference; run1 = run2 | same |
| 2026-09-22 | #94 s2, A run 1, `R3CY205ZMND` | same | `q40` | 512 | 267.8 | 51.11 | n/a | reference; run1 = run2 | same |
| 2026-09-22 | #94 s2, A run 2, `R3CY205ZMND` | same | `q40` | 512 | 268.1 | 47.33 | n/a | reference; run1 = run2 | same |
| 2026-09-22 | #94 s2, A run 1, `R3CY205ZMND` | same | `q40` | 1024 | 312.8 | 48.44 | n/a | reference; run1 = run2 | same |
| 2026-09-22 | #94 s2, A run 2, `R3CY205ZMND` | same | `q40` | 1024 | 285.7 | 48.18 | n/a | reference; run1 = run2 | same |
| 2026-09-22 | #94 s2, A run 1, `R3CY205ZMND` | `htp_moe` @ `2a75f7d9` (PR #86 merged, switch **off**: `moe m1 gemv: off (applied=0x0)`, `blocks=5632 m1_gemv=0/1408`), MoE FFN on HTP, HMX block loop | `q40-qs4cx-wh` | 64 | 389.1 | 17.52 | n/a | n/a (different weights from `q40`); run1 = run2 | same |
| 2026-09-22 | #94 s2, A run 2, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 64 | 496.1 | 17.93 | n/a | run1 = run2 | same |
| 2026-09-22 | #94 s2, A run 1, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 512 | 469.3 | 17.09 | n/a | run1 = run2 | same |
| 2026-09-22 | #94 s2, A run 2, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 512 | 506.4 | 16.96 | n/a | run1 = run2 | same |
| 2026-09-22 | #94 s2, A run 1, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 1024 | 527.3 | 16.83 | n/a | run1 = run2 | same |
| 2026-09-22 | #94 s2, A run 2, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 1024 | 433.2 | 17.78 | n/a | run1 = run2 | same |
| 2026-09-22 | #94 s2, A run 3 (block-4 re-run after the two trace runs), `R3CY205ZMND` | same | `q40-qs4cx-wh` | 64 | 428.1 | 17.45 | n/a | = run 1; no `HTP-PROFILE` residue (plan 87 §1 gate) | same §A |
| 2026-09-22 | #94 s2, **C** run 1, `R3CY205ZMND` | same binary, `NNTR_MOE_HTP_M1_GEMV=1` (`on (applied=0x1)`, `blocks=0 m1_gemv=1408/1408`) — M=1 MoE on the HVX GEMV over arena weights, no DMA ring (LEDGER ⑯) | `q40-qs4cx-wh` | 64 | 426.0 | **19.19** (+8.3 % vs A mean) | n/a | **= A text** (same G, run); run1 = run2 | same §C |
| 2026-09-22 | #94 s2, C run 2, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 64 | 430.3 | **18.46** (+4.2 %) | n/a | = A text | same |
| 2026-09-22 | #94 s2, C run 1, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 512 | 433.5 | **18.74** (+10.0 %) | n/a | = A text | same |
| 2026-09-22 | #94 s2, C run 2, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 512 | 433.2 | **17.93** (+5.3 %) | n/a | = A text | same |
| 2026-09-22 | #94 s2, C run 1, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 1024 | 529.5 | 17.91 (+3.5 %) | n/a | = A text | same |
| 2026-09-22 | #94 s2, C run 2, `R3CY205ZMND` | same | `q40-qs4cx-wh` | 1024 | 504.4 | 17.27 (−0.2 %) | n/a | = A text | same |

#94 sitting 2 side tables (unit `R3CY205ZMND`, 2026-09-22 13:40–14:35 KST,
one binary set; verdicts in LEDGER §2, source `94-sitting2-anchor-trace.md`
@ `dad0f476`):

* **Same-unit drift, #94 s2 vs #77 (same unit, next day, different build with the same forward path), per cell, decode tok/s:** CPU G=64 51.6 / 53.3 vs 54.1 / 52.9 (−4.7 % / +0.7 %); G=512 51.1 / 47.3 vs 52.7 / 52.0 (−3.0 % / −8.9 %); G=1024 48.4 / 48.2 vs 46.4 / 48.5 (+4.3 % / −0.7 %). NPU G=64 17.5 / 17.9 vs 19.7 / 21.3 (−11.0 % / −15.7 %); G=512 17.1 / 17.0 vs 18.4 / 18.2 (−7.0 % / −7.0 %); G=1024 16.8 / 17.8 vs 19.0 / 17.0 (−11.6 % / +4.8 %). Prefill (NPU) 389–527 vs 403–541. The profile B row moved ≤ 4 % on every DSP column (dsp +3.9 %, mm +4.7 %), so the NPU's −7..−16 % in the tok/s cells is host/thermal (thermal_zone0 29 → 51 °C over block 2), not DSP code: rule 9 holds for the NPU path too, and cross-sitting NPU tok/s is not a verdict.
* **B — M==1 row (1408 calls, `qos_mode` 2), µs/call:** level 2 host 2062.5 / dsp **1414.0** / transport 648.6, gather 146.5, drain 113.1+8.9, mm 783.4, `blocks=5632 m1_gemv=0/1408`, weight DMA 21504 KB/call avg **15.6 GB/s**; level 3 host 1939.1 / dsp **1271.6** / transport 667.5, gather 124.8, drain 2.4+1.3, mm 781.9, avg **17.3 GB/s**. Within 4 % of #77 on every column. **First `DMA ring:` line ever (#87/#93):** level 2 `desc=46 waits=30 (blocked 5.1) wait=269.0 [act 145.4 gu 113.1+dn 8.9] busy=904..1198 → engine 18.4..24.3 GB/s depth max=11 first expert ready 165 last issue 1090 of 1414`; level 3 `wait=129.2 [act 123.7 gu 2.4+dn 1.3] busy=682..1005 → 21.9..32.3 GB/s depth max=9, first ready 151, last issue 989 of 1272`. M>1 row (23 calls): host 18888 / dsp 16576 / transport 2313, mm 9601, `blocks=1105`, 9.6 GB/s (level 2) — +4 % vs #77, prefill shape unmoved.
* **Plan 87 §3.2 attribution (share of `dsp_us`, L3 warm / L2 cold):** a ring-serialised-behind-compute **refuted** (wait 10.2 / 19.0 %, depth 9–11, only the first wait blocked); b slow probe shape **refuted** (pushed shapes = probe iii); c one-chain concurrency **refuted, provisional #99** (workers 1/2/4 unpaced: 40.2 / 39.1 / 36.5 GB/s); d DDR contention **sensitivity, not cause** (DDR competitor: wait 74 → 1273 µs; HVX/VTCM competitor: no change); e intra-call DVFS ramp **refuted, sign reversed** (expert 0 fastest); f un-hidden first-expert load **confirmed, 9.7 / 10.3 %** (`site=act` wait = gather to 1 µs); g cold-call penalty **real, 11.0 / 7.1 %, unexplained** by `fresh=1` or `gap_us=600`; **h (new): the traced 46-descriptor / 32-region list replays unpaced, 1 worker, no compute at 40.2 GB/s vs `DMA_PROBE` shape iii 106.9 GB/s on the same unit and skel — 2.7× lost before any interleaving, ≈ 60–70 % of the gap; provisional on #99.** Wall-2 nomination = h.
* **`MoeChunkReplay` (20 calls/cell, `plan_shape_ok=y`, traced call 1217 µs), `checksum_ok=n` on all 11 cells (#99), GB/s:** unpaced w=1/2/4: 40.2 / 39.1 / 36.5 (us/call 561 / 577 / 618); paced (in-situ schedule) w=1/2/4: 18.6 / 18.6 / 17.5 (1215 / 1214 / 1290 µs — reproduces the in-situ 15.6–17.3); paced + DDR load 16.1 (1404 µs, blocked 518/600); paced + VTCM/HVX load 18.6 (unchanged); `fresh=1` 18.6 (unchanged); `gap_us=600` 12.6 (1785 = 1215 + 570, the gap itself). `bytes_per_call=22560768` (+2.5 % vs the in-situ 22020096), `regions=32`.
* **`DMA_PROBE` (16 lines, `vote=1`, all `checksum_ok=y`), GB/s hot (block 5, ~51 °C) / cool (gate run, 29 °C), workers 1 → 4:** (i) 4 KiB × 256: 66.4 / 68.1 / 71.7 / 70.8 hot, 69.3 / 75.7 / 73.8 / 72.2 cool; (i1) 1 MiB: 66.3 / 68.0 / 71.9 / 71.0, 73.6 / 75.8 / 73.9 / 72.2; (ii) 16 KiB × 64 @ 56 KiB: 91.4 / 98.8 / 108.8 / 107.6, 110.2 / 113.3 / 111.4 / 109.0; (iii) 8 KiB × 64 @ 56 KiB: 88.8 / 95.1 / 104.8 / 105.3, **106.9** / 112.1 / 110.1 / 108.1. Cool = #77 within 4 %; hot is 3–19 % below cool (rule 18).
* **C proof run (level 2, G=64) vs A's B level 2, M==1:** host 2062.5 → 1792.2, dsp **1414.0 → 1044.0 (−26 %)**, transport 648.6 → 748.2, gather 146.5 → 0, mm **783.4 → 974.7 (+24 %)**, `DMA ring:` desc 46 → 2, wait 269.0 → 4.3 µs, depth 11 → 1; M>1 row dsp 16575.9 → 16554.3 (0.13 %, prefill untouched). Gtest `MoeLayerM1GemvMatchesHmx`: `bad_elems_M1 0 of 2048`, `bad_elems_M4 0 of 8192`, `bit_identical yes`, PASSED. Print defect in the C row: `swiglu 5601.2`, `rest<=-5588.2 (-311.8% of host)`, no `weight DMA:` line (cosmetic, LEDGER ㉑).

#77 side tables (unit `R3CY205ZMND`, same sitting; verdicts in LEDGER §2):

* **Per-type decode cost, ms/token, `--profile` build (never tok/s), (sum at G=512 − sum at G=64) / 448:** `lfm2_moe` NPU **42.12** vs CPU **22.32**; `fully_connected` **28.16** vs **10.29**; `output_of_causallm` 2.85 vs 3.40; `mha_core` 2.30 vs 2.81; `addition` 0.21 vs 0.14; `rms_norm` 0.19 vs 0.12; `custom_multiply` 0.15 vs 0.09; `causal_conv1d` 0.06 vs 0.03; `split` 0.05 vs 0.05; `reshaped_rms_norm` 0.06 vs 0.03; `swiglu` 0.04 vs 0.02; total 76.2 vs 39.3 (profile-inflated; TPS binary gives 54.4 vs 19.0).
* **Transport floor (`[HTP-PROFILE]`, M==1 row, 1408 calls, `qos_mode` 2):** level 2 host 1948.6 / dsp 1360.9 / transport **587.7** µs/call, weight DMA 16.2 GB/s; level 3 host 1750.4 / dsp 1222.7 / transport **527.7**, 18.0 GB/s. M>1 row (23 calls, prefill): transport 2231 / 2123 µs/call, DMA 10.0 / 10.1 GB/s.
  * #88 B (PR #103: size-class staging + 5 ms poll) / C (B with `NNTR_HTP_POLL_US=100`), same sitting as its own A: **to be filled from `88-moe-call-marshalling.md`** (gate: B ≤ 0.1 ms/call; the author's unit read decode call transport 553 → ≈ 161 (staging) and 158 → 83 µs (poll), docs 50 §3.7 / 51 §2.20).
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
| #94 set, `htp/94-sitting2-anchor-trace` @ `8029b76e` (code = `htp_moe` @ `2a75f7d9`): skel A, `tps/nntrainer_causallm`, `libcausallm_core.so`, `libnntrainer.so`, `libccapi-nntrainer.so`, `gtest/unittest_hvx_dma_probe`, `gtest/unittest_hvx_mm_u8i4` | `20fb9801…` / `dc3f4b8f…` / `68574a5c…` / `b74ae822…` / `b11b0ef1…` / `26db4bfe…` / `51ba46f5…` (staged `md5.txt`); **device** (`ec7ad296`, rebuilt from `8029b76e` on another machine, rule 14): skel `2bd15035…`, app `055703b2…`, `libnntrainer.so` `a508e889…`, `libcausallm_core.so` `8057cea3…`, `unittest_hvx_dma_probe` `76c43d3e…`, `unittest_hvx_mm_u8i4` `e79adbd0…`; **both skels carry the #97 defect** (seven undefined `hexkl_dma_trace_*`, `hexagon-nm -u -D`) and must not be pushed again | `94-sitting2-anchor-trace.md` Artifacts | `nntrainer_causallm`, `libcausallm_core.so`, `libccapi-nntrainer.so` are byte-identical to the discarded `0038cce2` build; #86 lives in the skel, `libnntrainer.so` and both gtests. `libc++_shared.so` `b1586b9b…`, `libsdkl.so` `0ad4e22a…`, `prompt512.txt` `fc65c158…` == #77 |

| #94 sitting 2 **device** set (`dad0f476`, the run of record): skel, `nntrainer_causallm`, `libnntrainer.so`, `libcausallm_core.so`, `libccapi-nntrainer.so`, `unittest_hvx_dma_probe`, `unittest_hvx_mm_u8i4` | skel **`d6568c8bb95e755485e982811e8db755`** (172,624 B) / `055703b2…` / `a508e889…` / `8057cea3…` / `b057acb6…` / `76c43d3e…` / `e79adbd0…`; `libc++_shared.so`, `libsdkl.so`, `prompt512.txt`, both `tokenizer.json` == table | skel: `test/htp/build.sh` @ `htp_moe` `08afbb10` on the user's workstation (`HEXKL_SDK_VER=6.4.0.1`, `UNDEFINED SYMBOLS OK (46 runtime imports)`; DSP sources identical to `2a75f7d9`, the #98 fix only); ARM files: the first attempt's local build of `8029b76e` (= `2a75f7d9`) | not the staged `ce7bb512…` / `dc3f4b8f…` set (rule 14, other build path); **accepted as the provenance of the sitting** (LEDGER rule 14, cycle 5 decision): A and C are one binary set, the switch is an env var, so the A/B is internally consistent |

**Sitting 2 (#94) artifact set, 2026-09-22 — measured (`dad0f476`); the
table rows above are the staged set, the device row is what ran.** Every
artifact in the first rows was built from `htp_moe` @ `2ce38d65` (or the
#77 branch). PR #93 (`b6ebc2b7`) changed `test/htp/nntr_hvx.idl` and PR #86
(`2a75f7d9`) added `moe_set_opts` and a 30th MoE stage slot, so the
reference set for sitting 2 was rebuilt from **`htp_moe` @ `2a75f7d9`**
(handoff `htp/94-sitting2-anchor-trace` @ `8029b76e`, worktree
`/home/j2z0-lee/nntrainer-94`, staged at `/local/mnt/workspace/htp_moe/94/`,
`md5.txt` there = the row below). Variant A of that sitting is this binary
with `NNTR_MOE_HTP_M1_GEMV` unset; variant C is the same binary with it set
(LEDGER ⑯). The two model `.bin`s are unchanged and keep the md5s above.
Pushing an old app with the new skel, or the reverse, fails with
`AEE_EBADPARM` (LEDGER rule 3).

## Log

| date | what | rows / goals touched |
|---|---|---|
| 2026-09-21 | File created from PR #4327 docs 49 and 50; goals from the contract | all |
| 2026-09-21 | Weights prepared (#78); first handoff written (#77, `docs/measurements/77-first-handoff.md` on `htp/77-first-handoff`) | artifacts |
| 2026-09-21 | #77 filled (`e8b930ad`) on unit **`R3CY205ZMND`**, not ours: 12 A rows added, unit-tagged readings next to the provisional "now" (not replaced; anchor sitting #91), side tables A-profile / B / C / ④, artifact md5 note; #77 closed | results, goals (now column annotated, status), artifacts |
| 2026-09-22 | Cycle 3: #87 closed (PR #93 merged `b6ebc2b7`), #91 folded into #94 (sitting 2 = anchor cells + #87 trace/replay, variant C only if PR #86 merges first); rebuild note for the #94 artifact set; PR #92 guide merged (LEDGER §3a); verdict ① corrected (no FC on the HTP in the NPU config → LEDGER ⑰) | goals (status column), artifacts |
| 2026-09-22 | Cycle 4: no filled handoff (#94 tables empty, still `state:needs-measurement`); PR #86 merged `2a75f7d9` → #94 rebuilt from it with variant C on (LEDGER ⑯); #94 artifact row added, rebuild note rewritten; #95 (Hadamard on down_proj input, accuracy) recorded as LEDGER ⑱; upstream head `0a0c0402` → `b0a384d6` (LEDGER cycle 4) | artifacts |
| 2026-09-22 | Cycle 5a: #94 first attempt (`ec7ad296`) folded in — 6 CPU cells on **`R3CY205ZMND`** again (same-unit re-run of #77, not the anchor; deviations: wrong unit, remote bridge, skel would not load), no NPU/B/gtest/C cell; the goal column is untouched; #97 filed (p0) for the loader failure = undefined `hexkl_dma_trace_*` in the skel (`hexkl_dma_trace.c` missing from `build.sh`), LEDGER rule 17; device md5s added to the #94 artifact row; upstream head unchanged at `b0a384d6` | results, goals (CPU status), artifacts |
| 2026-09-22 | Cycle 5b: #94 sitting 2 (`dad0f476`) folded in — 13 A + 6 C rows on `R3CY205ZMND`, "now" replaced (NPU 17.72 / 17.03 / 17.30, CPU 52.43 / 49.22 / 48.31, prefill 389–527; contract §1 updated); C = first measured lever (+6.2 / +7.7 / +1.7 %, text identical, LEDGER ⑯ verdict); B `DMA ring:` first reading, attribution a–h, replay and probe side tables; artifact device row; #94 and #97 closed; **unit requirement dropped** (user decision 2026-09-22): the Method paragraph no longer names a serial; upstream head `b0a384d6` → `80fa1a1a` (4 commits, one in `hexkl_mm_u8i4_moe.c`, LEDGER cycle 5b) | goals (all three "now" cells), results, artifacts, method |
