# Measurement 185: the decode MoE call's idle DMA time on the S26 Ultra (v81, 4 queues)

Branch `htp/185-m1-schedule-overlap`, stacked on `htp/177-m1-dma-queues`
(`926819a2`). Plan `docs/plans/185-m1-schedule-overlap.md` (on `htp_moe_v81`
at `31a59586`). Run by the implementer agent on `R5KL20NFRCK` (SM-S948N, S26
Ultra) under the contract's v81 adb exception (§12, 2026-09-29), 2026-09-30
03:55–05:26. Device time ≈ 90 min. The phone was idle at the start
(`pidof nntrainer_causallm unittest_hvx_dma_probe` empty; every run re-checks it).

## Why

At 4 DMA queues (#177, Q4) the decode MoE call is transfer-bound: ≈ 21.5 MB
per call at ≈ 62 GB/s is ≈ 346 µs of the 456 µs `dsp`. The plan moves work
between pool runs so the four queues sit idle less: C1 (D) sends an even
n's downs 2 and 3 on C(0) and C(1) instead of run B, C2 (Q) issues GU(0) as a
workers-only job while the caller scans and packs, and C3 (R) folds each
requant into a later run so run B is left only for an odd n. The bits must
not change. Goal: S26 decode above the S25's #158 B (51.82 / 50.61 / 47.36
tok/s at G 64 / 512 / 1024).

## Artifacts (SDK 6.4.0.1, HexKL 1.0-beta.2 `lib/6.4.0.1`, NDK r30)

All three variants run #177's app set (`/local/mnt/workspace/htp_moe/177/A`),
so only the skel differs. One full directory per variant under
`/data/local/tmp/nntrainer/causallm/s185/{A,DQ,DQR}`, each with its own
`md5.txt`. `md5sum -c md5.txt` ran on the phone before every run. `run()`
exits on a failure, and no log has `MD5 FAIL`.

| file | md5 | built from |
|---|---|---|
| skel A (#177 Q4t, reference) | `0200414e809fc7dfe851fbfb4dd60481` | `a1a43a28` (the #177 head kernel), from `177/Q4t` |
| skel DQ (C1 + C2) | `2f15d30b749a92192e2f488ed2232457` | `7d0cb151`, `HEX_ARCH=v81 ./test/htp/build.sh`: `Flags: 0x81`, `UNDEFINED SYMBOLS OK (51 runtime imports)` |
| skel DQR (C1 + C2 + C3) | `5541847c2cdaf8fd1d2cc53d0cf44030` | `8dea5f0a` (kept on `htp/185-dqr`), same build, `Flags: 0x81`, `UNDEFINED SYMBOLS OK (51)` |
| nntrainer_causallm | `0c0e0f78d86f202d91f3f86d67437406` | #177's set |
| libcausallm_core.so | `de1323e80bb62d6c4607cef9efa7847c` | #177's set |
| libnntrainer.so | `2657fdae10ca91a16e06950fedc0ddae` | #177's set |
| libccapi-nntrainer.so | `65c9034c6341384443166b89de66191d` | #177's set |
| libc++_shared.so | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 |
| libsdkl.so | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `armv8_android26` |
| model `nntr_lfm2_8b_a1b_q40_arm.bin` | `7b7867fab51845664c0050c0a837073e` | `models/q40-qs4cx-wh`, read only |

The PR's rung 3 rebuild of the app (`builddir` reconfigured with
`-Denable-profile=false`, `ninja install`, then
`build_android.sh --htp --cache`, no `--profile`) gave the same four md5s as
#177's set above, so the staged set is not a profile build. The skel md5s do
not reproduce across builds (measurement 177). A second v81 build of
`7d0cb151` gave `b90f2c5c…`. Its per-function disassembly (addresses
normalised, sorted by function) matches the staged DQ skel's (`6df04e79…` for
both), and DQR's differs (`92b4593f…`). The v79 skel builds too
(`HEX_ARCH=v79`: `Flags: 0x79`, `UNDEFINED SYMBOLS OK (51)`, `59f4ff85…` at
`7d0cb151`, `74a0f8ae…` at `8dea5f0a`). It was not run.

`variant.env` is `export NNTR_MOE_DMA_QUEUES=4` for all three. Every one of
the 84 logs has the same banner:
`applied=0x1f03e1 lead=192KB rows1=1 feed=vtcm dma_bypass=1 dma_q=4 source=default`.
The model directory `s185/model` holds copies of the JSON files and a
symlink to the `.bin`. Only its `nntr_config.json` has `num_to_generate`
rewritten. `NNTR_NUM_THREADS=8` and prompt 512 (`prompt512.txt`,
`fc65c158…`) are used throughout.

## Results: decode and prefill (G4, G5), with the S25 comparison

Mirrored order A DQ DQR DQR DQ A per round. Rounds: G=512 s1, s2; G=64 s3,
s4; G=1024 s5, s6; then the re-run rounds G=1024 s7 and G=512 s8 (see G5).
G=64 and G=1024 got two rounds rather than the plan's one, so each prefill
median has ≥ 3 runs. The G=8 sanity runs (below) absorbed the first run
after idle. Cool-down: the loop waits until zone0 ≤ 45.0 °C (max 5 min).
Every run's check read ≤ 44.8 °C after 0–90 s (`logs/cool.log`). The
`pre` snapshot, taken 1–2 s later after a `dumpsys battery`, reads 42.0–46.3 °C.
zone0 and battery are in °C. S25 reference: #158 B on `R3CY10WM83Y`, prefill
509.4 / 423.5 / 405.5 and decode 51.82 / 50.61 / 47.36 at G 64 / 512 / 1024.

| variant | G | round | prefill tok/s | decode tok/s (all) | decode (last 64) | vs S25 prefill / decode | zone0 pre → post | bat pre → post |
|---|---|---|---|---|---|---|---|---|
| A | 64 | s3r1 | 484.4 | 49.12 | 49.12 | -4.9 % / -5.2 % | 46 → 63 | 39.9 → 39.9 |
| DQ | 64 | s3r2 | 506.4 | 51.57 | 51.57 | -0.6 % / -0.5 % | 45 → 67 | 39.4 → 39.3 |
| DQR | 64 | s3r3 | 508.9 | 52.76 | 52.76 | -0.1 % / +1.8 % | 45 → 72 | 39.5 → 39.6 |
| DQR | 64 | s3r4 | 505.9 | 53.24 | 53.24 | -0.7 % / +2.7 % | 46 → 73 | 39.5 → 39.3 |
| DQ | 64 | s3r5 | 515.6 | 52.50 | 52.50 | +1.2 % / +1.3 % | 46 → 73 | 39.8 → 39.7 |
| A | 64 | s3r6 | 492.8 | 50.00 | 50.00 | -3.3 % / -3.5 % | 46 → 69 | 39.7 → 40.0 |
| A | 64 | s4r1 | 517.2 | 50.27 | 50.27 | +1.5 % / -3.0 % | 46 → 70 | 39.7 → 39.7 |
| DQ | 64 | s4r2 | 512.5 | 52.46 | 52.46 | +0.6 % / +1.2 % | 46 → 69 | 39.9 → 39.9 |
| DQR | 64 | s4r3 | 509.5 | 52.76 | 52.76 | +0.0 % / +1.8 % | 45 → 69 | 39.8 → 39.6 |
| DQR | 64 | s4r4 | 509.5 | 52.29 | 52.29 | +0.0 % / +0.9 % | 46 → 69 | 40.0 → 39.9 |
| DQ | 64 | s4r5 | 500.5 | 51.82 | 51.82 | -1.7 % / +0.0 % | 46 → 67 | 39.9 → 40.1 |
| A | 64 | s4r6 | 506.9 | 50.59 | 50.59 | -0.5 % / -2.4 % | 46 → 66 | 39.8 → 39.8 |
| A | 512 | s1r1 | 516.6 | 49.29 | 48.85 | +22.0 % / -2.6 % | 42 → 65 | 34.4 → 34.7 |
| DQ | 512 | s1r2 | 519.3 | 52.35 | 51.61 | +22.6 % / +3.4 % | 43 → 65 | 35.4 → 35.4 |
| DQR | 512 | s1r3 | 520.3 | 51.87 | 51.20 | +22.9 % / +2.5 % | 44 → 65 | 36.5 → 36.5 |
| DQR | 512 | s1r4 | 375.6 | 50.66 | 40.13 | -11.3 % / +0.1 % | 44 → 64 | 36.9 → 37.4 |
| DQ | 512 | s1r5 | 486.2 | 51.12 | 49.31 | +14.8 % / +1.0 % | 46 → 64 | 37.4 → 37.4 |
| A | 512 | s1r6 | 478.5 | 49.96 | 48.41 | +13.0 % / -1.3 % | 46 → 63 | 38.1 → 38.1 |
| A | 512 | s2r1 | 503.4 | 49.83 | 48.60 | +18.9 % / -1.5 % | 45 → 65 | 37.9 → 38.2 |
| DQ | 512 | s2r2 | 506.4 | 51.19 | 48.16 | +19.6 % / +1.1 % | 44 → 65 | 38.6 → 38.8 |
| DQR | 512 | s2r3 | 501.5 | 51.58 | 50.55 | +18.4 % / +1.9 % | 45 → 66 | 38.7 → 38.9 |
| DQR | 512 | s2r4 | 508.4 | 51.36 | 50.27 | +20.1 % / +1.5 % | 45 → 66 | 39.3 → 39.5 |
| DQ | 512 | s2r5 | 501.0 | 51.55 | 50.08 | +18.3 % / +1.8 % | 46 → 66 | 39.4 → 39.4 |
| A | 512 | s2r6 | 446.4 | 49.13 | 48.56 | +5.4 % / -2.9 % | 46 → 66 | 39.9 → 40.0 |
| A | 512 | s8r1 | 504.9 | 49.57 | 47.23 | +19.2 % / -2.1 % | 46 → 67 | 41.1 → 42.4 |
| DQ | 512 | s8r2 | 501.0 | 51.23 | 49.31 | +18.3 % / +1.2 % | 46 → 68 | 41.2 → 41.9 |
| DQR | 512 | s8r3 | 520.3 | 51.77 | 49.42 | +22.9 % / +2.3 % | 46 → 67 | 40.9 → 41.5 |
| DQR | 512 | s8r4 | 517.2 | 51.76 | 49.42 | +22.1 % / +2.3 % | 46 → 68 | 41.0 → 42.0 |
| DQ | 512 | s8r5 | 529.5 | 51.69 | 49.88 | +25.0 % / +2.1 % | 46 → 68 | 40.7 → 41.6 |
| A | 512 | s8r6 | 526.2 | 49.34 | 43.63 | +24.3 % / -2.5 % | 46 → 67 | 40.8 → 42.1 |
| A | 1024 | s5r1 | 519.8 | 47.85 | 44.72 | +28.2 % / +1.0 % | 46 → 66 | 40.0 → 41.7 |
| DQ | 1024 | s5r2 | 396.0 | 48.95 | 46.24 | -2.3 % / +3.4 % | 46 → 67 | 40.3 → 41.9 |
| DQR | 1024 | s5r3 | 523.0 | 48.79 | 45.52 | +29.0 % / +3.0 % | 46 → 67 | 40.8 → 42.6 |
| DQR | 1024 | s5r4 | 533.9 | 49.08 | 43.72 | +31.7 % / +3.6 % | 46 → 67 | 40.5 → 42.1 |
| DQ | 1024 | s5r5 | 500.0 | 48.49 | 45.65 | +23.3 % / +2.4 % | 45 → 67 | 40.7 → 42.6 |
| A | 1024 | s5r6 | 513.0 | 46.01 | 43.75 | +26.5 % / -2.8 % | 46 → 67 | 40.9 → 42.5 |
| A | 1024 | s6r1 | 518.2 | 47.44 | 43.93 | +27.8 % / +0.2 % | 46 → 66 | 41.0 → 42.8 |
| DQ | 1024 | s6r2 | 513.0 | 48.82 | 45.58 | +26.5 % / +3.1 % | 46 → 67 | 40.8 → 42.4 |
| DQR | 1024 | s6r3 | 386.7 | 48.57 | 46.08 | -4.6 % / +2.6 % | 46 → 67 | 41.4 → 43.0 |
| DQR | 1024 | s6r4 | 510.5 | 48.60 | 44.69 | +25.9 % / +2.6 % | 46 → 67 | 41.2 → 43.3 |
| DQ | 1024 | s6r5 | 515.1 | 48.24 | 44.44 | +27.0 % / +1.8 % | 46 → 67 | 41.0 → 42.8 |
| A | 1024 | s6r6 | 514.1 | 47.37 | 42.95 | +26.8 % / +0.0 % | 46 → 67 | 41.4 → 43.4 |
| A | 1024 | s7r1 | 525.7 | 47.53 | 44.17 | +29.6 % / +0.4 % | 46 → 67 | 41.1 → 43.3 |
| DQ | 1024 | s7r2 | 355.6 | 48.26 | 45.85 | -12.3 % / +1.9 % | 46 → 67 | 41.3 → 42.9 |
| DQR | 1024 | s7r3 | 511.0 | 49.24 | 46.01 | +26.0 % / +4.0 % | 46 → 67 | 41.3 → 43.3 |
| DQR | 1024 | s7r4 | 505.4 | 48.25 | 45.88 | +24.6 % / +1.9 % | 46 → 67 | 40.6 → 42.7 |
| DQ | 1024 | s7r5 | 504.4 | 48.87 | 44.79 | +24.4 % / +3.2 % | 45 → 68 | 41.0 → 43.2 |
| A | 1024 | s7r6 | 448.3 | 46.87 | 44.51 | +10.6 % / -1.0 % | 46 → 66 | 41.0 → 42.8 |

Summary (prefill: median over the runs, min..max; decode: mean, min..max):

| variant | G | n | prefill median (min..max) | prefill mean | decode mean (min..max) | decode vs A | decode vs S25 | prefill median vs A | prefill median vs S25 |
|---|---|---|---|---|---|---|---|---|---|
| A | 64 | 4 | 499.9 (484.4..517.2) | 500.3 | 50.00 (49.12..50.59) | — | -3.5 % | — | -1.9 % |
| DQ | 64 | 4 | 509.5 (500.5..515.6) | 508.8 | 52.09 (51.57..52.50) | +4.2 % | +0.5 % | +1.9 % | +0.0 % |
| DQR | 64 | 4 | 509.2 (505.9..509.5) | 508.4 | 52.76 (52.29..53.24) | +5.5 % | +1.8 % | +1.9 % | -0.0 % |
| A | 512 | 6 | 504.2 (446.4..526.2) | 496.0 | 49.52 (49.13..49.96) | — | -2.2 % | — | +19.1 % |
| DQ | 512 | 6 | 503.7 (486.2..529.5) | 507.2 | 51.52 (51.12..52.35) | +4.0 % | +1.8 % | -0.1 % | +18.9 % |
| DQR | 512 | 6 | 512.8 (375.6..520.3) | 490.6 | 51.50 (50.66..51.87) | +4.0 % | +1.8 % | +1.7 % | +21.1 % |
| A | 1024 | 6 | 516.1 (448.3..525.7) | 506.5 | 47.18 (46.01..47.85) | — | -0.4 % | — | +27.3 % |
| DQ | 1024 | 6 | 502.2 (355.6..515.1) | 464.0 | 48.61 (48.24..48.95) | +3.0 % | +2.6 % | -2.7 % | +23.9 % |
| DQR | 1024 | 6 | 510.7 (386.7..533.9) | 495.1 | 48.76 (48.25..49.24) | +3.3 % | +2.9 % | -1.0 % | +25.9 % |

* **G4 (the goal), DQ:** 52.09 / 51.52 / 48.61 tok/s, above the S25's 51.82 /
  50.61 / 47.36 (+0.5 / +1.8 / +2.6 %). DQ's G=512 mean 51.52 is above A's max
  49.96. At G=64 the margin is thin: DQ's four runs are 51.57 / 52.50 / 52.46
  / 51.82.
* **G4, DQR:** 52.76 / 51.50 / 48.76 (+1.8 / +1.8 / +3.0 % vs S25). Every
  G=64 run is above 51.82 (52.29..53.24). Its G=512 mean is above A's max.
* **Ship rule (plan §1):** ship DQR if it passes G4 *and* its G=512 mean
  ≥ DQ's. DQR's is 51.50 against DQ's 51.52 (−0.04 %), so the rule ships
  **DQ**, and commit 3 (C3) is dropped from the PR. DQR is ahead at G=64
  (+1.3 %) and G=1024 (+0.3 %) and has the lower `dsp` (−4.6 µs/call, below).
  C3 stays on `htp/185-dqr` for the user's call.
* **G5 (prefill, standing):** on the medians, every variant is within
  −2.7..+1.9 % of A at every G. On the means, DQ at G=1024 is −8.4 % (464.0
  against 506.5) from two outlier runs, 396.0 (s5r2) and 355.6 (s7r2). The
  re-run round s7 got one of them. The M>1 prefill path runs no new code in
  DQ: the GU(0) job and the scan's `NULL` pool sit only in the M ≤ 4 feed
  branch. Every variant has prefill outliers (A 446.4 / 448.3, DQR 375.6 /
  386.7), which matches the brief's "333–574 tok/s, same binary".
* **Prefill vs the S25 (information):** the medians are above the S25's at
  G=512 and 1024 (+19..+27 %). At G=64 they are at its level: A −1.9 %, DQ
  +0.0 %, DQR −0.0 %. This issue does not touch prefill.

## M==1 profile (G6), level 2, G=64, one run each (A DQ DQR)

`K=2048 N=2048 M==1 calls=1408`. zone0 before each run 42.4 / 42.8 / 43.2 °C.

| variant | dsp µs/call | mm | requant | quant | rest | swiglu (lane-µs) | feed / dmaq | first 3584 KB |
|---|---|---|---|---|---|---|---|---|
| A | 454.1 | 319.3 | 57.3 | 6.8 | 61.9 | 1198.0 | 1408/1408 dmaq=4.00 | 59 µs (the whole GU(0) run) |
| DQ | 427.1 (−27.0) | 351.8 | 4.5 | 7.8 | 54.2 | 1233.6 | 1408/1408 dmaq=4.00 | 50 µs (the remainder after QUANT) |
| DQR | 422.5 (−31.6) | 352.4 | 0.0 | 8.0 | 53.7 | 1234.5 | 1408/1408 dmaq=4.00 | 50 µs (the remainder after QUANT) |

* G6 holds. DQR's `dsp` is 31.6 µs below A's (≥ 25), its `requant` is 0 (no run
  B at n = 4), and `dmaq=4.00`. DQ's `rest` (54.2) is below A's (61.9), so the
  GU(0) job ran; the 3-worker fallback would have left `rest` at A's level.
* Where the 27 µs of DQ comes from: `requant` 57.3 → 4.5 (run B no longer
  waits for D(2)/D(3), and 4.5 is B's requant alone, 4 experts on 4 lanes),
  `rest` 61.9 → 54.2 (≈ 8 µs of GU(0) hidden behind QUANT, the plan's ≈ 7),
  and `mm` 319.3 → 351.8 (C(0) and C(1) now carry D(2) and D(3)). C3 adds
  −4.6 µs: run B's requant moves into the A and C runs.
* The plan's model expected ≈ 419 µs for DQR. The measurement is 422.5.
  About 346 µs of the call is transfer at 62 GB/s. What is left to recover is
  C(2)/C(3)'s idle queues and the GU(0) remainder, and the section comment's
  `ponytail:` names it.

## Bits (G1, G2) and text (G3)

| check | result |
|---|---|
| G1 dump: A vs #177's Q4 dump (`177/dump/Q4`), G=64 | `files=2862 bit_identical=1` |
| G1 dump: DQ vs A, DQR vs A | `files=2862 bit_identical=1` both |
| G2 nll: A's 512 `[PPL] decode step=` lines vs `177/logs/ppl_Q4.log` | byte-identical (`cmp` 0, 512 lines) |
| G2 nll: DQ, DQR forced on A's `cont.ids` (G=512) | byte-identical to A |
| G3 text, 8 prompts, G=64 | A ≡ #177 `text_Q4_p0{1..8}`: 8/8. DQ ≡ A: 8/8. DQR ≡ A: 8/8 |
| G7 hygiene | all 84 logs have `dspq: close calls=N served=N bad=0` (176 at G=8, 1408 at G=64, 11264 at G=512, 22528 at G=1024) and one banner kind (above). No `MD5 FAIL`, `dspq: off`, `gemv: off`, `arena: cannot` or `EXIT≠0`. The `pidof` guard never fired |
| G8 text vs CPU `q40` | not run. Information only on v81, and A ≡ #177 ≡ #168 carries #168's 0/8 |

The text comparison uses #158's `strip` plus a filter that drops the lines
echoing the model path (`s177/model`, `s185/model`).

## Host proof (rung 1) and the negatives

`bash test/htp/host/run_host_checks.sh` → `ALL CHECKS PASS` at each commit,
with `M1 FEED OVERLAP OK`, `M1 FEED DATAFLOW OK`, `M1 FEED QUEUES OK`,
`M1 FEED QUEUES 3-WORKER POOL OK (N=4, fed=4)` and `WORKER POOL LANES OK`.
`run_inproc_e2e.sh` at `7d0cb151` and at `8dea5f0a`, unset and with
`NNTR_MOE_DMA_QUEUES=4`, → `E2E eval golden … bit_identical=1`, `E2E tokens
htp==cpu 8/8`, `INPROC E2E PASS`. Each negative was run once and reverted:

| step | mutation | the check prints |
|---|---|---|
| C1 | D(2) posted on C(2)'s run instead of C(0)'s | `FEED read not covered: … push not waited` (lanes of C(2) read the slices lane 0 polls only after its units), `M1 FEED QUEUES WRONG` |
| C2 | `moe_m1_q_join` deleted | `FEED pool used while a submitted job is in flight` (169×), `M1 FEED QUEUES WRONG` |
| C3 | `m1q.rq = i` (requant in A(i)'s own run) | `FEED cross-lane RAW/WAR in run …: lane 5 reads +7168 B, lane 0 writes it` (a gate_f32 row), `M1 FEED DATAFLOW WRONG (624 races)` |

The C1 negative printed `push not waited`, not the plan's expected
`slice read in the run that issued it`. The reading lanes are not the
issuing lane, so the first check to fire is the wait.

## Notes from the run

* Order: sanity G=8 (A DQ DQR; all `calls=176 served=176 bad=0`, 43.5 /
  41.7 / 44.0 tok/s, not read as tok/s), profiles, the six mirrored rounds,
  the two re-run rounds, dumps, nll, and texts.
* Heat: every run took zone0 from ≈ 45 to 63–73 °C. Battery 28.2 → 40.3 °C
  over the sitting.
* Clean-up: `s185/` and `/data/local/tmp/s185dump` removed. `models/` was
  never written (its `nntr_config.json` still reads `num_to_generate: 512`).
* Workstation: `/local/mnt/workspace/htp_moe/185/{set,logs,dump}`, drivers
  `lib.sh`, `tps.sh`, `tps2.sh`, `bits.sh`, tables `table.py`.
