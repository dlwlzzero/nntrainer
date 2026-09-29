# Measurement 177: the decode MoE weight DMA over 1 / 2 / 4 queues on the S26 Ultra (v81)

Branch `htp/177-m1-dma-queues` @ `b1a7c4cc` (kernel `b1a7c4cc`, app knob
`dfd0242e`). Plan `docs/plans/177-m1-dma-queues.md`. Run by the implementer
agent on `R5KL20NFRCK` (SM-S948N, S26 Ultra) under the contract's v81 adb
exception, 2026-09-29 21:48–23:23. Device time ≈ 60 min.

## Why

The S26 decode (#168: ≈ 40–42 tok/s) trails the S25 (#158 B: 47–52), and the
M=1 feed moved every weight byte on one thread's DMA queue. Question: does
splitting each matrix over 2 or 4 queues raise decode tok/s with the bits
unchanged, and should the default flip from N = 1?

## Step 1 — go/no-go probe (G0)

`unittest_hvx_dma_probe --gtest_filter='*DmaSettings*'` with #168's v81 skel
(`fcc5b02e…`, `Flags: 0x81`), probe `5b3c4759…`, `libc++_shared.so`
`b1586b9b…`. Fresh 168 MiB footprint, f2 shape, tag-checked. Two runs, zone0
33.6 → 42.0 → 44.0 °C. `checksum_ok=y valid=y` in all 24 cells.

| src_bypass | queues | run 1 GB/s | run 2 GB/s | vs bypass 1 / 1 queue |
|---|---|---|---|---|
| 0 | 1 | 24.58 | 24.68 | |
| 0 | 2 | 31.16 | 31.17 | |
| 0 | 4 | 32.38 | 32.47 | |
| 1 | 1 | 33.00 | 33.06 | (anchor) |
| 1 | 2 | 43.22 | 43.29 | **+31 %** |
| 1 | 4 | 55.92 | 56.03 | **+69 %** |

G0 passes (≥ +10 %). Unlike the S25 (LEDGER rule 43: more queues added
nothing once bypass was on), on the S26 the queues scale with bypass on. No
cell exceeded 85.3 GB/s, so the DDR ceiling constant never fired.

## Artifacts (SDK 6.4.0.1, HexKL 1.0-beta.2 `lib/6.4.0.1`, NDK r30)

One binary set for all three variants, one full directory per variant under
`/data/local/tmp/nntrainer/causallm/s177/{A,Q2,Q4}`. `md5sum -c md5.txt`
passed on the phone before every run (`run()` exits on a failure; no
`MD5 FAIL` in any log).

| file | md5 | built with |
|---|---|---|
| libnntr_hvx_skel.so | `9b752543e38195ba011c81313deebb78` | `HEX_ARCH=v81 ./test/htp/build.sh` (`Flags: 0x81`, `UNDEFINED SYMBOLS OK (51)`) |
| libnntrainer.so | `2657fdae10ca91a16e06950fedc0ddae` | `(cd builddir && ninja install)` then `build_android.sh --htp --cache` |
| nntrainer_causallm | `0c0e0f78d86f202d91f3f86d67437406` | same (identical to #168's) |
| libcausallm_core.so | `de1323e80bb62d6c4607cef9efa7847c` | same (identical to #168's) |
| libccapi-nntrainer.so | `65c9034c6341384443166b89de66191d` | same (identical to #168's) |
| libc++_shared.so | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 |
| libsdkl.so | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `armv8_android26` |
| model `nntr_lfm2_8b_a1b_q40_arm.bin` | `7b7867fab51845664c0050c0a837073e` | `models/q40-qs4cx-wh`, read only |

The v79 skel builds too (`HEX_ARCH=v79`: `Flags: 0x79`, `UNDEFINED SYMBOLS
OK`, `002297dd…`). Not run.

| variant | `variant.env` | banner (every log of the variant, 22 each) |
|---|---|---|
| A | *(empty)* | `applied=0x703e1 … dma_bypass=1 dma_q=1 source=default` |
| Q2 | `export NNTR_MOE_DMA_QUEUES=2` | `applied=0xf03e1 … dma_bypass=1 dma_q=2 source=default` |
| Q4 | `export NNTR_MOE_DMA_QUEUES=4` | `applied=0x1f03e1 … dma_bypass=1 dma_q=4 source=default` |

Model directory: `s177/model` holds copies of the four JSON files and
symlinks to the model's `.bin` and `tokenizer.json`. Only its
`nntr_config.json` has `num_to_generate` rewritten per run, so nothing
outside `s177` / `s177dump` was written. The model is greedy
(`do_sample: false`), with `bad_word_ids: [124900]` and `moe_engine: htp`.
`NNTR_NUM_THREADS=8` and prompt 512 (`prompt512.txt`, `fc65c158…`) throughout.

## Results — decode / prefill (G4, G5), with the S25 comparison

Mirrored order A Q2 Q4 Q4 Q2 A per G (G=512: three rounds s1, s2, s5; s5 is
the G5 re-run). Cool-down to zone0 ≤ 45 °C before each run (0–120 s, in
`logs/tps_driver.log`, `tps5_driver.log`). zone0 in °C, battery in °C.
S25 reference: #158 B on `R3CY10WM83Y` — prefill 509.4 / 423.5 / 405.5,
decode 51.82 / 50.61 / 47.36 at G 64 / 512 / 1024.

| variant | G | round | prefill tok/s | decode tok/s (all) | decode (last 64) | vs S25 prefill / decode | zone0 pre → post | bat pre → post |
|---|---|---|---|---|---|---|---|---|
| A | 64 | s3r1 | 510.0 | 40.02 | 40.02 | +0.1 % / -22.8 % | 44 → 64 | 39.6 → 39.3 |
| Q2 | 64 | s3r2 | 506.9 | 45.39 | 45.39 | -0.5 % / -12.4 % | 45 → 68 | 39.7 → 39.7 |
| Q4 | 64 | s3r3 | 513.0 | 50.43 | 50.43 | +0.7 % / -2.7 % | 45 → 70 | 39.4 → 39.3 |
| Q4 | 64 | s3r4 | 514.6 | 50.79 | 50.79 | +1.0 % / -2.0 % | 45 → 70 | 39.7 → 39.6 |
| Q2 | 64 | s3r5 | 532.2 | 46.08 | 46.08 | +4.5 % / -11.1 % | 45 → 72 | 39.6 → 39.9 |
| A | 64 | s3r6 | 559.6 | 41.10 | 41.10 | +9.8 % / -20.7 % | 45 → 70 | 39.6 → 39.4 |
| A | 512 | s1r1 | 560.8 | 40.76 | 40.38 | +32.4 % / -19.5 % | 41 → 65 | 32.5 → 33.3 |
| Q2 | 512 | s1r2 | 499.5 | 45.09 | 45.39 | +17.9 % / -10.9 % | 44 → 64 | 34.4 → 34.7 |
| Q4 | 512 | s1r3 | 402.8 | 50.36 | 49.31 | -4.9 % / -0.5 % | 45 → 64 | 35.6 → 36.2 |
| Q4 | 512 | s1r4 | 524.1 | 50.88 | 49.92 | +23.7 % / +0.5 % | 43 → 65 | 36.2 → 36.4 |
| Q2 | 512 | s1r5 | 512.0 | 45.97 | 45.33 | +20.9 % / -9.2 % | 44 → 65 | 37.1 → 37.3 |
| A | 512 | s1r6 | 465.0 | 40.18 | 38.76 | +9.8 % / -20.6 % | 44 → 62 | 37.4 → 37.9 |
| A | 512 | s2r1 | 501.5 | 39.86 | 39.12 | +18.4 % / -21.2 % | 44 → 63 | 37.8 → 37.9 |
| Q2 | 512 | s2r2 | 520.9 | 44.89 | 43.90 | +23.0 % / -11.3 % | 45 → 64 | 38.7 → 38.6 |
| Q4 | 512 | s2r3 | 507.4 | 48.47 | 47.44 | +19.8 % / -4.2 % | 44 → 65 | 38.6 → 39.3 |
| Q4 | 512 | s2r4 | 420.0 | 50.00 | 48.71 | -0.8 % / -1.2 % | 45 → 65 | 38.7 → 39.1 |
| Q2 | 512 | s2r5 | 508.9 | 45.30 | 44.02 | +20.2 % / -10.5 % | 45 → 65 | 39.3 → 39.8 |
| A | 512 | s2r6 | 497.6 | 39.85 | 39.17 | +17.5 % / -21.3 % | 45 → 64 | 39.5 → 40.3 |
| A | 512 | s5r1 | 441.8 | 38.87 | 37.34 | +4.3 % / -23.2 % | 45 → 64 | 41.3 → 43.0 |
| Q2 | 512 | s5r2 | 365.5 | 44.14 | 42.33 | -13.7 % / -12.8 % | 45 → 65 | 41.1 → 42.1 |
| Q4 | 512 | s5r3 | 517.2 | 49.48 | 47.58 | +22.1 % / -2.2 % | 45 → 66 | 41.2 → 42.3 |
| Q4 | 512 | s5r4 | 539.5 | 49.44 | 47.16 | +27.4 % / -2.3 % | 45 → 66 | 40.9 → 41.7 |
| Q2 | 512 | s5r5 | 514.1 | 44.01 | 42.50 | +21.4 % / -13.0 % | 45 → 66 | 41.2 → 42.4 |
| A | 512 | s5r6 | 513.0 | 39.10 | 37.67 | +21.1 % / -22.7 % | 45 → 65 | 40.8 → 41.7 |
| A | 1024 | s4r1 | 518.7 | 39.14 | 37.17 | +27.9 % / -17.4 % | 44 → 65 | 39.7 → 41.6 |
| Q2 | 1024 | s4r2 | 542.9 | 41.00 | 40.46 | +33.9 % / -13.4 % | 44 → 65 | 40.3 → 42.3 |
| Q4 | 1024 | s4r3 | 517.7 | 47.75 | 44.72 | +27.7 % / +0.8 % | 45 → 66 | 40.4 → 41.9 |
| Q4 | 1024 | s4r4 | 507.9 | 47.07 | 44.38 | +25.3 % / -0.6 % | 45 → 66 | 40.8 → 42.5 |
| Q2 | 1024 | s4r5 | 513.5 | 42.78 | 40.18 | +26.6 % / -9.7 % | 45 → 66 | 40.9 → 43.0 |
| A | 1024 | s4r6 | 516.6 | 38.47 | 36.70 | +27.4 % / -18.8 % | 45 → 66 | 40.7 → 43.0 |

Means (min..max):

| variant | G | n | prefill | decode (all) | decode vs A | decode vs S25 | prefill vs A | prefill vs S25 |
|---|---|---|---|---|---|---|---|---|
| A | 64 | 2 | 534.8 (510.0..559.6) | 40.56 (40.02..41.10) | — | -21.7 % | — | +5.0 % |
| Q2 | 64 | 2 | 519.6 (506.9..532.2) | 45.73 (45.39..46.08) | +12.7 % | -11.8 % | -2.8 % | +2.0 % |
| Q4 | 64 | 2 | 513.8 (513.0..514.6) | **50.61** (50.43..50.79) | **+24.8 %** | -2.3 % | -3.9 % | +0.9 % |
| A | 512 | 6 | 496.6 (441.8..560.8) | 39.77 (38.87..40.76) | — | -21.4 % | — | +17.3 % |
| Q2 | 512 | 6 | 486.8 (365.5..520.9) | 44.90 (44.01..45.97) | +12.9 % | -11.3 % | -2.0 % | +14.9 % |
| Q4 | 512 | 6 | 485.2 (402.8..539.5) | **49.77** (48.47..50.88) | **+25.1 %** | -1.7 % | -2.3 % | +14.6 % |
| A | 1024 | 2 | 517.7 (516.6..518.7) | 38.81 (38.47..39.14) | — | -18.1 % | — | +27.7 % |
| Q2 | 1024 | 2 | 528.2 (513.5..542.9) | 41.89 (41.00..42.78) | +7.9 % | -11.6 % | +2.0 % | +30.3 % |
| Q4 | 1024 | 2 | 512.8 (507.9..517.7) | **47.41** (47.07..47.75) | **+22.2 %** | +0.1 % | -0.9 % | +26.5 % |

* **G4:** Q4's mean is above A's max at every G (G=512: 49.77 > 40.76; even
  Q4's min, 48.47, is above it), and Q2's is too. So the plan's flip rule
  holds for Q4 (§1): the PR proposes the flip, and the user decides.
* **G5:** passes on the G=512 means over six runs each: Q2 −2.0 %, Q4
  −2.3 % (≥ −5 %). After rounds s1 and s2 alone, Q4 read −8.4 %, from two
  outliers (402.8, 420.0). Round s5 was the plan's re-run. Prefill outliers
  land on every variant (A 441.8, 465.0; Q2 365.5), and the prefill path
  (M = 512, HMX) never reads the queue bits (plan §1 G5).
* **S25 goal (user, 2026-09-29):** Q4 matches the S25 decode within
  −2.3..+0.1 % (−2.3 % at G=64, −1.7 % at 512, +0.1 % at 1024) and beats its
  prefill at every G. It does not yet beat the S25 decode at G=64 and 512.
  A trails the S25 decode by 18–22 %.

## M==1 profile (level 2, G=64, one run each, mirrored A Q2 Q4)

`K=2048 N=2048 M==1 calls=1408`. zone0 before each run 44.8 / 45.9 / 45.5 °C.

| variant | dsp µs/call | mm | requant | quant | rest | feed / dmaq | weight DMA, first 3584 KB |
|---|---|---|---|---|---|---|---|
| A | 685.7 | 653.2 | 12.4 | 7.1 | 3.8 | 1408/1408 dmaq=1.00 | 131 µs = 28.1 GB/s |
| Q2 | 552.1 | 384.3 | 72.0 | 6.9 | 79.8 | 1408/1408 dmaq=2.00 | 77 µs = 47.8 GB/s |
| Q4 | 456.0 | 320.4 | 57.5 | 6.9 | 62.1 | 1408/1408 dmaq=4.00 | 59 µs = 62.1 GB/s |

How to read it (plan §5):

* **The first-chunk figures are not comparable.** A's `DMA_FIRST` times only
  the remainder exposed after the scan: its GU(0) went out before it. Q's
  times the whole GU(0) transfer in a run of its own, so Q2's 47.8 and Q4's
  62.1 GB/s are real transfer rates, and in app they meet or beat the
  probe's 43.2 / 55.9.
* **A Q variant's `rest` is that GU(0) run.** It sits outside every named
  stage (≈ 77 / 59 µs, matching the first-chunk times).
* **Q's `requant` includes D(2) and D(3).** It is 72.0 / 57.5 against A's
  12.4, because those two downs are polled inside run B. This is §3's
  ponytail showing: B's wall ≫ its requant.
* **`DMA ring:` covers only the two copy descriptors under Q** (desc=2/call),
  so its engine figures mean nothing there. The `weight DMA … averaged over
  the call` figure is bytes / mm-stage time, not a DMA rate, and is not used.

Where Q4's 456 µs goes: 62 rest (the exposed GU(0) run) + 320 mm + 57.5
requant (≈ 45 of it D(2)/D(3), against A's 12.4 of compute) + 16 of quant,
scatter and stage. 21.5 MB at the in-app
62 GB/s is ≈ 346 µs. So ≈ 100 µs/call (GU(0) exposed, D(2)/D(3) serialised
behind requant) is schedule, not bandwidth. At 22 MoE calls per token, that
is ≈ 2.2 ms of Q4's ≈ 20 ms token, enough to clear the S25 decode.

## Bits (G1, G2) and text (G6)

| check | result |
|---|---|
| G1 dump: A vs #168's v81 dump (`168/dump/v81`), G=64 | `files=2862 bit_identical=1` |
| G2 dump: Q2 vs A, Q4 vs A | `files=2862 bit_identical=1` both |
| G1 nll: A's 512 `[PPL] decode step=` lines vs `168/logs/ppl_npu.log` | byte-identical (`cmp` 0); nll/token 0.216411, top1 512/512 |
| G2 nll: Q2, Q4 forced on A's `cont.ids` | byte-identical to A (`source=file`) |
| text, 8 prompts, G=64 | A ≡ #168 `text_npu_p0{1..8}`: 8/8. Q2 ≡ A: 8/8. Q4 ≡ A: 8/8 |
| G6 text vs CPU `q40` (`168/logs/text_cpu_p0*`, information only) | 0/8 identical; first differing word 483 / 179 / 98 / 126 / 133 / 210 / 320 / 23 (same as A = #168) |
| G3 | every one of the 66 logs: `dspq: close calls=N served=N bad=0` (176 at G=8, 1408 at G=64, 11264 at G=512, 22528 at G=1024); one banner kind per variant (table above); no `MD5 FAIL`, `dspq: off`, `gemv: off`, `arena: cannot` or `EXIT≠0` |

Text comparison: #168's `strip`, plus dropping the two lines that echo the
model path (`s177/model` here, `models/q40-qs4cx-wh` in #168). Those two
lines were the only difference before the filter.

## Notes from the run

* All three variants share one binary set. A is the new code with the knob
  unset, i.e. #117's ring path, which is bit-identical to #168 (G1).
* Heat: every run takes zone0 from ≈ 45 to 62–72 °C. Cool-downs took 0–120 s,
  longer late in the sitting (battery 32.5 → 43 °C).
* The shared checkout was switched to `htp_moe_v81` and back for about 15 s
  by the orchestrator after the builds. The staged artifacts are untracked
  build outputs whose md5s match the post-build ones above, so they were
  unaffected.
* Clean-up: `s177/` and `/data/local/tmp/s177dump` removed. `models/` was
  never written (its `nntr_config.json` still reads `num_to_generate: 512`).
* Workstation logs: `/local/mnt/workspace/htp_moe/177/{probe,logs,dump}`.
