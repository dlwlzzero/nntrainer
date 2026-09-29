# Measurement 168: the MoE decode on the Galaxy S26 Ultra NPU (SM8850, HTP v81), end to end

Branch `htp/168-s26-v81-bringup` @ `fc6bb52a` (code commit `76903a12` on
`htp_moe_v81` @ `70365453`). **Agent-run sitting** on `R5KL20NFRCK`
(contract decision log, 2026-09-29), 2026-09-29 17:29–17:45, ≈ 16 min of
device time plus ≈ 3.5 min of model push. Recipe: plan
`docs/plans/168-s26-v81-bringup.md` §4 step 5, run as written (plus one
diagnostic cell, 5.6b). Raw logs: `/local/mnt/workspace/htp_moe/168/logs/`
(workstation).

## Why
Does the v81 skel load in an unsigned PD on the S26 and run every MoE layer
on the DSP, bit for bit as the S25 (v79) does? The answer decides whether
`htp_moe_v81` can carry the decode work on this phone; tok/s are recorded,
not gated.

## Verdict

| # | check | result |
|---|---|---|
| G0 | skel on the device is v81 | **pass**: device md5 `fcc5b02e…` = staged file, `hexagon-readelf -h` → `Flags: 0x81, V81`; `build.sh` printed `ARCH OK (V81)` |
| G1 | the DSP ran the MoE, no CPU fallback | **pass**: every NPU log (18) has the banner `[HTP] moe m1 gemv: on (applied=0x703e1) lead=192KB rows1=1 feed=vtcm dma_bypass=1 source=default` once, `dspq: on` once, `dspq: close calls=served=22·G bad=0` (176 / 1408 / 11264 / 22528); no `dspq: off`, `moe m1 gemv: off`, `HTP arena: cannot`; no logcat `nntr_hvx_open failed`; profile `K=2048 N=2048 M==1 calls=1408 … m1_gemv=1408/1408 feed=1408/1408`; arena mapped in full (15 × 256 MiB = 3840 MiB). CPU logs carry no `moe m1 gemv` / `dspq` line |
| G2 | MoE dumps v81 ≡ v79 (#158 B) | **pass**: `E2E eval v81 files=2862 bit_identical=1 min_snr_db=inf first_diff=-` |
| G3 | decode nll v81 ≡ v79 | **pass**: 512 `[PPL] decode step=` lines byte-identical to `158b/logs/ppl_A.log` and `ppl_B.log`; `nll/token=0.216411 ppl=1.24161 top1=512/512 nll_sum=110.80222048851223` on both phones |
| G4 | text v81 ≡ v79 | **pass**: p01 at G 64 / 512 / 1024 identical to `158b/logs/B_G{64,512,1024}_r1.log`; 8/8 prompts at G=64 identical to `text_B_p0{1..8}.log` |
| G5 | text NPU vs CPU `q40` (the issue's wording) | **n** on every prompt and G, as on the S25 (different MoE weights). p01: first difference at generated word **42 (0-based) = word 43** at G 64 / 512 / 1024, the S25's index. `NNTR_L2_DIFF`: n/a, no hook on the MoE-layer path. **Needs the user's reading (issue comment).** |
| G6 | tok/s recorded | below |
| G7 | v79 default unchanged | **pass**: unset `HEX_ARCH` → `UNDEFINED SYMBOLS OK (51 runtime imports)`, `ARCH OK (V79)` |

Unsigned PD: logcat shows `remote_session_control Unsigned PD enable 1`,
`Successfully opened file /vendor/dsp/cdsp/fastrpc_shell_unsigned_3`,
`Created user PD on domain 100000 … Unsigned:Y, Signed:N`. No AEE
`0x80000481 / 0x80000438 / 0x80000415`. The other logcat errors in every
run (`open_shell failed for domain 3 … Permission denied` before the
unsigned shell opens, `Unable to add watcher for folder /vendor/…`,
`apps_std_fopen_fd failed for ./cdsp/./libnntr_hvx_skel.so` before
`././libnntr_hvx_skel.so` opens, `adspmsgd … 0x8000040d` at teardown) also
appear in the CPU runs, which open the session too, and none of them stops a
run: 33/33 logs end `EXIT=0`.

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL beta.2 6.4.0.1, NDK r30, v81)

Staged in `/local/mnt/workspace/htp_moe/168/`; the device `md5sum` of all 15
files matched `md5.txt` (`MD5 OK`).

| file | md5 | built with |
|---|---|---|
| libnntr_hvx_skel.so | `fcc5b02ee592c053275b2ad6f01dbe7d` | `HEX_ARCH=v81 ./test/htp/build.sh` @ `76903a12` (`ARCH OK (V81)`); the v79 build of the same tree: `76ac026bc52764b374a23b0eef4cdc20` |
| nntrainer_causallm | `0c0e0f78d86f202d91f3f86d67437406` | `build_android.sh --htp --cache` @ `fc6bb52a` (after `ninja -C builddir install`: the installed `compute_ops.h` was stale) |
| libcausallm_core.so | `de1323e80bb62d6c4607cef9efa7847c` | same; `NNTR_HTP_FORWARD_KINDS` count 2 |
| libnntrainer.so | `e46888353d41e8232d6b43c4c6a9698a` | same (`jni/obj/local`); NEEDED `libsdkl.so`, `libcdsprpc.so` |
| libccapi-nntrainer.so | `65c9034c6341384443166b89de66191d` | same (`jni/obj/local`) |
| libc++_shared.so | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 |
| libsdkl.so | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `6.4.0.1/armv8_android26` |
| prompt512.txt | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt` |
| bitset-0{2..8}-*.txt | as `158b/md5.txt` | `docs/measurements/prompts/` |
| models/q40/nntr_lfm2_8b_a1b_q40_arm.bin | `d28f55c5bd7adeb8bf73b02de582eb88` | CPU control (device md5) |
| models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin | `7b7867fab51845664c0050c0a837073e` | NPU model (device md5) |

Device paths: `/data/local/tmp/nntrainer/causallm/s168/` (binaries),
`…/causallm/models/q40{,-qs4cx-wh}/` (kept for later `htp_moe_v81` issues),
`/data/local/tmp/s168dump/` (deleted after the sitting). Config on both
models: `do_sample: false`, `bad_word_ids: [124900]`, `init_seq_len: 512`;
`q40`: no `_engine` key, `moe_layer_dtype: Q4_0`; `q40-qs4cx-wh`:
`moe_engine: htp`, `moe_htp_layers: ""`, `QS4CX_WH`.

## Device

`adb devices`: `R3CN80CW3FY`, `R3CY10WM83Y`, `R5KL20NFRCK` (only the last was
addressed). SM-S948N, `ro.soc.model=SM8850`, Android 16, USB powered (not
AC), screen off (`mWakefulness=Dozing`). `/data` 50 GB free before the push.

Thermal checkpoints (battery level, battery °C·10, zone0 m°C):

| time | point | level | batt | zone0 |
|---|---|---|---|---|
| 17:29:30 | 0, before sanity | 97 | 270 | 30800 |
| 17:30:17 | before CPU tok/s | 97 | 299 | 41300 |
| 17:33:06 | after CPU tok/s | 95 | 373 | 61400 |
| 17:33:44 / 17:34:43 / 17:36:10 | after NPU G 64 / 512 / 1024 | 94 / 94 / 93 | 380 / 391 / 409 | 63300 / 64800 / 66000 |
| 17:37:49 | after dump + ppl | 92 | 403 | 65600 |
| 17:43:52 | after the text set | 90 | 421 | 61400 |
| 17:45:08 | after the profiles | 90 | 403 | 45100 |

The phone warmed from 27 °C to 42 °C (battery) over the sitting; the NPU cells
ran after the CPU cells, on a warmer phone (plan order).

## Results — prompt 512, `NNTR_NUM_THREADS=8`, non-profile binary

| engine | gen | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS (KB) | `dspq: close` | text ≡ S25 v79 B | text = CPU q40 |
|---|---|---|---|---|---|---|---|---|---|
| CPU `q40` | 64 | 1 | 200.08 | 52.29 | 52.29 | 4947196 | — | — | reference |
| CPU `q40` | 64 | 2 | 336.62 | 51.45 | 51.49 | 5099160 | — | — | r2 ≡ r1 |
| CPU `q40` | 512 | 1 | 326.32 | 51.24 | 51.00 | 5362060 | — | — | reference |
| CPU `q40` | 512 | 2 | 262.16 | 50.74 | 50.24 | 5406288 | — | — | r2 ≡ r1 |
| CPU `q40` | 1024 | 1 | 284.44 | 48.52 | 47.41 | 5233416 | — | — | reference |
| CPU `q40` | 1024 | 2 | 241.28 | 48.08 | 45.98 | 5465660 | — | — | r2 ≡ r1 |
| NPU `q40-qs4cx-wh` | 64 | 1 | 505.93 | 41.64 | 41.64 | 5301836 | 1408/1408/0 | y | n (word 43) |
| NPU `q40-qs4cx-wh` | 64 | 2 | 349.01 | 42.24 | 42.24 | 3344244 | 1408/1408/0 | y (≡ r1) | n |
| NPU `q40-qs4cx-wh` | 512 | 1 | 406.03 | 42.45 | 41.75 | 4747976 | 11264/11264/0 | y | n (word 43) |
| NPU `q40-qs4cx-wh` | 512 | 2 | 337.73 | 42.07 | 40.66 | 4732004 | 11264/11264/0 | y (≡ r1) | n |
| NPU `q40-qs4cx-wh` | 1024 | 1 | 433.16 | 40.43 | 39.80 | 4836572 | 22528/22528/0 | y | n (word 43) |
| NPU `q40-qs4cx-wh` | 1024 | 2 | 420.02 | 39.90 | 38.30 | 4698384 | 22528/22528/0 | y (≡ r1) | n |

Means, CPU / NPU: decode **51.87 / 41.94** (G=64), **50.99 / 42.26**
(G=512), **48.30 / 40.17** (G=1024); prefill 268.4 / 427.5, 294.2 / 371.9,
262.9 / 426.6. **On this phone the NPU decode is 19 % below the CPU and
below the S25's NPU** (#158 B, `R3CY10WM83Y`: 51.82 / 50.61 / 47.36, prefill
509.4 / 423.5 / 405.5). The CPU control is on par with the S25-era CPU cells
(#94 s2 A 52.43 / 49.22 / 48.31 on `R3CY205ZMND`). These are the first cells
of the S26 block (a different unit, never compared unscaled with S25 rows,
LEDGER rule 13 / 34). The first NPU prefill (505.9) is the `htp_moe_v81` A
for later prefill gates; the prefill cells spread 337–506 with the phone's
temperature, as on the S25.

Other sitting cells (not tok/s): sanity G=8: CPU prefill 140.1 / decode 22.8
(first run after the model push, cold page cache), NPU 550.5 / 35.7,
`dspq: close calls=176 served=176 bad=0`. Dump run G=64: 484.8 / 39.19.
PPL run G=512: 518.7 / 40.26.

## Accuracy detail

G5, first differing word of NPU vs CPU (generated-word index, 0-based;
the echoed prompt is word-for-word the prompt file in every log):

| prompt | tokens | NPU ≡ S25 v79 B | NPU = CPU | first diff (gen. word) | NPU / CPU word |
|---|---|---|---|---|---|
| p01 prompt512 | 512 | y | n | 42 | `museum` / `museum,` |
| p02 code | 207 | y | n | 38 | `list` / `write` |
| p03 math | 117 | y | n | 0 | `The` / `Check` |
| p04 korean | 326 | y | n | 1 | `"시전` / `"한양은` |
| p05 json | 207 | y | n | 0 | `I` / `Return` |
| p06 dialogue | 276 | y | n | 0 | `That` / `A` |
| p07 facts | 402 | y | n | 1 | `final` / `concluding` |
| p08 short | 24 | y | n | 1 | `colour` / `usual` |

(The CPU `q40` p03 text loops on "Check your answer at the end." from its
first generated word; noted, not investigated.)

p01, G=64, run 1, generated text:

* NPU: ` town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station`
* CPU: ` town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself`

## Level-2 profile, G=64 (5.6; not tok/s)

`[HTP-PROFILE] level=2 qos_mode=2`. Rows (S25 v79 #158 B in brackets):

* `K=2048 N=2048 M==1 calls=1408 … host= 694.6 us/call dsp= 680.2 (97.9 %) transport= 14.4 … mm 647.4 … blocks=0 m1_gemv=1408/1408 feed=1408/1408` [dsp 417.5, mm 388]
* `weight DMA: 21504 KB/call, first 3584 KB took 124 us = 29.5 GB/s; averaged over the call 32.4 GB/s`
* `DMA ring: desc=10/call waits=10 (blocked 9.9) wait=447.1 us … busy=622..656 us -> engine 33.6..35.4 GB/s depth max=4 first expert ready at 133 us last issue at 483 us of 680` [engine 56.3..57.9 GB/s]
* `K=2048 N=2048 M>1 calls=23 … dsp=14690.6 us/call … mm 9675.7 … m1_gemv=0/23 feed=0/23`; ring `engine 22.8..50.5 GB/s` [dsp 14723.8]

**5.6b (diagnostic, added):** the same run with `NNTR_MOE_DMA_BYPASS=0`
(banner `applied=0x303e1 … dma_bypass=0`): M==1 `dsp= 812.4 us/call mm
784.9`, ring `engine 28.0..29.9 GB/s`, decode 35.36 tok/s at G=64; text ≡
the default run. So `src_bypass` still helps on v81 (680 vs 812 µs, +19 %),
but the v81 DMA engine in this app reads ≈ 34 GB/s with it, where the S25's
reads ≈ 57. **That covers the decode gap to the S25**: 22 × (680 − 418) µs
≈ 5.8 ms/token of DSP time, against a measured token-time gap of ≈ 4.6 ms
(23.8 vs 19.3 ms at G=64). Why the v81 ring is slower (DDR, DVFS vote, v81
`src_bypass` / L2 behaviour) is not answered by this sitting.

VTCM size and HVX context count are not printed (FARF HIGH is compiled out);
not read in this sitting.

## Notes from the run

* Unsigned PD opens for the shell user on this Android 16 build; no signed
  skel and no root were needed.
* `builddir`'s installed headers were from another branch
  (`gemm_qs4cx_moe_layer_fp32` without `down_hadamard`); `ninja -C builddir
  install` before `build_android.sh --htp --cache` fixed it, as the plan
  anticipated.
* The `strip` function was checked on the 158b logs: it yields the
  prompt echo plus the generated text; the model-path lines are identical
  between v81 and v79 logs (same device path) and are dropped for the
  NPU-vs-CPU word comparison.
* Host rung on this branch: `*qs4cx*` 2 passed, `*Lfm2Moe*` 6 passed,
  `ALL CHECKS PASS`, `WORKER POOL LANES OK`, syntax check exit 0,
  `INPROC E2E PASS`.
