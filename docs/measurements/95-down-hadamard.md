# Measurement 95: Hadamard rotation on the MoE down_proj input — latency and accuracy, on/off

Branch `htp/95-down-hadamard` @ `6c89a912` (base `htp_hadamard` = upstream
PR #4327 head `3006d255` + tooling + plan) — estimated device time: **75 min**
(+5 min if the rotated model must be pushed, +8 min per other model).

## Why

Does folding a block-256 Hadamard into every expert's `down_proj` weight and
running the matching FWHT on the DSP before the u8 requant lower perplexity,
and what does it cost in prefill and decode tok/s? The decision that hangs on
it: carry `QS4CX_WH_HAD` over to `htp_moe` (plus the M=1 GEMV insert site), or
close #95 with these numbers as "no effect".

## Variants (at most 4, A first)

| variant | binaries (device dir) | skel | model | reads as |
|---|---|---|---|---|
| **A** | `$W/tpsA` → `$DA` | `libnntr_hvx_skel.A.so` | `q40-qs4cx-wh` | control: `htp_hadamard` @ `63167235`, unchanged code |
| **B** | `$W/tps` → `$D` | `libnntr_hvx_skel.so` (new) | `q40-qs4cx-wh` | code change alone, flag off: **text bit-identical to A, PPL == A**, TPS within ±5 % |
| **C** | `$W/tps` → `$D` | `libnntr_hvx_skel.so` (new) | `q40-qs4cx-wh-had` | the measurement: flag set by the model's dtype |
| **D** | `$W/tpsA` → `$DA` | (not used) | `q40`, CPU | text / PPL reference |

Two device directories so no file is ever swapped mid-sitting: `$DA` holds A's
binaries and skel, `$D` holds B/C's. Both see the same `models/`.

## Artifacts (built on the workstation 2026-09-22, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

Staged under **`W=/local/mnt/workspace/htp_hadamard/95/`**; `$W/md5.txt` is
the `md5sum` of every staged file.

| file (staged as) | md5 | built from |
|---|---|---|
| `$W/libnntr_hvx_skel.A.so` | `5c20b19d2795787a70575af045beb059` | `test/htp/build.sh` @ `63167235` (`htp_hadamard`) |
| `$W/libnntr_hvx_skel.so` | `851dde2ce013603ba99e03860d9a0442` | `test/htp/build.sh` @ `84c5f71a` (no DSP source changed after it); exports `hvx_fwht_rows_f32` |
| `$W/tpsA/libnntrainer.so` | `1c09a8602ca389c53289a5c9e6be7897` | `generate_stub.sh`, `builddir` `ninja install`, `build_android.sh --htp --cache` @ `63167235` |
| `$W/tpsA/libcausallm_core.so` | `49380a583c7b43c9865e0c30621453c0` | same |
| `$W/tpsA/libccapi-nntrainer.so` | `65c9034c6341384443166b89de66191d` | same |
| `$W/tps/libnntrainer.so` | `1fbae5bc8724e601664ed06dceb22b11` | same recipe @ `6c89a912` |
| `$W/tps/libcausallm_core.so` | `ea91835a85cad4683144d72f1da90a82` | same (unchanged since `84c5f71a`) |
| `$W/tps/libccapi-nntrainer.so` | `30fece8b96a1c6c349167aca42b15e70` | same (unchanged since `84c5f71a`) |
| `$W/tpsA/nntrainer_causallm`, `$W/tps/nntrainer_causallm` | `b02a47e4fcc4b18dc2e2818e049669e9` | identical in both builds (thin executable) |
| `$W/gtest/unittest_hvx_mm_u8i4` | `48f523ac775a4a0f34725cbebcd0de68` | `ndk-build` @ `84c5f71a` |
| `$W/gtest/unittest_hvx_softmax` | `0401d81dc3c430a6e52d165d9d596dd9` | same |
| `libc++_shared.so` (all three dirs) | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 (== #94) |
| `libsdkl.so` (all three dirs) | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` (== #94) |
| `$W/prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | == `docs/measurements/77-prompt512.txt` used by #77/#94; 512 tokens |
| `$M/q40/nntr_lfm2_8b_a1b_q40_arm.bin` (4768855808 B) | `d28f55c5bd7adeb8bf73b02de582eb88` | #78, CPU control |
| `$M/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (4316133120 B) | `7b7867fab51845664c0050c0a837073e` | #78, `--moe_dtype QS4CX_WH` |
| `$M/q40-qs4cx-wh-had/nntr_lfm2_8b_a1b_q40_arm.bin` (4316133120 B) | `a2829cd9d33209669d03b1613b0f5a02` | `nntr_quantize_stream` @ `22ab4b3e`, `--moe_dtype QS4CX_WH_HAD` |

Workstation checks already done (2026-09-22): both skels pass the #97
undefined-symbol guard (44 runtime imports each, no project symbol);
`libnntrainer.so` NEEDED lists `libsdkl.so` and `libcdsprpc.so` in both
builds; neither `nntrainer_causallm` is a profile build. The rotated model
differs from `q40-qs4cx-wh` in exactly 704 regions of ≈ 1.81 MB
(22 MoE layers × 32 experts × one `down` tensor); every gate_up, attention,
conv, embedding byte is identical.

```
W=/local/mnt/workspace/htp_hadamard/95; M=/local/mnt/workspace/models/lfm2.5-8b-a1b
(cd $W && md5sum -c md5.txt)          # every line OK
find $W -name 'libcdsprpc*' | wc -l   # 0
```

## Steps (workstation, phone on USB)

Shell setup once:

```
cd /home/j2z0-lee/nntrainer && git fetch && git checkout htp/95-down-hadamard && source tools/htp/env.sh
W=/local/mnt/workspace/htp_hadamard/95; M=/local/mnt/workspace/models/lfm2.5-8b-a1b; mkdir -p $W/logs
D=/data/local/tmp/nntrainer/causallm; DA=/data/local/tmp/nntrainer/causallm_A; T=/data/local/tmp/htp_u8i4_layer_test
therm() { adb shell dumpsys battery | grep -E 'level|temperature'; adb shell cat /sys/class/thermal/thermal_zone0/temp; }
```

### 0. Device state (2 min)

```
adb devices; therm; adb shell df -h /data
```

Note battery %, temperature, warm/cool under Notes. Screen off, charger in.

### 1. Install (5–20 min)

Models: `q40` and `q40-qs4cx-wh` are on the device from #94 if nothing
removed them. Check sizes and hashes; push only what is missing or wrong.

```
adb shell ls -l $D/models/*/nntr_lfm2_8b_a1b_q40_arm.bin
adb shell md5sum $D/models/*/nntr_lfm2_8b_a1b_q40_arm.bin    # ≈ 1 min each; must equal the table
adb push $M/q40-qs4cx-wh-had $D/models/q40-qs4cx-wh-had        # new; ≈ 5 min
```

Binaries, both dirs:

```
adb shell mkdir -p $D $DA $T
adb push $W/tps/nntrainer_causallm $W/tps/libcausallm_core.so $W/tps/libnntrainer.so $W/tps/libccapi-nntrainer.so $W/tps/libc++_shared.so $W/tps/libsdkl.so $D/
adb push $W/libnntr_hvx_skel.so $D/libnntr_hvx_skel.so
adb push $W/tpsA/nntrainer_causallm $W/tpsA/libcausallm_core.so $W/tpsA/libnntrainer.so $W/tpsA/libccapi-nntrainer.so $W/tpsA/libc++_shared.so $W/tpsA/libsdkl.so $DA/
adb push $W/libnntr_hvx_skel.A.so $DA/libnntr_hvx_skel.so
adb push $W/prompt512.txt $D/prompt512.txt
adb push $W/prompt512.txt $DA/prompt512.txt
adb shell "rm -rf $DA/models && ln -s $D/models $DA/models"
adb push $W/gtest/unittest_hvx_mm_u8i4 $W/gtest/unittest_hvx_softmax $W/gtest/libc++_shared.so $W/gtest/libsdkl.so $T/
adb push $W/libnntr_hvx_skel.so $T/libnntr_hvx_skel.so
adb shell "chmod 755 $D/nntrainer_causallm $DA/nntrainer_causallm $T/unittest_hvx_mm_u8i4 $T/unittest_hvx_softmax"
```

Never push `builddir/.../libcdsprpc.so`.

Config edits, re-applied unconditionally (greedy, exact generation count):

```
for m in q40 q40-qs4cx-wh q40-qs4cx-wh-had; do
  adb shell "cd $D/models/$m && \
    sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
    sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
    grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|init_seq_len|moe_engine|moe_htp_layers|moe_layer_dtype' nntr_config.json"
done
```

Expected: `do_sample": false`, `bad_word_ids": [124900]`, `init_seq_len": 512`
for all three; the two NPU dirs show `moe_engine": "htp"`, `moe_htp_layers": ""`
and `moe_layer_dtype` `QS4CX_WH` / `QS4CX_WH_HAD` respectively.

Provenance (paste under Notes; every hash must equal the table):

```
adb shell "md5sum $D/libnntr_hvx_skel.so $DA/libnntr_hvx_skel.so $T/libnntr_hvx_skel.so $D/libnntrainer.so $DA/libnntrainer.so $D/libcausallm_core.so $DA/libcausallm_core.so $T/unittest_hvx_mm_u8i4 $T/unittest_hvx_softmax"
```

### 2. Device gtests first (3 min) — new skel

```
adb shell "cd $T && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_softmax --gtest_filter='HvxFwht*'" 2>&1 | tee $W/logs/G_fwht.log | grep -E 'OK|PASSED|FAILED|mismatch|stage'
adb shell "cd $T && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_mm_u8i4 --gtest_filter='*Hadamard*:*MoeSetOpts*'" 2>&1 | tee $W/logs/G_moe.log | grep -E 'OK|PASSED|FAILED|snr|SNR'
```

Pass: `HvxFwht.MatchesScalarBitExact`, `HvxFwht.RejectsPartialBlock`,
`HmxMmU8I4Layer.MoeLayerHadamardMatchesTwoCallReference`,
`HmxMmU8I4Layer.MoeSetOptsEchoesKnownBits` all `OK`.
If only `MatchesScalarBitExact` fails and only on subnormal inputs, paste the
per-stage lines and **continue** with the E2E runs (plan §4 step 5: the
reference's flushed-zero sign, fixed in one round, does not change TPS).
Any other gtest failure: stop, paste the log, do not run C.

### 3. TPS cells (≈ 40 min) — prompt 512, G 64 / 512 / 1024, twice each

```
run() { # $1 variant  $2 dir  $3 model  $4 G  $5 rep
  adb shell "cd $2 && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $4/' models/$3/nntr_config.json && \
    NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/$3 \"\$(cat prompt512.txt)\"" \
    2>&1 | tee $W/logs/$1_G$4_r$5.log | grep -E '^(prefill|generation|total|peak memory)|opts applied'
}
for G in 64 512 1024; do run D $DA q40 $G 1; done
for G in 64 512 1024; do for r in 1 2; do run A $DA q40-qs4cx-wh    $G $r; done; done
for G in 64 512 1024; do for r in 1 2; do run B $D  q40-qs4cx-wh    $G $r; done; done
for G in 64 512 1024; do for r in 1 2; do run C $D  q40-qs4cx-wh-had $G $r; done; done
for G in 64; do run A $DA q40-qs4cx-wh $G 3; done     # A again at the end: drift check
therm
```

Expected per run: `prefill: 512 tokens, <ms> ms, <tps> TPS`,
`generation: <G> tokens, <ms> ms, <tps> TPS`. Once per run, B prints
`[HTP-MOE] opts applied=0` and C prints `[HTP-MOE] opts applied=2`; A and D
print no `opts` line. If B or C throws `nntr_hvx_moe_set_opts(...) failed ... older than this binary`, `$D`
holds the wrong skel: re-push and re-run.

Text comparison (workstation, after the loop; the text is everything
before the first `=====` line, as in #94):

```
txt() { sed -n '/^=====/q;p' "$1"; }
for g in 64 512 1024; do
  for pair in "A_G${g}_r1 A_G${g}_r2" "B_G${g}_r1 A_G${g}_r1" "C_G${g}_r1 C_G${g}_r2" "C_G${g}_r1 A_G${g}_r1" "C_G${g}_r1 D_G${g}_r1" "A_G${g}_r1 D_G${g}_r1"; do
    set -- $pair; printf '%-12s vs %-12s: ' $1 $2
    diff <(txt $W/logs/$1.log) <(txt $W/logs/$2.log) >/dev/null && echo same || echo DIFFERENT
  done
done
```

For every DIFFERENT, note the first differing word (`cmp` on the two
`txt` outputs gives the byte offset) in the "text ==" columns.

### 4. Perplexity (≈ 6 min) — one run each, G=64, not for TPS

```
ppl() { adb shell "cd $2 && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/$3/nntr_config.json && \
  NNTR_PPL=1 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/$3 \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/$1_ppl.log | grep '^\[PPL\]'; }
ppl D $DA q40; ppl A $DA q40-qs4cx-wh; ppl B $D q40-qs4cx-wh; ppl C $D q40-qs4cx-wh-had
```

Expected: `[PPL] prompt tokens=511 nll/token=<x> ppl=<y>` (or 512).
B must print the same `ppl=` as A to the last digit.

### 5. Requant SNR (≈ 10 min; host GEMM per expert call, slow) — B and C, G=64

```
l2() { adb shell "cd $D && NNTR_L2_DIFF=1 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/$2 \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/$1_l2.log | grep -c '^\[L2-DIFF-MOE\] expert'; }
l2 B q40-qs4cx-wh; l2 C q40-qs4cx-wh-had
for v in B C; do grep -oE '^\[L2-DIFF-MOE\] expert=[0-9]+ M=([0-9]+) had=[01] snr=[-0-9.]+' $W/logs/${v}_l2.log | awk -F'snr=' -v v=$v '{s[NR]=$2} END{n=asort(s); c=0; for(i=1;i<=n;i++) if(s[i]<100) c++; printf "%s n=%d min=%.2f median=%.2f p10=%.2f count<100dB=%d\n", v, n, s[1], s[int((n+1)/2)], s[int(n/10)+1], c}'; done
```

`had=0` on every B line, `had=1` on every C line. Paste the two summary lines
and any `skipped`/`failed` line.

### 6. Profile (≈ 4 min) — one run each, A B C, G=64

```
prof() { adb shell "cd $2 && NNTR_HTP_PROFILE=2 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/$3 \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/$1_prof.log | grep '^\[HTP-PROFILE\]' | head -60; }
prof A $DA q40-qs4cx-wh; prof B $D q40-qs4cx-wh; prof C $D q40-qs4cx-wh-had
```

Paste the MoE rows; the FWHT lands inside the existing `requant` stage, so
read `requant` (min / mean µs) for the M=512 prefill call and the M=1 decode
calls, A vs B vs C.

### 7. Finish (3 min)

Fill the tables below, `git add docs/measurements/95-down-hadamard.md`,
commit with `-s` on `htp/95-down-hadamard`, push, set #95 to
`state:measured`. Keep `$W/logs/` (the supervisor reads it).

## Results (fill in)

### TPS (prompt 512, `NNTR_NUM_THREADS=8`)

| variant | G | run | prefill tok/s | decode tok/s | peak RSS | text == A? | text == D? |
|---|---|---|---|---|---|---|---|
| D | 64 | 1 | | | | — | — |
| D | 512 | 1 | | | | — | — |
| D | 1024 | 1 | | | | — | — |
| A | 64 | 1 | | | | — | |
| A | 64 | 2 | | | | | |
| A | 512 | 1 | | | | — | |
| A | 512 | 2 | | | | | |
| A | 1024 | 1 | | | | — | |
| A | 1024 | 2 | | | | | |
| B | 64 | 1 | | | | | |
| B | 64 | 2 | | | | | |
| B | 512 | 1 | | | | | |
| B | 512 | 2 | | | | | |
| B | 1024 | 1 | | | | | |
| B | 1024 | 2 | | | | | |
| C | 64 | 1 | | | | | |
| C | 64 | 2 | | | | | |
| C | 512 | 1 | | | | | |
| C | 512 | 2 | | | | | |
| C | 1024 | 1 | | | | | |
| C | 1024 | 2 | | | | | |
| A | 64 | 3 (drift) | | | | | |

"text == A/D": `y`, or `n @ <first differing token index>`. C is expected
to differ from A (different int4 weights); that is recorded, not a failure.
B ≠ A is a failure: stop and report.

### Accuracy

| variant | PPL | nll/token | requant SNR min / median / p10 (dB) | calls < 100 dB |
|---|---|---|---|---|
| D (CPU q40) | | | — | — |
| A | | | — | — |
| B | | | | |
| C | | | | |

### Profile — `requant` stage (µs)

| variant | prefill call (M=512) | decode call (M=1), min | decode call (M=1), mean |
|---|---|---|---|
| A | | | |
| B | | | |
| C | | | |

### Device gtests

| test | result |
|---|---|
| `HvxFwht.MatchesScalarBitExact` | |
| `HvxFwht.RejectsPartialBlock` | |
| `HmxMmU8I4Layer.MoeLayerHadamardMatchesTwoCallReference` | |
| `HmxMmU8I4Layer.MoeSetOptsEchoesKnownBits` | |

## Reading (for the supervisor)

* B vs A: text identical, PPL identical, TPS within ±5 % → the code change is
  inert when the flag is off. Anything else voids C.
* C vs A: the verdict. Accuracy: PPL(C) < PPL(A), and the SNR median of C
  above B. Cost: prefill and decode tok/s deltas, and the `requant` stage
  growth that explains them. The user sets the speed threshold from these
  numbers (issue #95: no gate fixed in advance).
* Numbers are not comparable to `htp_moe` BENCHMARK rows: this tree has no
  M=1 GEMV path and no DMA trace. Read only A/B/C/D against each other.

## Notes from the run

<battery / thermal at 0 and after step 3, config echo, provenance md5s,
anything unexpected>
