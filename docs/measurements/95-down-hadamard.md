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

## Results (filled 2026-09-22, 16:10–17:05 KST)

Unit **`R3CY205ZMND`** (SM-S938N, S25 Ultra), through the ADF SSH adb
bridge (`adf.sraisys.com`), not local USB: the bridge accepts only
`shell`/`push`/`pull`, so every device-side command below ran as a pushed
`/system/bin/sh` script (`d95_cfg.sh`, `d95_run.sh`, `d95_runenv.sh`,
`d95_gtest.sh`) invoked as `shell sh <path>`. Battery 100 %, charger in,
screen off. `thermal_zone0` **29.4 °C** before D, **59.3 °C** after A's six
cells, **60.8 °C** after B's, **58.1 °C** after C's + the drift re-run.
`/data` 36 G free.

### TPS (prompt 512, `NNTR_NUM_THREADS=8`)

| variant | G | run | prefill tok/s | decode tok/s | peak RSS | text == A? | text == D? |
|---|---|---|---|---|---|---|---|
| D | 64 | 1 | 315.077 | 53.601 | 5896980 | — | — |
| D | 512 | 1 | 312.005 | 50.236 | 5886856 | — | — |
| D | 1024 | 1 | 282.249 | 49.495 | 5127552 | — | — |
| A | 64 | 1 | 524.053 | 25.157 | 5326272 | — | `n @ gen word 43` |
| A | 64 | 2 | 487.619 | 25.000 | 5326320 | `y` | `n @ 43` |
| A | 512 | 1 | 480.751 | 24.460 | 5329308 | — | `n @ 43` |
| A | 512 | 2 | 437.607 | 24.345 | 5147280 | `y` | `n @ 43` |
| A | 1024 | 1 | 436.860 | 24.053 | 5561692 | — | `n @ 43` |
| A | 1024 | 2 | 433.898 | 23.603 | 5268652 | `y` | `n @ 43` |
| B | 64 | 1 | 484.390 | 24.951 | 5316188 | **`y`** | `n @ 43` |
| B | 64 | 2 | 474.074 | 24.682 | 5324960 | **`y`** | `n @ 43` |
| B | 512 | 1 | 433.164 | 24.091 | 5329284 | **`y`** | `n @ 43` |
| B | 512 | 2 | 403.785 | 24.148 | 5317764 | **`y`** | `n @ 43` |
| B | 1024 | 1 | 403.785 | 23.730 | 5320608 | **`y`** | `n @ 43` |
| B | 1024 | 2 | 403.467 | 23.775 | 4732020 | **`y`** | `n @ 43` |
| C | 64 | 1 | 428.094 | 24.587 | 5323400 | `n @ gen word 52` | `n @ 43` |
| C | 64 | 2 | 401.254 | 24.390 | 4749252 | `n @ 52` | `n @ 43` |
| C | 512 | 1 | 396.285 | 24.002 | 4973712 | `n @ 52` | `n @ 43` |
| C | 512 | 2 | 398.444 | 23.290 | 4945840 | `n @ 52` | `n @ 43` |
| C | 1024 | 1 | 400.000 | 23.492 | 5270628 | `n @ 52` | `n @ 43` |
| C | 1024 | 2 | 395.062 | 23.573 | 5334012 | `n @ 52` | `n @ 43` |
| A | 64 | 3 (drift) | 404.743 | 24.502 | 5330124 | `y` (= A run 1) | `n @ 43` |

`[HTP-MOE] opts applied=0` printed once in every B log and `applied=2` in
every C log; A and D print no `opts` line — the flag is exactly where the
dtype puts it. Every run printed `prefill: 512 tokens` and
`generation: <G> tokens`; run 1 = run 2 in all nine two-run cells; no
`AEE_*`, no `0x8000…`, no `older than this binary`.

**Reading the text column.** The first pass compared the whole pre-`=====`
block and reported B ≠ A; the only difference was B's extra
`[HTP-MOE] opts applied=0` line. With the `[HTP*` lines and the model-path
echo stripped, **B is byte-identical to A at every G** — the gate holds.
C diverges from A at **generated word 52** and both NPU variants diverge
from the CPU reference D at **generated word 43**, identically at all three
G (the divergence point is a property of the weights, not of G).

**Decode means and the drift control** (two runs each; A's re-run at the
end is the drift control):

| G | A | B | B vs A | C | C vs A | C vs A run 3 (drift-matched, G=64) |
|---|---|---|---|---|---|---|
| 64 | 25.079 | 24.817 | −1.0 % | 24.489 | −2.4 % | **+0.3 %** (24.587 vs 24.502) |
| 512 | 24.403 | 24.119 | −1.2 % | 23.646 | −3.1 % | — |
| 1024 | 23.828 | 23.752 | −0.3 % | 23.533 | −1.2 % | — |

A's own G=64 cell moved 25.157 → 24.502 (−2.6 %) between the start and the
end of the sitting as `thermal_zone0` went 29 → 58 °C, so **every B and C
decode delta is inside this sitting's own drift**: the FWHT is free on the
decode path.

**Prefill is dominated by the same drift and cannot be read as a code
delta.** Six-cell means: A 466.8, B 433.8 (−7.1 %), C 403.2 (−13.6 %) —
but A's drift re-run at the end of the sitting reads **404.7**, i.e. the
same as C's cells, which ran third. Order, not code: A ran cool, B warm, C
hot. The profile's M>1 `dsp=` column (below) is the honest prefill
comparison, and it moves +2.7 % from B to C.

### Accuracy

| variant | PPL | nll/token | requant SNR min / p10 / median (dB) | calls < 100 dB |
|---|---|---|---|---|
| D (CPU `q40`) | 109.759 | 4.69829 | — | — |
| A | 115.095 | 4.74576 | — | — |
| B | **115.095** | **4.74576** | 25.01 / 31.27 / 34.68 | 6214 of 6246 |
| C | **100.497** | **4.61013** | **39.68 / 41.67 / 42.67** | 6242 of 6274 |

`prompt tokens=511` in all four. **B's PPL equals A's to the last digit**
— the code change is inert with the flag off, as the gate requires.

**C is the measurement's answer, and it is positive on both accuracy
metrics:**

* **PPL 115.095 → 100.497, −12.7 %.** The rotated model is not only better
  than the unrotated NPU model, it is **8.4 % better than the CPU `q40`
  reference** (109.759), which no NPU variant has ever beaten in this
  project.
* **Per-call requant SNR rises on every statistic**: median **+7.99 dB**
  (34.68 → 42.67), p10 **+10.40 dB** (31.27 → 41.67), min **+14.67 dB**
  (25.01 → 39.68), over ≈ 6.2 k expert calls per run (`had=0` on all 6246
  B lines, `had=1` on all 6274 C lines — the flag reached every call). The
  `999.00 dB` tail is the degenerate-shape artefact the plan already
  documented (32 lines of each run); it is excluded from min/p10/median by
  construction since those come from the sorted low end.

### Profile — `requant` stage (µs/call, `NNTR_HTP_PROFILE=2`, G=64, `qos_mode=2`)

| variant | prefill call (M>1, 23 calls) | decode call (M==1, 1408 calls) | M>1 `dsp=` | M==1 `dsp=` |
|---|---|---|---|---|
| A | 1097.3 | 41.9 | 15915.6 | 1333.5 |
| B | 1121.2 | 42.0 | 16061.7 | 1334.3 |
| C | **1567.4** | **46.6** | **16491.0** | **1338.5** |

The row prints one mean per stage on this tree, so the "min" column the
handoff asked for does not exist; the means are above. The FWHT's cost is
therefore **+446.2 µs (+39.8 %) in the prefill call's `requant`** and
**+4.6 µs (+11.0 %) in the decode call's**, which is **+2.7 %** and
**+0.3 %** of the respective whole-call `dsp=` — invisible in decode tok/s
and, once the thermal order is accounted for, in prefill as well.

### Device gtests

| test | result |
|---|---|
| `HvxFwht.MatchesScalarBitExact` | **FAILED — the documented subnormal exception** |
| `HvxFwht.RejectsPartialBlock` | `OK` |
| `HmxMmU8I4Layer.MoeLayerHadamardMatchesTwoCallReference` | **FAILED on its SNR criterion; bit-exactness passed** |
| `HmxMmU8I4Layer.MoeSetOptsEchoesKnownBits` | `OK` |

* `MatchesScalarBitExact`: `FWHT_FIELD bad_gated=2 bad_subnormal_row=2
  bad_overflow_row=127 of 8192`, first mismatch `i=7681 (row 30)
  x=-0x1.5c73p-130 dsp=0x1.7f4b4p-127 ref=0x0p+0`. `bad_gated ==
  bad_subnormal_row == 2`: **every** bad element is in the subnormal row,
  which is exactly the case §2 of this handoff says to continue past
  (`fwht_det_ftz`'s flushed-zero sign). No E2E consequence — real SwiGLU
  outputs are not subnormal.
* `MoeLayerHadamardMatchesTwoCallReference`: `bad_elems=0 of 409600
  max_ulp=0` — the DSP's rotated path matches the two-call host reference
  **bit-exactly**, so the kernel, the fold axis and the flag plumbing are
  all correct. What failed is the test's accuracy assertion:
  `snr_db_folded=17.5064` vs `snr_db_unfolded=23.3210`, threshold
  `snr_off − 3.0 = 20.321` — on that one synthetic shape the rotation is
  5.8 dB **worse**.

**This sitting ran C in spite of that failure** (§2 says stop; the user
decided to continue on 2026-09-22 after seeing that the same test proved
the kernel bit-exact and that the failing assertion is an accuracy verdict,
not a build check). That decision is what produced the result: **the
synthetic single-shape gtest and the full model disagree, and they
disagree in opposite directions** — −5.8 dB on the gtest's shape, +8.0 dB
median and −12.7 % PPL on the real model over 6.2 k calls. The gtest's
assertion is measuring something real about its own activation
distribution, not about the model, so **the gtest's threshold is the thing
that is wrong here, not the rotation**. Hypothesis for the planner (not
measured): the gtest's synthetic down-input is already well-conditioned,
so the rotation only adds f32 rounding, while real SwiGLU output carries
the column-wise outliers the rotation exists to spread. Deciding that
needs the gtest's input distribution replaced with a captured real
`gate_off` buffer — a follow-up, not a blocker.

### Verdict

**The rotation works and is effectively free.** PPL −12.7 % (and below the
CPU reference), requant SNR +8.0 dB median / +14.7 dB min, decode cost
inside the sitting's own ±2.6 % thermal drift, prefill cost +2.7 % of the
M>1 `dsp=` column. The issue's "carry `QS4CX_WH_HAD` over to `htp_moe`
(plus the M=1 GEMV insert site)" branch is the one the numbers support.
Two items travel with it: the subnormal-row sign in `fwht_det_ftz`, and
the `MoeLayerHadamardMatchesTwoCallReference` threshold, which as written
fails on a model that measurably improves.

## Notes from the run

* **Deviation 1 — everything was rebuilt on this workstation.** Nothing was
  staged under `/local/mnt/workspace/htp_hadamard/95/` when the sitting
  started (the directory did not exist), the worktree the Steps name
  (`/home/j2z0-lee/nntrainer`) does not exist on this box, and
  `$M/q40-qs4cx-wh-had` had never been converted. Built here instead:

  | staged file | md5 that ran | built from |
  |---|---|---|
  | `$W/libnntr_hvx_skel.so` | `3f31be9c36d81c494d5c3bf35c8a4bd8` (176,592 B) | `test/htp/build.sh` @ `6c89a912`; 44 runtime imports, no project symbol undefined (#97 guard run by hand — this tree has no guard in `build.sh`); exports `hvx_fwht_rows_f32` |
  | `$W/libnntr_hvx_skel.A.so` | `077a50a3985685276deb505036425791` (171,952 B) | same @ `63167235`; 44 imports; no `hvx_fwht_rows_f32` |
  | `$W/tps/nntrainer_causallm` | `9e15a15b533cd30534ceefd4c9e1b839` | `Applications/CausalLM/build_android.sh --htp` then `--htp --cache` @ `6c89a912`; profile-strings count 0 |
  | `$W/tps/libcausallm_core.so` | `f64274aa7c6c12f03a4b98d6e0d0ad2d` | same |
  | `$W/tps/libnntrainer.so` | `ebd2d7822df3aee63ec5227c0086634e` | same; `readelf -d` lists `libsdkl.so` + `libcdsprpc.so` |
  | `$W/tps/libccapi-nntrainer.so` | `25e9174aa616c7992de3f3b1f5234da9` | same |
  | `$W/tpsA/nntrainer_causallm` | `16ae1ffd5d2752cf989bbd508a0b8afe` | same recipe @ `63167235`; profile-strings count 0 |
  | `$W/tpsA/libcausallm_core.so` | `aed5af6b2c7df40d5abbd5201273c1db` | same |
  | `$W/tpsA/libnntrainer.so` | `8e94e8b1b0c28449641ac9f7c91235c0` | same; both NEEDED entries present |
  | `$W/tpsA/libccapi-nntrainer.so` | `65c3dfc621d03a98f6234f0e1bec03f0` | same |
  | `$W/gtest/unittest_hvx_mm_u8i4` | `1a609110b0e237af3d696b3807360b65` | `ndk-build` @ `6c89a912` |
  | `$W/gtest/unittest_hvx_softmax` | `7c595b5284bf71056192dc48dae67c8d` | same |
  | `$W/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | == the table (NDK r30) |
  | `$W/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | == the table (HexKL `lib/6.4.0.1/armv8_android26`, `HEXKL_ROOT=~/Downloads/hexkl_addon`) |
  | `$W/prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | == the table |
  | `$M/q40-qs4cx-wh-had/nntr_lfm2_8b_a1b_q40_arm.bin` | **`a2829cd9d33209669d03b1613b0f5a02`** — **equal to the Artifacts table's prediction**, and the device copy re-hashes to the same value; size **4316133120 B**, exactly `q40-qs4cx-wh`'s | `nntr_quantize_stream` @ `6c89a912` (host `build/`), `$M/fp32 --fc_dtype Q4_0 --embd_dtype Q4_0 --lmhead_dtype Q4_0 --moe_dtype QS4CX_WH_HAD --isa ARM`; converter log: 24/24 layers, source 32302 MiB → output 4116 MiB (12.7 %) |
  | `$M/q40-qs4cx-wh/…` , `$M/q40/…` | `7b7867fab51845664c0050c0a837073e`, `d28f55c5bd7adeb8bf73b02de582eb88` | == the table; already on the device from #94 |

  The staged md5s differ from the Artifacts table's predictions (different
  build path, LEDGER rule 14); the device-side `md5sum` of every pushed
  file was checked against the staged set and matched line for line before
  any run. `$W/md5.txt` regenerated from what ran. No `libcdsprpc.so`
  anywhere under `$W`.
* **The rotated model is the down tensors and nothing else**, verified
  byte-wise against `q40-qs4cx-wh` rather than assumed:
  `diff_bytes = 1,277,247,516` of 4,316,133,120, first difference at offset
  225,833,600, last at 4,316,124,927. 704 `down` tensors
  (22 layers × 32 experts) × 2048 × 1792 int4 = 1.835 MB each = 1.277 GB —
  the Artifacts table's "704 regions of ≈ 1.81 MB" to within rounding. Every
  other byte of the file is identical, so `gate_up`, attention, conv,
  embedding and lm_head are untouched.
* **Config.** The converter writes neither `moe_engine` nor
  `moe_htp_layers`, and points `tokenizer_file` at `models/q40/`. The had
  dir's `nntr_config.json` was therefore rewritten before the sitting to
  `"moe_engine": "htp"`, `"moe_htp_layers": ""`, `"bad_word_ids":
  [124900]`, its own `tokenizer.json` path, and pushed; `do_sample: false`
  and `bad_word_ids` were applied to all three dirs. Echo confirmed
  `moe_layer_dtype` `Q4_0` / `QS4CX_WH` / `QS4CX_WH_HAD` and
  `init_seq_len: 512` in the three dirs.
* **Order of the sitting** (one continuous session, no reboot, no file
  swapped mid-variant — A/D live in `$DA`, B/C in `$D`, `$DA/models` is a
  symlink to `$D/models`): gtests → D → A → B → C → A run 3 → PPL ×4 →
  `NNTR_L2_DIFF` ×2 → profiles ×3.
* **Not run:** the `htp_moe`-side comparison (this tree is upstream #4327 +
  tooling: it already carries the size-class staging and the 5 ms poll that
  #88 measured, which is why A's decode reads 25 tok/s here against
  `htp_moe` A's 18.7 — the two sets of numbers are not comparable, as the
  handoff's Reading section says).
* Logs: `$W/logs/{D,A,B,C}_G<G>_r<n>.log` (21), `{A,B,C}_prof.log`,
  `{B,C}_l2.log`, `{D,A,B,C}_ppl.log`, `G_gtests.log`; build logs
  `$W/build_*.log`, `$W/convert.log`, `$W/model_diff.log`.
