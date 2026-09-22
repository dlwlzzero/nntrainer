# Measurement 100: row h's cause and a DDR→VTCM feed shape for the M=1 GEMV, with the #101 ride-along (A = GEMV default vs A0)

Branch `htp/100-dma-chunk-list` (PR into `htp_moe`; code head
`a8642bd1`, this file on the same branch). Estimated device time:
**≈ 32 min**, plus 5–8 if the model has to be pushed.

Plan: `docs/plans/100-dma-chunk-list.md` §4 step 5. Issue #100 (tracker
#76; LEDGER ⑥ / ㉒, rules 11 and 19). Contract
`docs/plans/0001-htp-moe-decode-agent-system.md`. Ride-along cells from
`docs/measurements/101-ride-along.md` (PR #108, merged as `8e121dbe`).

## Why

The traced M=1 chunk list replays at 40 GB/s. DMA probe shape iii, with
the same 8 KiB × 64 @ 56 KiB geometry, reads 107, which is above the
DRAM peak. #100 must name the cause from 16 replay cells of one run, and
it must certify a descriptor shape that feeds the GEMV from VTCM at
≥ 37 GB/s while HVX reads VTCM. The denominator is the same-run `c_star`.
**What hangs on it:** if a shape passes, the ㉒ feed issue is filed with
that shape. It rebases onto PR #107, and its gate is M==1 `mm` ≤ 600 µs.
If no shape passes, ㉒'s feed half closes and #107 lands alone after a
lead sweep. The same sitting also carries PR #108's ride-along (A0 and
the fixed M==1 level-2 row).

## Variants (one app set, one skel, one gtest)

* **A**: `htp_moe` app with the GEMV default, env unset. Every A log must
  print `[HTP] moe m1 gemv: on (applied=0x1) source=default` exactly
  once. An A log without that line voids the sitting (#108's inverted
  rule). Run first.
* **A0**: A's binaries with `NNTR_MOE_HTP_M1_GEMV=0` (the HMX path), G = 64
  only. This is #101's cell R3 and R4.
* #100 adds **no variant**. Its cells are a gtest (`unittest_hvx_dma_probe`)
  run once under A's skel. No model binary changes: `git diff --stat
  db7c6eb2 a8642bd1 -- nntrainer/ Applications/ test/htp/nntr_hvx.idl` is
  empty. The skel carries the new replay entry, but no model call reaches
  that code.
* **#99's replay check is not in this sitting.** Its fix is not merged, so
  the 11 old `DMA_REPLAY` lines still fail their `res[6]` check, as they
  did in #94 (see "Reading the gtest").

Accuracy: the text of A0 must equal A at G = 64. Run 1 must equal run 2
for A at every G. `NNTR_L2_DIFF` and text-vs-CPU `q40` are n/a for
`QS4CX_WH`.

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

Staged under **`W=/local/mnt/workspace/htp_moe/100/`**. `$W/md5.txt`
reproduces this table. `(cd $W && md5sum -c md5.txt)` must pass on every
line. Rule 22: the staging path is a convenience, and the recipe below
rebuilds the set from the commit.

| file (staged as) | md5 | built from / with |
|---|---|---|
| `$W/libnntr_hvx_skel.so` (→ device `libnntr_hvx_skel.so`; A, A0, gtest) | `25f8518986a5821a91218fa715092fde` | `test/htp/build.sh` @ `a8642bd1`, `HEXKL_SDK_VER=6.4.0.1`; `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK (46 runtime imports)`. Skel builds are not byte-reproducible (LEDGER rule 14), so a rebuild gives another md5; record the one that ran |
| `$W/A/nntrainer_causallm` | `9ed4e439f30dba04dfa9638e02cbc788` | `build_android.sh --htp`, then the prefix fix, then `--htp --cache`, on this branch (app sources = `htp_moe` @ `db7c6eb2` = `8e121dbe`'s code) `jni/libs/arm64-v8a/` |
| `$W/A/libcausallm_core.so` | `7c84f335615ca48c7dbf574fe9fafe75` | same, `jni/libs/arm64-v8a/` |
| `$W/A/libnntrainer.so` | `dcd57fc1b598ec420725ee1a5c6429d6` | same, `jni/obj/local/arm64-v8a/`; `readelf -d` lists `libsdkl.so` and `libcdsprpc.so`; `strings … \| grep -c 'source=%s'` = 2 |
| `$W/A/libccapi-nntrainer.so` | `c57dd47f6f835b9499dc5eb07dafb3cc` | same, `jni/obj/local/arm64-v8a/` |
| `$W/gtest/unittest_hvx_dma_probe` | `0f6d53dfa418a83b3591608c213c722e` | `ndk-build … unittest_hvx_dma_probe` @ `a8642bd1` (`test/jni/obj/local/arm64-v8a/`) |
| `$W/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot `toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/` (== #77/#88/#101) |
| `$W/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` (== #77/#88/#101) |
| `$W/prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | == #77/#88/#101 (`docs/measurements/77-prompt512.txt`) |
| model `q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (4316133120 B) | `7b7867fab51845664c0050c0a837073e` | #78, `--moe_dtype QS4CX_WH` (NPU model, the only model in this sitting) |
| `q40-qs4cx-wh/tokenizer.json` | `7b8067a580173d3eb1697afae3b456f5` | HF |

The app's sources are the same as #101's staged A (`git diff 8018c88a
db7c6eb2 -- nntrainer Applications` is empty). The md5s differ because
this is another build path (rule 14). Never staged or pushed:
`builddir/…/libcdsprpc.so` (rule 5).

### Rebuild recipe (rule 22; any workstation)

```
git fetch origin && git checkout a8642bd1              # or the branch head
git submodule update --init --depth 1                  # a fresh worktree has empty subprojects/ (LEDGER §3a)
cp <another checkout>/Applications/CausalLM/lib/libtokenizers_android_c.a Applications/CausalLM/lib/   # or build_tokenizer_android.sh
source tools/htp/env.sh
export HEXKL_ROOT=<your HexKL package root, holds lib/6.4.0.1/>   HEXKL_SDK_VER=6.4.0.1   ANDROID_NDK=<NDK r30>
./test/htp/build.sh                                    # must print UNDEFINED SYMBOLS OK (46 runtime imports)
(cd Applications/CausalLM && ./build_android.sh --htp) # fresh builddir: ends in a /usr/local permission error (googletest install); then
(cd builddir && meson configure -Dprefix=$PWD/android_build_result && ninja install)
(cd Applications/CausalLM && ./build_android.sh --htp --cache)
ln -sfn $PWD/subprojects/googletest/googletest test/jni/googletest
(cd test/jni && $ANDROID_NDK/ndk-build NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk \
   APP_BUILD_SCRIPT=./Android.mk NNTRAINER_ROOT=$PWD/../.. HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT unittest_hvx_dma_probe -j8)
cp $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so <stage>/   # b1586b9b…
cp $HEXKL_ROOT/lib/6.4.0.1/armv8_android26/libsdkl.so <stage>/                                                           # 0ad4e22a…
```

Take the four A files from `Applications/CausalLM/jni/libs/arm64-v8a/`
(`nntrainer_causallm`, `libcausallm_core.so`) and `jni/obj/local/arm64-v8a/`
(`libnntrainer.so`, `libccapi-nntrainer.so`). Env var names in this
sitting: `NNTR_MOE_HTP_M1_GEMV` (unset = on, `0` = off),
`NNTR_HTP_PROFILE`, `NNTR_NUM_THREADS=8`.

Workstation sanity before pushing:

```
W=/local/mnt/workspace/htp_moe/100; M=/local/mnt/workspace/models/lfm2.5-8b-a1b
(cd $W && md5sum -c md5.txt)                                          # every line OK
strings $W/A/nntrainer_causallm | grep -c 'per-layer-type totals'      # 0 (not a profile binary)
strings $W/A/libnntrainer.so | grep -c 'source=%s'                     # 2 (the GEMV default is in)
find $W -name 'libcdsprpc*' | wc -l                                   # 0
```

## Steps (workstation, phone on USB)

Shell setup once:

```
cd <your checkout> && git fetch && git checkout htp/100-dma-chunk-list && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/100; M=/local/mnt/workspace/models/lfm2.5-8b-a1b; mkdir -p $W/logs
D=/data/local/tmp/nntrainer/causallm
therm() { adb shell dumpsys battery | grep -E 'level|temperature'; adb shell cat /sys/class/thermal/thermal_zone0/temp; }
```

### 0. Device state (2 min)

```
adb devices                     # exactly one device; write its serial under Notes (any S25 Ultra, contract §4.2)
therm                           # checkpoint 0: battery %, temperature (tenths of °C), thermal_zone0 (m°C)
adb shell df -h /data
```

Screen off, charger in.

### 1. Install (5 min; + 5–8 if the model must be pushed)

```
adb shell ls -l $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin   # 4316133120
adb shell md5sum $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin  # 7b7867fab51845664c0050c0a837073e, else:
#   adb shell mkdir -p $D/models && adb push $M/q40-qs4cx-wh $D/models/q40-qs4cx-wh
adb shell mkdir -p $D/models
adb push $W/libnntr_hvx_skel.so $W/libc++_shared.so $W/libsdkl.so $W/prompt512.txt $D/
adb push $W/A/nntrainer_causallm $W/A/libcausallm_core.so $W/A/libnntrainer.so $W/A/libccapi-nntrainer.so $D/
adb push $W/gtest/unittest_hvx_dma_probe $D/
adb shell "chmod 755 $D/nntrainer_causallm $D/unittest_hvx_dma_probe"
```

Config edits, applied every time (greedy decoding, exact token count, and
the WH dir must carry `moe_engine: htp`):

```
adb shell "cd $D/models/q40-qs4cx-wh && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|moe_engine|moe_htp_layers' nntr_config.json"
```

Expected echo: `do_sample": false`, `bad_word_ids": [124900]`,
`init_seq_len": 512`, `moe_engine": "htp"`, `moe_htp_layers": ""`.

Provenance (paste under Notes; must equal the table):

```
adb shell "md5sum $D/libnntr_hvx_skel.so $D/nntrainer_causallm $D/libnntrainer.so $D/libcausallm_core.so $D/libccapi-nntrainer.so $D/unittest_hvx_dma_probe $D/prompt512.txt $D/models/q40-qs4cx-wh/tokenizer.json"
```

### 2. A: 6 cells (≈ 14 min), env unset

```
run() { # run <variant> <G> <run#> [extra env]
  adb shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' models/q40-qs4cx-wh/nntr_config.json && \
    grep num_to_generate models/q40-qs4cx-wh/nntr_config.json && \
    $4 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/$1_G$2_r$3.log | grep -E '^(prefill|generation|total|peak memory)|moe m1 gemv'
}
therm                                                          # checkpoint 1
run A 64 1;  run A 64 2
run A 512 1; run A 512 2
sleep 60; run A 1024 1; sleep 60; run A 1024 2
therm                                                          # checkpoint 2
grep -c 'moe m1 gemv: on (applied=0x1) source=default' $W/logs/A_G*_r*.log      # R1: 1 per file
```

Expected in every A log: `prefill: 512 tokens, … TPS`, `generation: <G>
tokens, … TPS`, `total: … ms`, `peak memory: … KB`, exactly one
`[HTP] moe m1 gemv: on (applied=0x1) source=default`, and **no**
`[HTP-PROFILE]` block. A count of `0` for R1 voids that A as the GEMV
reference. `prefill: N` with N ≠ 512 voids the prefill column, and
`generation: N` with N ≠ G is a failed config edit.

### 3. #100 gtest ride-along (≈ 2 min), A's skel, right after checkpoint 2

```
adb shell "cd $D && md5sum libnntr_hvx_skel.so unittest_hvx_dma_probe && cat /sys/class/thermal/thermal_zone0/temp && \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='*DmaProbeShapes*:*MoeChunkReplay*'" 2>&1 \
  | tee $W/logs/G_replay.log | grep -E 'md5|^[0-9]+$|^DMA_REPLAY_NOTE|^DMA_REPLAY_TRACE dsp_us|^DMA_REPLAY |^DMA_REPLAY_X|^DMA_PROBE|OK \]|PASSED|FAILED'
therm                                                          # checkpoint 3
```

Read-out (paste the output under Results):

```
L=$W/logs/G_replay.log
grep -c '^DMA_PROBE .*checksum_ok=y' $L                          # 16
grep -c '^DMA_REPLAY_X .*checksum_ok=y' $L                       # 16
grep -c '^DMA_REPLAY workers.*checksum_ok=n' $L                  # 11 (#99, expected; see below)
grep -E 'skipped|stale_skel' $L                                  # nothing
grep -E '^DMA_PROBE shape=iii workers=1 |^DMA_REPLAY_X' $L | sed -E 's/ (bytes_per_call|regions|load_units)=[^ ]*//g'
awk '/^DMA_PROBE shape=iii workers=1 /{for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="gbs")p=a[2]}}
     /^DMA_REPLAY_X/{n="";g="";for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]=="name")n=a[2];if(a[1]=="gbs")g=a[2]};r[n]=g}
     END{c=r["c_star"];
       printf "H0: c_star/probe_iii = %.1f/%.1f = %.2f (fires < 0.80)\n",c,p,c/p;
       printf "H1: iii_dmstart/probe_iii = %.2f (>= 0.80), iii_chain/iii_dmstart = %.2f (<= 0.60), iii_link1/iii_dmstart = %.2f\n",r["iii_dmstart"]/p,r["iii_chain"]/r["iii_dmstart"],r["iii_link1"]/r["iii_dmstart"];
       printf "list: iii_chain/iii_dmstart >= 0.80 and traced/iii_chain = %.2f (<= 0.60); traced/c_star = %.2f\n",r["traced"]/r["iii_chain"],r["traced"]/c;
       printf "  H3 traced_nowait/traced = %.2f (>= 1.25)  H2 traced_gu/traced_dn = %.2f (>= 1.5 either way)  H4 traced_f/traced = %.2f (15%%)  H5 iii_strided/iii_chain = %.2f (15%%)\n",r["traced_nowait"]/r["traced"],r["traced_gu"]/r["traced_dn"],r["traced_f"]/r["traced"],r["iii_strided"]/r["iii_chain"];
       split("f1 f2 f3",f," ");for(k=1;k<=3;k++)printf "feed %s: load0/c_star = %.2f (>= 0.80), load2 = %.1f GB/s (>= 37.0)\n",f[k],r[f[k]]/c,r[f[k]"_load"]}' $L
```

**Reading the gtest.**

* `DmaProbeShapes` must print `[       OK ]` with 16 `DMA_PROBE` lines,
  all `checksum_ok=y`. Its check is the same as before: the fill keeps
  0xA5 at every 64-byte sample.
* `MoeChunkReplay` is **expected to print `[  FAILED  ]`**. The failures
  must all come from `unittest_hvx_dma_probe.cpp:633` (`res[6]` 1467840
  vs `want_sum` 9461760), one for each of the 11 old `DMA_REPLAY` lines.
  That is #99, unchanged by this PR. A failure at `:715` (an X line's
  `res[12]`) or at `:692` (`skipped err=`) is a #100 result. Read it as
  follows:
  * `skipped err=`, or `stale_skel_or_nothing_landed` on every X line,
    means the skel on the device is not this branch's. Re-push
    `$W/libnntr_hvx_skel.so`, check the md5 line, and redo step 3.
  * `checksum_ok=n` on some X lines only means one of two things.
    * If the lines are among `traced`, `traced_f`, `traced_nowait`,
      `traced_gu` and `iii_chain`: these are the only cells whose writes
      overlap while in flight (the packed gate/up chunks of #99, or a slot
      reused 8 pushes on). Record which ones, because the result says the
      engine lets overlapping writes land out of list order. Their timing
      is still read. It is not a dropped transfer.
    * If `c_star`, `iii_dmstart`, `iii_link1`, `iii_strided` or any `f`
      line fails: those cells wait or drain before they rewrite a
      destination. File the line, and do not read its rate.
* In the same log, the step-0 line `DMA_REPLAY_TRACE dsp_us=… desc=46 …
  plan_shape_ok=y` must be present. It runs the HMX path, because the
  gtest does not call `moe_set_opts`.

### 4. A0: opt-out TPS, G = 64, one run (1.5 min)

```
run A0 64 1 NNTR_MOE_HTP_M1_GEMV=0
```

Expected: `[HTP] moe m1 gemv: off (applied=0x0) source=env`. The text
must equal `A_G64_r1`
(`diff <(sed -n '/^=====/q;p' $W/logs/A0_G64_r1.log) <(sed -n '/^=====/q;p' $W/logs/A_G64_r1.log)`).
Decode is expected ≈ 6 % **below** A. A 0 % delta means the default did
not take, and that voids A as the GEMV reference. Prefill: A must be
within −5 % of A0.

### 5. Profiles (≈ 4 min; never read for tok/s), G = 64, one run each

```
adb shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/q40-qs4cx-wh/nntr_config.json"
prof() { # prof <name> <env...>
  adb shell "cd $D && $2 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/prof_$1.log | grep -E '^\[HTP-PROFILE\]' | grep -E 'level=|K=2048 N=2048|staging:|weight DMA' | cut -c1-400
}
prof A_L2  "NNTR_HTP_PROFILE=2"                                   # R2
prof A0_L2 "NNTR_HTP_PROFILE=2 NNTR_MOE_HTP_M1_GEMV=0"            # R4
```

Expected, from #101's R2 and R4:

* **A_L2** header `qos_mode=2`.
  * M==1 row: `blocks=0 m1_gemv=1408/1408`, `rest` ≥ 0 and ≤ 5 % of
    `host`, `swiglu / mm` ≈ 5.7.
  * Next line: `weight DMA: n/a (direct arena read inside mm, no ring;
    swiglu = lane-time, N.NN lanes busy over mm)` with N.NN ≈ 5.7, and a
    `staging:` line.
  * M>1 row: `m1_gemv=0/23`.
* **A0_L2** M==1 row: `blocks=5632 m1_gemv=0/1408`, followed by a normal
  `weight DMA:` line.
* **Transport question (supervisor).** Report M==1 `transport=` as
  `transport GEMV / HMX = x / y µs/call (Δ)` from A_L2 minus A0_L2, with
  both `staging:` lines. Expect equal `act`/`out` classes (65536 B).

### 6. A re-run (2 min): thermal check, then the text checks

```
run A 64 3
for g in 64 512 1024; do printf 'A G=%s r1=r2: ' $g; diff <(sed -n '/^=====/q;p' $W/logs/A_G${g}_r1.log) <(sed -n '/^=====/q;p' $W/logs/A_G${g}_r2.log) >/dev/null && echo same || echo DIFFERENT; done
printf 'A0 vs A G=64: '; diff <(sed -n '/^=====/q;p' $W/logs/A0_G64_r1.log) <(sed -n '/^=====/q;p' $W/logs/A_G64_r1.log) >/dev/null && echo same || echo DIFFERENT
printf 'A r3 vs r1 G=64: '; diff <(sed -n '/^=====/q;p' $W/logs/A_G64_r3.log) <(sed -n '/^=====/q;p' $W/logs/A_G64_r1.log) >/dev/null && echo same || echo DIFFERENT
grep -l 'HTP-PROFILE' $W/logs/A*_G*_r*.log                             # nothing
therm                                                                   # checkpoint 4
```

Then fill the tables below, commit this file on the branch, push, and
set #100 to `state:measured`.

## Results (fill in)

Reference (other sittings, so context only, not a verdict; rules 9 and
20):

* NPU now: 27.74 / 26.88 / 24.83 decode tok/s at G 64 / 512 / 1024.
  This is #105 A, the GEMV-on path; the #88 B GEMV-off path read
  24.50 / 23.80 / 23.52.
* NPU prefill means: 498.5 / 453.5 / 380.7.
* CPU now: 52.43 / 49.22 / 48.31.
* Goal: ≥ 50 decode tok/s. Prefill gate: ≥ −5 % of this sitting's A.

| variant | gen | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS (KB) | text = A run 1? | `source=default` line | libnntrainer.so md5 (device) |
|---|---|---|---|---|---|---|---|---|---|
| A | 64 | 1 | | | n/a (#89) | | reference | | |
| A | 64 | 2 | | | n/a | | | | |
| A | 512 | 1 | | | n/a | | reference | | |
| A | 512 | 2 | | | n/a | | | | |
| A | 1024 | 1 | | | n/a | | reference | | |
| A | 1024 | 2 | | | n/a | | | | |
| A0 | 64 | 1 | | | n/a | | | `off … source=env` | |
| A | 64 | 3 (after the rest) | | | n/a | | | | |

Profiles (G = 64, µs/call):

| profile | qos_mode | M==1 row (`blocks`, `m1_gemv`, host / dsp / transport / mm / swiglu / rest) | `weight DMA:` line | `staging:` | M>1 row (`m1_gemv`, host / dsp / transport) |
|---|---|---|---|---|---|
| A_L2 (R2) | | | | | |
| A0_L2 (R4) | | | | | |

Transport GEMV / HMX = … / … µs/call (Δ …).

#100 gtest (`G_replay.log`, `thermal_zone0` = … m°C before it):

| line | gbs | us_per_call | busy_us lo..hi | depth_max | wait_us | checksum_ok |
|---|---|---|---|---|---|---|
| `DMA_PROBE shape=iii workers=1` | | (us) | — | — | — | |
| traced | | | | | | |
| traced_f | | | | | | |
| traced_nowait | | | | | | |
| traced_gu | | | | | | |
| traced_dn | | | | | | |
| iii_chain | | | | | | |
| iii_dmstart | | | | | | |
| iii_link1 | | | | | | |
| iii_strided | | | | | | |
| **c_star** | | | | | | |
| f1 (load 0) | | | | | | |
| f1_load (load 2) | | | | | | |
| f2 (load 0) | | | | | | |
| f2_load (load 2) | | | | | | |
| f3 (load 0) | | | | | | |
| f3_load (load 2) | | | | | | |

Paste the `awk` read-out (the §3.3 ratios) here, together with the
`DmaProbeShapes` / `MoeChunkReplay` result lines and the failure
locations (`:633` × 11 expected).

## Notes from the run

<serial, battery %, `thermal_zone0` at checkpoints 0–4, the provenance
md5 line, anything stale (FARF/AEE errors, `0x8000…`), deviations>
