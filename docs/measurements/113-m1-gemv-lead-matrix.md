# Measurement 113: the M=1 GEMV's (row loop × l2fetch lead) matrix, and the lead swept past 192 KB

Branch `htp/113-m1-gemv-lead-matrix` (code head `5be9c4ba`, this file on
the same branch) — estimated device time: **≈ 45–50 min** (≈ 35 min if the
short form in step 4 is taken)

Plan: `docs/plans/113-m1-gemv-lead-matrix.md` §4 step 5. Issue #113
(tracker #76; LEDGER ㉒ cycle 11, rules 26–30). Contract
`docs/plans/0001-htp-moe-decode-agent-system.md`. This branch **absorbs PR
#107** (`htp/105-m1-gemv-compute`): its one-row loop, `gemv_native_check.c`
and `MoeM1GemvFeedVsCompute` are rebased onto `htp_moe` here, so the
sitting also decides #107's fate.

Only the NPU model `q40-qs4cx-wh` runs here; no CPU `q40` cell. The model
is the one the #105 / #100 sittings installed — step 1 checks its md5 and
skips the 4.3 GB push.

## Why

At M = 1 the GEMV's `mm` is **74 % feed**, and that feed is DDR *latency*,
not bandwidth (rule 26). The only lever measured against it is `l2fetch`,
and #105 left it monotonic and unsaturated: on the `arena` cell the lead
moved `ns_per_tile` **145.23 → 137.46 → 120.23** at 0 / 64 / 192 KB. But
#105 confounded two axes — its A was the four-row loop with a zero lead
and its B1/B2/B3 were the one-row loop with the three leads — because the
kernel picked the one-row loop from `m - r0 == 1` inside the
prefetch-free entry. **The four-row loop has never run with a lead at
all**, and it already beat the one-row loop by 23 % at lead 0 (111.72 vs
145.23 ns/tile).

This sitting fills the matrix and sweeps the lead past 192 KB.
**Decision that hangs on it:** M==1 `mm` ≤ **840 µs** on at least one
variant, with decode ≥ A at all three G — and with it, whether PR #107
lands with the one-row loop as the default or closes with only its lead
machinery kept.

**One skel, one app, the cell chosen by two env vars.** Since this branch
the `(loop, lead)` pair rides in spare bits of the `moe_set_opts` word
(bit 7 = "bits [15:8] are authoritative", bits [15:8] = lead / 64 KB;
bit 6 = "bit 16 is authoritative", bit 16 = the row loop — one tune bit
per knob, so naming one leaves the other at the skel's build default), and
the ARM side throws when the skel's echo differs from what it sent. So a stale skel cannot silently run the wrong cell (rule 21 /
#97's failure mode, removed by construction), and the whole 2 × 5 matrix
comes out of **one** gtest invocation.

### Variants (4; contract §4.2)

| | row loop | lead | `applied` word | env beyond `NNTR_MOE_HTP_M1_GEMV=1` | cells |
|---|---|---|---|---|---|
| **A** (unchanged reference, runs first) | four-row | 0 (each column fetches itself) | `0x1` | none | full: G = 64 / 512 / 1024 × 2, mirrored |
| **B** | four-row | 192 KB | `0x381` | `NNTR_MOE_HTP_GEMV_LEAD_KB=192` | full |
| **C** | four-row | **`L*`** (from step 3's sweep, > 192 KB) | see table below | `NNTR_MOE_HTP_GEMV_LEAD_KB=<L*>` | full |
| **D** | one-row | `L*` | C's word `\| 0x10040` | `…LEAD_KB=<L*> NNTR_MOE_HTP_GEMV_ROWS1=1` | G = 64 × 2 + profile + gtest only |

`L*` is **read on the device in step 3**: the best `arena` lead of the
four-row (`rows1=0`) row of the microbench matrix. If that column is still
falling at 1536 KB, `L* = 1536`. If 192 KB is already the best four-row
cell, take the runner-up above it (384) so C still tests "past 192" — and
say so under Notes.

`applied` word per `L*` (the number the banner and the gtest must echo):

| `L*` | C | D |
|---|---|---|
| 384 | `0x681` | `0x106c1` |
| 768 | `0xc81` | `0x10cc1` |
| 1536 | `0x1881` | `0x118c1` |

A is the branch's own compile-time default (`HVX_GEMV_PF_LEAD_KB = 0u`,
`HVX_GEMV_M1_ROWS1 = 0u`), i.e. today's `htp_moe` behaviour, so A also
proves the runtime knob costs nothing: its `mm` must land near #100 A_L2's
**980.9** and #105 A's **973.0** (a cross-sitting sanity check, rule 23 —
not a gate).

`NNTR_MOE_HTP_M1_GEMV=1` is set **explicitly in every cell, A included**.
Every log must print one `[HTP] moe m1 gemv: on (applied=0x…) lead=…
rows1=… source=env` with **that variant's** word — `source=` still refers
to the `NNTR_MOE_HTP_M1_GEMV` switch alone (LEDGER ⑯), the `lead=` and
`rows1=` fields are what name the cell, and `default` there means the
skel's build value (0 and 0 on this branch). Every level-2 M==1 row must
read `blocks=0 m1_gemv=1408/1408`, or the run is void.

`NNTR_L2_DIFF` and "text = CPU q40" are n/a for `QS4CX_WH` (different
weights); the accuracy column is **"= A"**, and the arithmetic is unchanged
by construction, so any difference voids the variant.

### Ride-along (one line, free)

`DMA_REPLAY workers=1 load=0 pace=0` from `unittest_hvx_dma_probe`, on
code that has not changed since #94: **561.5 µs / 40.2 GB/s** means rule
30's ≈ 22 % drift is a session effect and ㉒'s feed half re-opens;
**≈ 719 µs / 31.4 GB/s** means the engine bound is real and the feed half
stays closed (LEDGER ⑥'s loose end).

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

Staged under **`W=/local/mnt/workspace/htp_moe/113/`**; `$W/md5.txt` is the
`md5sum` of every staged file and is reproduced here. All of it was built
in `/home/j2z0-lee/nntrainer-113` at `5be9c4ba`. Skel builds are **not**
byte-reproducible on this machine (two builds of the same source gave two
md5s), so the *staged file's* md5 is the identity and the device `md5sum`
in every log is what ties a cell to it.

| file (staged as) | md5 | built from |
|---|---|---|
| `$W/libnntr_hvx_skel.so` | `7e8eed5fdbf7266f04f6acc0952a8d79` | `5be9c4ba`, `HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh`, no `HEX_EXTRA_CFLAGS`; `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK (46 runtime imports)`. **One skel for all four variants** |
| `$W/nntrainer_causallm` | `d0cba1cd5f809c2826d9cc2b7fe1a6bc` | same commit, `build_android.sh --htp` (`jni/obj/local/arm64-v8a/`) |
| `$W/libcausallm_core.so` | `aff4e2197f5c6ff87cf47c6e08a0ca92` | same |
| `$W/libnntrainer.so` | `4ef373c9d44f35f5da7940fa7ccf07d6` | same; `readelf -d` NEEDED `libsdkl.so`, `libcdsprpc.so` |
| `$W/libccapi-nntrainer.so` | `cf83b88c1ed36e7d27aeaa5793904b8d` | same |
| `$W/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot (== #77 / #88 / #94 / #105) |
| `$W/unittest_hvx_mm_u8i4` | `31724757985f6f4aafb39faac4bb6989` | `ndk-build … unittest_hvx_mm_u8i4` (`test/jni/obj/local/arm64-v8a/`); carries the swept `MoeM1GemvFeedVsCompute` and `MoeLayerM1GemvMatchesHmx` |
| `$W/unittest_hvx_dma_probe` | `9372c5ddcc214f4864e09c1f3c274ab0` | same build line; **unchanged source since #94** — the rule-30 anchor |
| `$W/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` (== #77 / #88 / #94 / #105) |
| `$W/prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt` (the #88 / #94 / #105 prompt) |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (4316133120 B) | `7b7867fab51845664c0050c0a837073e` | #78, `--moe_dtype QS4CX_WH` (NPU model; already on the phone) |
| `tokenizer.json` (`q40-qs4cx-wh/`) | `7b8067a580173d3eb1697afae3b456f5` | HF |

Not staged, never pushed: `builddir/.../libcdsprpc.so` (rule 5).

Workstation sanity before pushing (all four ran clean here):

```
(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')             # 0
strings $W/nntrainer_causallm | grep -c 'per-layer-type totals'      # 0 (not a profile binary)
strings $W/libnntrainer.so | grep -c 'rows1=default'                 # 1 (#113's banner is in the app)
find $W -name 'libcdsprpc*' | wc -l                                  # 0
```

**The `rows1=default` line is not cosmetic.** `build_android.sh --cache`
skips the nntrainer `builddir` entirely when
`builddir/android_build_result/lib/arm64-v8a/libnntrainer.so` already
exists, so an ARM-side edit after the first build is silently left out
(this bit once in this very branch). After touching anything under
`nntrainer/`, run `ninja -C builddir && ninja -C builddir install` before
`build_android.sh --htp --cache`.

### If `$W` is not on the workstation you measure from (rule 22)

Rebuild from the commit. Skel builds are not byte-reproducible, so the
md5s will differ from the table — record the ones you push under "Notes
from the run" (rules 14 and 21). Clean checkout, NDK r30, SDK 6.4.0.1,
HexKL 6.4.0.1 (`source tools/htp/env.sh`, and export `HEXKL_ROOT`
explicitly — its location differs between workstations).

```
W=/local/mnt/workspace/htp_moe/113; mkdir -p $W
git fetch origin && git worktree add /tmp/n113 5be9c4ba && cd /tmp/n113
git submodule update --init --depth 1                       # a fresh worktree starts with subprojects/ empty
cp <another worktree>/Applications/CausalLM/lib/libtokenizers_android_c.a Applications/CausalLM/lib/ \
  || ./Applications/CausalLM/build_tokenizer_android.sh     # per-checkout, needs Rust
HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh && cp test/htp/build/libnntr_hvx_skel.so $W/
                                                            # prints UNDEFINED SYMBOLS OK (46 runtime imports)
(cd Applications/CausalLM && ./build_android.sh --htp)      # fresh builddir fails in `ninja install` on /usr/local:
#   cd /tmp/n113/builddir && meson configure -Dprefix=$PWD/android_build_result && ninja install
#   then: (cd Applications/CausalLM && ./build_android.sh --htp --cache)
# --cache skips builddir wholesale: after any edit under nntrainer/, run
#   ninja -C builddir && ninja -C builddir install   first, or libnntrainer.so stays stale.
cp Applications/CausalLM/jni/obj/local/arm64-v8a/{nntrainer_causallm,libcausallm_core.so,libnntrainer.so,libccapi-nntrainer.so} $W/
cp $ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so $W/
ln -sfn $PWD/subprojects/googletest/googletest test/jni/googletest
(cd test/jni && $ANDROID_NDK/ndk-build NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk \
   APP_BUILD_SCRIPT=./Android.mk NNTRAINER_ROOT=$PWD/../.. HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT \
   unittest_hvx_mm_u8i4 unittest_hvx_dma_probe -j8)
cp test/jni/obj/local/arm64-v8a/{unittest_hvx_mm_u8i4,unittest_hvx_dma_probe} $W/
cp $HEXKL_ROOT/lib/6.4.0.1/armv8_android26/libsdkl.so $W/
cp docs/measurements/77-prompt512.txt $W/prompt512.txt
(cd $W && md5sum * > md5.txt)
```

## Steps (workstation, phone on USB)

Shell setup once:

```
cd /home/j2z0-lee/nntrainer-113 && git fetch && git checkout htp/113-m1-gemv-lead-matrix && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/113; mkdir -p $W/logs
D=/data/local/tmp/nntrainer/causallm; T=/data/local/tmp/htp_u8i4_layer_test
therm() { adb shell dumpsys battery | grep -E 'level|temperature'; adb shell cat /sys/class/thermal/thermal_zone0/temp; }
# One skel for every variant; the cell is env, not a binary.
opts() { case $1 in A) echo "";; B) echo "NNTR_MOE_HTP_GEMV_LEAD_KB=192";;
                    C) echo "NNTR_MOE_HTP_GEMV_LEAD_KB=$LSTAR";;
                    D) echo "NNTR_MOE_HTP_GEMV_LEAD_KB=$LSTAR NNTR_MOE_HTP_GEMV_ROWS1=1";; esac; }
```

### 0. Device state (1 min)

```
adb devices          # exactly one device; record the serial under Notes (any S25 Ultra, contract §4.2)
therm                # checkpoint 0: battery %, temperature (tenths of °C), thermal_zone0 (m°C)
```

Screen off, charger in, cool start.

### 1. Install (≈ 8 min) — model reused

```
adb shell ls -l $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin     # 4316133120
adb shell md5sum $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin    # 7b7867fab5... -> no push
# only if either line differs:  adb push /local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh $D/models/q40-qs4cx-wh
adb shell mkdir -p $D/models $T
adb push $W/nntrainer_causallm $W/libcausallm_core.so $W/libnntrainer.so $W/libccapi-nntrainer.so \
         $W/libc++_shared.so $W/libsdkl.so $W/prompt512.txt $W/libnntr_hvx_skel.so $D/
adb push $W/unittest_hvx_mm_u8i4 $W/unittest_hvx_dma_probe $W/libc++_shared.so $W/libsdkl.so $W/libnntr_hvx_skel.so $T/
adb shell "chmod 755 $D/nntrainer_causallm $T/unittest_hvx_mm_u8i4 $T/unittest_hvx_dma_probe"
adb shell "md5sum $D/libnntr_hvx_skel.so $T/libnntr_hvx_skel.so" | tee $W/logs/skel.log   # both 947b013f...
```

Config edits, re-applied unconditionally (greedy, exact count, `moe_engine: htp`):

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
adb shell "md5sum $D/nntrainer_causallm $D/libnntrainer.so $D/libcausallm_core.so $D/libccapi-nntrainer.so \
  $D/libnntr_hvx_skel.so $T/unittest_hvx_mm_u8i4 $T/unittest_hvx_dma_probe $D/prompt512.txt \
  $D/models/q40-qs4cx-wh/tokenizer.json"
```

### 2. Ride-along, once, cold (≈ 2 min) — rule 30's anchor

Run this **before** anything warms the phone.

```
adb shell cat /sys/class/thermal/thermal_zone0/temp | sed 's/^/therm before dma probe: /' | tee -a $W/logs/therm.log
adb shell "cd $T && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='*MoeChunkReplay*'" \
  2>&1 | tee $W/logs/dma_probe.log | grep -E 'DMA_REPLAY |PASSED|FAILED'
```

Record the `DMA_REPLAY workers=1 load=0 pace=0 …` line verbatim in the
results. 561 µs ≈ #94 (drift is a session effect); 719 µs ≈ #100 (the
engine bound is real).

### 3. The matrix + bit-identity (≈ 6 min) — **this picks `L*`**

One gtest run, one skel, no env: `MoeM1GemvFeedVsCompute` drives
`moe_set_opts` itself and prints the whole matrix.

```
adb shell cat /sys/class/thermal/thermal_zone0/temp | sed 's/^/therm before matrix: /' | tee -a $W/logs/therm.log
adb shell "cd $T && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_mm_u8i4 \
  --gtest_filter='*MoeLayerM1GemvMatchesHmx*:*MoeM1GemvFeedVsCompute*'" \
  2>&1 | tee $W/logs/matrix.log | grep -E 'U8I4_FIELD|PASSED|FAILED|SKIPPED'
adb shell cat /sys/class/thermal/thermal_zone0/temp | sed 's/^/therm after matrix: /' | tee -a $W/logs/therm.log
# L* = the lead of the lowest ns_per_tile among the four-row arena lines (lead > 0):
grep 'cell=arena' $W/logs/matrix.log | grep 'rows1=0' | grep -v 'lead_kb=0 ' \
  | sed -E 's/.*lead_kb=([0-9]+).*ns_per_tile=([0-9.]+).*/\2 \1 KB/' | sort -n
```

Expected in `matrix.log`:

* `U8I4_FIELD path=moe_m1_gemv field=bad_elems_M1 value=0 of 20480`,
  `field=bad_elems_M4 value=0 of 81920`, `field=bit_identical value=yes`
  (the denominators are ten times #105's: all ten pairs are compared, and
  a single non-zero **voids that pair** — note which `bad_cell_M*` line
  appeared);
* **22 `U8I4_FIELD path=m1_bench` lines**: `cell=arena` and `cell=hot`
  × `lead_kb=0|192|384|768|1536` × `rows1=0|1` (10 each), plus `cell=heap`
  at `(0, rows1=0)` and `(192, rows1=1)`. Each carries `inflight_kb=` (**2 × the lead**: a lane holds the current
  block's boxes and the next block's — × 6 lanes for the L2 footprint),
  `mm_us= mm_min_us= lane_us= lanes= tiles= ns_per_tile= gbps=`. No
  `INVALID`;
* `[  PASSED  ] 2 tests.`

Then set `LSTAR` in the shell and note it: `LSTAR=<384|768|1536>`.

### 4. Profiles (≈ 5 min) — four level-2 runs at G = 64, never read for tok/s

```
prof() { # prof <A|B|C|D>
  adb shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/q40-qs4cx-wh/nntr_config.json && \
    NNTR_MOE_HTP_M1_GEMV=1 $(opts $1) NNTR_HTP_PROFILE=2 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" \
    2>&1 | tee $W/logs/prof_$1.log | grep -E 'moe m1 gemv|level=|K=2048 N=2048' | cut -c1-420
}
therm                                  # checkpoint 1
prof A; prof B; prof C; prof D
therm                                  # checkpoint 2
```

Expected per variant: one `[HTP] moe m1 gemv: on (applied=0x…) lead=…
rows1=… source=env` with the variant's word from the table above; the header
`[HTP-PROFILE] level=2 qos_mode=2 …` (**`qos_mode=2` in all four**; a `1`
voids that profile); the M==1 row `K=2048 N=2048 M==1 calls=1408 … mm <µs>
… blocks=0 m1_gemv=1408/1408]`; the M>1 row with `m1_gemv=0/23`.

### 5. E2E cells (≈ 26 min) — mirrored order per G

```
run() { # run <A|B|C|D> <G> <run#>
  adb shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' models/q40-qs4cx-wh/nntr_config.json && \
    grep num_to_generate models/q40-qs4cx-wh/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    NNTR_MOE_HTP_M1_GEMV=1 $(opts $1) NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" \
    2>&1 | tee $W/logs/$1_G$2_r$3.log | grep -E '^(prefill|generation|total|peak memory)|moe m1 gemv|libnntr_hvx_skel'
}
run D 64 1
for g in 64 512 1024; do
  run A $g 1; run B $g 1; run C $g 1
  run C $g 2; run B $g 2; run A $g 2
  therm                                # checkpoints 3, 4, 5
done
run D 64 2
```

Expected in every log: `prefill: 512 tokens, … TPS`, `generation: <G>
tokens, … TPS`, `total: … ms`, `peak memory: … KB`, exactly one
`[HTP] moe m1 gemv: on (applied=0x…)` with the **variant's** word, and
**no** `[HTP-PROFILE]` block. `prefill: N` with N ≠ 512 voids the prefill
column; `generation: N` ≠ G is a failed config edit.

**Short form (≈ 35 min total), if time runs out:** keep A and C at all
three G, B at G = 64 and 1024, D at G = 64 only (steps 2–4 stay whole —
they are what the verdict is read from).

### 6. Checks on the workstation (1 min)

```
grep -c 'moe m1 gemv: on' $W/logs/*_G*_r*.log $W/logs/prof_*.log          # 1 each
grep -h -o 'applied=0x[0-9a-f]*' $W/logs/prof_*.log                       # 0x1, 0x381, C's, D's
grep -l 'HTP-PROFILE' $W/logs/*_G*_r*.log                                 # nothing
for v in A B C D; do printf '%s: ' $v; grep 'M==1' $W/logs/prof_$v.log | grep -o 'blocks=[0-9]* m1_gemv=[0-9/]*'; done
for g in 64 512 1024; do for r in 1 2; do for v in B C; do
  printf '%s vs A G=%s r%s: ' $v $g $r
  diff <(sed -n '/^=====/q;p' $W/logs/A_G${g}_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel') \
       <(sed -n '/^=====/q;p' $W/logs/${v}_G${g}_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel') >/dev/null && echo same || echo DIFFERENT
done; done; done
for r in 1 2; do printf 'D vs A G=64 r%s: ' $r
  diff <(sed -n '/^=====/q;p' $W/logs/A_G64_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel') \
       <(sed -n '/^=====/q;p' $W/logs/D_G64_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel') >/dev/null && echo same || echo DIFFERENT; done
cat $W/logs/skel.log $W/logs/therm.log
```

Then fill the tables below, commit this file on the same branch, push, set
#113 to `state:measured`.

## Results (fill in)

Unit (serial from `adb devices`): **`________`**. Battery %, charger,
screen state:
`thermal_zone0` (m°C) at checkpoints 0 / 1 / 2 / 3 / 4 / 5:

`L*` chosen in step 3: **______ KB** — because:

### Ride-along (step 2)

| cell | µs/call | GB/s | reads as |
|---|---|---|---|
| `DMA_REPLAY workers=1 load=0 pace=0` | | | #94's 561.5 / 40.2 → drift is a session effect · #100's 719.6 / 31.4 → the engine bound is real |

### The matrix (step 3) — `ns_per_tile`, lower is better

| cell | lead 0 | 192 | 384 | 768 | 1536 |
|---|---|---|---|---|---|
| arena, rows1=0 (four-row) | | | | | |
| arena, rows1=1 (one-row) | | | | | |
| hot, rows1=0 | | | | | |
| hot, rows1=1 | | | | | |
| heap | (rows1=0, lead 0): | (rows1=1, lead 192): | | | |

Reference from #105 (different sitting, rule 23 — for orientation only):
arena four-row lead 0 **111.72**; arena one-row 0 / 64 / 192 =
**145.23 / 137.46 / 120.23**; hot one-row 192 = 31.25; hot four-row lead 0
= 70.77.

Where did the arena four-row column turn (the L2 budget answer), and what
`inflight_kb` was that (× 6 lanes = the L2 footprint)?

**Read the lead as an ordinal, not a byte count.** The hardware allows
three outstanding `l2fetch` per thread, so stage A issues block b+1's
*gate* box at the start of block b but its *up* box only after block b's
last gate column: the gate half gets the full lead and the up half about
two column-computes. Half of stage A's traffic therefore runs at well
under the nominal KB, and a fourth outstanding box — giving the up half
its own lead — is #114's, not this sitting's.

Accuracy: `bad_elems_M1` ___ of 20480 · `bad_elems_M4` ___ of 81920 ·
`bit_identical` ___ · any `bad_cell_M*` line:

### Profiles (step 4), `[HTP-PROFILE]` level 2, G = 64, µs/call

| variant | applied | qos_mode | M==1 `mm` | M==1 `dsp` | M==1 host / transport | blocks / m1_gemv | M>1 `dsp` | M>1 m1_gemv |
|---|---|---|---|---|---|---|---|---|
| A | `0x1` | | | | | | | |
| B | `0x381` | | | | | | | |
| C | | | | | | | | |
| D | | | | | | | | |

`mm` vs A (%): B ___ · C ___ · D ___. **Gate: ≤ 840.0 µs on at least
one.** Reference: #100 A_L2 **980.9**, #105 A 973.0, #105 B3 930.9.

### E2E (NPU model, prompt 512, 8 threads)

| variant | gen | run | prefill tok/s | decode tok/s (all) | peak RSS (KB) | text = A (same G, run)? | applied (from the log) |
|---|---|---|---|---|---|---|---|
| A | 64 | 1 | | | | reference | |
| B | 64 | 1 | | | | | |
| C | 64 | 1 | | | | | |
| C | 64 | 2 | | | | | |
| B | 64 | 2 | | | | | |
| A | 64 | 2 | | | | reference | |
| D | 64 | 1 | | | | | |
| D | 64 | 2 | | | | | |
| A | 512 | 1 | | | | reference | |
| B | 512 | 1 | | | | | |
| C | 512 | 1 | | | | | |
| C | 512 | 2 | | | | | |
| B | 512 | 2 | | | | | |
| A | 512 | 2 | | | | reference | |
| A | 1024 | 1 | | | | reference | |
| B | 1024 | 1 | | | | | |
| C | 1024 | 1 | | | | | |
| C | 1024 | 2 | | | | | |
| B | 1024 | 2 | | | | | |
| A | 1024 | 2 | | | | reference | |

Decode means (of the two mirrored runs) and deltas against A:

| G | A | B | C | D |
|---|---|---|---|---|
| 64 | | | | |
| 512 | | | | n/a |
| 1024 | | | | n/a |

Prefill means (G = 64 / 512 / 1024), read with rule 27's mirrored-order
caveat — a single cell's −5 % is position drift, the M>1 `dsp` column
above is the tie-breaker:

Reference, #105's sitting: decode A **27.743 / 26.879 / 24.826**; prefill
A 498.5 / 453.5 / 380.7. Goal ≥ 50 decode, prefill ≥ 497.

## Verdict (fill in)

* **Gate** = M==1 `mm` ≤ **840.0 µs** on at least one variant **and** that
  variant's decode mean ≥ A at G = 64, 512 **and** 1024 **and** text
  identical to A at all three G.
* **Accuracy gate (a)** = `bit_identical yes`, `bad_elems_M1/M4 = 0`. A
  single non-zero voids the pair; the lead and the loop change no
  arithmetic.
* **Prefill (standing)** = M>1 `dsp` within 2.5 % of A and prefill tok/s
  ≥ −5 % of A under rule 27.
* **PR #107's fate:** a four-row variant (B or C) winning closes it and
  keeps only the lead machinery (already on this branch); D winning lands
  the one-row loop as the default (`HVX_GEMV_M1_ROWS1 = 1u`).
* **The landed defaults** become the winning pair: set
  `HVX_GEMV_PF_LEAD_KB` and `HVX_GEMV_M1_ROWS1` in
  `nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.h`. The two env vars
  stay as documented measurement switches, like `HTP_MM_NO_PREFETCH`.

## Notes from the run

<serial, battery, thermal, first-run page faults, FARF/AEE errors, rebuilt
md5s if `$W` was not present, anything stale>
