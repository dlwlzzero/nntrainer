# Measurement 117: the M=1 GEMV fed from VTCM by DMA (the `f2` shape) under the one-row loop, with #113's device confirmation riding along

Branch `htp/117-m1-gemv-vtcm-feed` (code head `3c4912fa`, this file on the
same branch) — estimated device time: **≈ 55 min** (≈ 40 min if the short
form in step 5 is taken)

Plan: `docs/plans/117-m1-gemv-vtcm-feed.md` §4 step 5, with the
supervisor's cycle-13 amendments on issue #117 (base = `htp_moe` head with
the D192 default already landed; an **A0** cell carries #113's device
confirmation). Issue #117 (tracker #76; LEDGER ㉒ feed half, rules 26–33).
Contract `docs/plans/0001-htp-moe-decode-agent-system.md`.

Only the NPU model `q40-qs4cx-wh` runs here; no CPU `q40` cell (the CPU
"now" stands). The model is the one the #105 / #113 sittings installed —
step 1 checks its md5 and skips the 4.3 GB push.

## Why

Under D192 (one-row loop + 192 KB `l2fetch` lead, the default since PR
#115) the M=1 GEMV's `mm` is **937.0 µs/call** for 22.02 MB = **23.5 GB/s**,
and rules 26 / 31 say the compute side is exhausted: no (loop, lead) cell
of the 2 × 5 board goes lower, and a direct HVX read of DDR tops out at
21–27 GB/s on this SoC (rule 27). The DMA engine reads the same bytes at
**31.6 GB/s while HVX streams out of VTCM** (#100's `f2` cell, rule 29),
and #113's cold anchor fixed that number as real (724.0 µs / 31.2 GB/s,
rule 32). 31.6 / 23.5 = 1.33× on the column that is 62 % of the token.

This branch stages each expert's `wh_bytes` into VTCM by DMA one expert
ahead — whole-matrix descriptors (`f2`: gate_up 57 344 B × 64 rows, down
32 768 B × 56 rows; 8 per call), two 3.5 MiB gate_up slabs double-buffered
by expert in the arena the M=1 path never used, the four downs two per
slab after — and the one-row loop reads the VTCM copy. Stage A and C run
one pool run per expert so every slab reuse follows a join (the host
scoreboard holds the schedule exactly); the lead is off by construction
under the feed. Bound: 22.02 MB / 31.6 GB/s = 697 µs + ≈ 20 µs compute
tail + 6 fork/joins ≈ **720–750 µs**.

**Decision that hangs on it:** B's level-2 M==1 `mm` ≤ **760.0 µs/call**
with decode ≥ A at all three G and text = A. Pass: `HVX_GEMV_M1_FEED`
flips to `1u` and the PR closes #117. `mm` > 760 with the anchor at ≈ 31
GB/s closes ㉒ for good at what the arena allows; the path then does not
land (the host scoreboard and the `vtcm` microbench cell stay as the
record).

**One skel, one app set, the cell chosen by env.** Bit 5 of the
`moe_set_opts` word says bit 17 (the feed) is authoritative, the same
per-knob shape as #113's pair; `NNTR_MOE_HTP_GEMV_FEED` is sent only when
set, so an unset run keeps A's word. The ARM side throws when the skel's
echo differs from what it sent (rule 21), so a stale skel cannot run the
wrong cell silently.

### Variants (3 + one profile pair; contract §4.2)

| | loop | lead | feed | `applied` | env beyond `NNTR_MOE_HTP_M1_GEMV=1` | cells |
|---|---|---|---|---|---|---|
| **A** (unchanged reference, first) | one-row | 192 KB | arena (build default) | `0x103c1` | none | full: G = 64 / 512 / 1024 × 2, mirrored |
| **A0** (#113's confirmation: the pre-#115 default) | four-row | 0 | arena | `0xc1` | `NNTR_MOE_HTP_GEMV_LEAD_KB=0 NNTR_MOE_HTP_GEMV_ROWS1=0` | full |
| **B** | one-row | (forced 0 by the kernel; banner prints 192) | **VTCM** | `0x303e1` | `NNTR_MOE_HTP_GEMV_FEED=1` | full |

A vs A0 is the same-sitting reproduction of #113's +4.11 / +3.29 / +4.26 %
(A0 was #113's A, A was its D192), and BENCHMARK's "now" moves to this A.
B vs A is #117's question.

`NNTR_MOE_HTP_M1_GEMV=1` is set **explicitly in every cell, A included**.
Every log must print exactly one

```
[HTP] moe m1 gemv: on (applied=0x…) lead=…KB rows1=… feed=… source=env
```

with **that variant's** word: A `applied=0x103c1 … lead=192KB rows1=1
feed=default`, A0 `applied=0xc1 … lead=0KB rows1=0 feed=default`, B
`applied=0x303e1 … lead=192KB rows1=1 feed=vtcm`. `feed=default` means
the skel's build value (arena on this branch); `source=` still refers to
the `NNTR_MOE_HTP_M1_GEMV` switch alone (LEDGER ⑯). Every level-2 M==1 row
must read `blocks=0 m1_gemv=1408/1408`, and B's additionally
`feed=1408/1408` with a `DMA ring: desc=10/call` line, or the run is void.

`NNTR_L2_DIFF` and "text = CPU q40" are n/a for `QS4CX_WH` (different
weights); the accuracy column is **"= A"**. A0 and B are the same int32
sums and the same epilogues on the same bytes (B reads a DMA'd copy of
them), so any differing byte voids that variant.

### The gate, as read

| gate | read from | pass |
|---|---|---|
| **verdict** | level-2 profile, G = 64, M==1 row, `mm` | **B ≤ 760.0 µs/call**; B `dsp` lower than A's by the same amount ± 20 µs |
| **decode** | E2E means of the mirrored pair, same sitting | B ≥ A at G = 64, 512 **and** 1024 |
| **text** | every A0 / B log vs the A log of the same G and run (banner and md5 lines excluded) | byte-identical |
| **accuracy (a), device** | step 3 | `bit_identical value=yes`, `bad_elems_M1 = 0 of 40960`, `bad_elems_M4 = 0 of 163840` (20 cells each: the ten (loop, lead) pairs × feed); the three `cell=vtcm` lines present, no `INVALID` |
| **prefill (standing)** | level-2 M>1 `dsp`; prompt-512 prefill tok/s | B's M>1 `dsp` within 2.5 % of A's; prefill tok/s ≥ −5 % of A under the mirrored-order rule (rule 27); `m1_gemv=0/23 feed=0/23` on every M>1 row |
| **DMA anchor** (rules 30/32) | step 2, cold, first | the `DMA_REPLAY workers=1 load=0 pace=0` line verbatim; ≈ 724 µs / 31.2 GB/s is the band; the verdict is read *scaled by it* if it moves |
| **in-situ feed rate** | B's level-2 M==1 `DMA ring:` line, `engine lo..hi GB/s` | **lower bound ≥ 28 GB/s**. This is what the issue's "`weight DMA:` line ≥ 28" means: the `weight DMA:` line's "averaged over the call" divides by `dsp` and reads ≈ 27 even at the gate; the ring line's `busy` bracket is the engine |

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

Staged under **`W=/local/mnt/workspace/htp_moe/117/`**; `$W/md5.txt` is the
`md5sum` of every staged file and is reproduced here. All of it was built
in `/home/j2z0-lee/nntrainer-117` at `3c4912fa`. Skel builds are **not**
byte-reproducible on this machine (two builds of the same source gave two
md5s), so the *staged file's* md5 is the identity and the device `md5sum`
in every log is what ties a cell to it.

| file (staged as) | md5 | built from |
|---|---|---|
| `$W/libnntr_hvx_skel.so` | `5cea0a5a35c72c72ce2daffa8e7588e5` | `3c4912fa`, `HEXKL_ROOT=~/Qualcomm/hexkl-1.0-beta.2/hexkl_addon HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh`, no `HEX_EXTRA_CFLAGS`; `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK (46 runtime imports)`. **One skel for all three variants** (`HVX_GEMV_M1_FEED = 0u`; the feed is bit 17 of the word) |
| `$W/nntrainer_causallm` | `21221e9006636d054844361a83a3836a` | same commit, `build_android.sh --htp` (`jni/obj/local/arm64-v8a/`) |
| `$W/libcausallm_core.so` | `ffbe434507a695132eb5dae7877fc647` | same |
| `$W/libnntrainer.so` | `39e375adb165e41e908e00b85d7bcd66` | same; `readelf -d` NEEDED `libsdkl.so`, `libcdsprpc.so`; carries the `feed=%s` banner and the `feed=%llu/%llu` row field |
| `$W/libccapi-nntrainer.so` | `d1bf42a37f3a5e530567fd180df414af` | same |
| `$W/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot (== #77 / #88 / #94 / #105 / #113) |
| `$W/unittest_hvx_mm_u8i4` | `2781e9fb2a70ac00bed6eb1da6608856` | `ndk-build … unittest_hvx_mm_u8i4` (`test/jni/obj/local/arm64-v8a/`); carries the ×feed `MoeLayerM1GemvMatchesHmx` and the `cell=vtcm` lines of `MoeM1GemvFeedVsCompute` |
| `$W/unittest_hvx_dma_probe` | `639c2eb4993fee118f8a86ddeb7adbe4` | same build line; its replay entry and cells are **unchanged since #94** — the rule-30 anchor; only its step-0 trace call moved to 31 stage slots |
| `$W/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` (== #77 … #113) |
| `$W/prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt` (the #88 / #94 / #105 / #113 prompt) |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (4316133120 B) | `7b7867fab51845664c0050c0a837073e` | #78, `--moe_dtype QS4CX_WH` (NPU model; already on the phone) |
| `tokenizer.json` (`q40-qs4cx-wh/`) | `7b8067a580173d3eb1697afae3b456f5` | HF |

Not staged, never pushed: `builddir/.../libcdsprpc.so` (rule 5).

Workstation sanity before pushing (all four ran clean here):

```
(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')             # 0
strings $W/nntrainer_causallm | grep -c 'per-layer-type totals'      # 0 (not a profile binary)
strings $W/libnntrainer.so | grep -c 'feed=%s source=%s'             # 1 (#117's banner is in the app)
find $W -name 'libcdsprpc*' | wc -l                                  # 0
```

**The `feed=%s` line is not cosmetic.** `build_android.sh --cache` skips
the nntrainer `builddir` entirely when
`builddir/android_build_result/lib/arm64-v8a/libnntrainer.so` already
exists, so an ARM-side edit after the first build is silently left out
(#113's trap). After touching anything under `nntrainer/`, run `ninja -C
builddir && ninja -C builddir install` before `build_android.sh --htp
--cache`.

### If `$W` is not on the workstation you measure from (rule 22)

Rebuild from the commit. Skel builds are not byte-reproducible, so the
md5s will differ from the table — record the ones you push under "Notes
from the run" (rules 14 and 21). Clean checkout, NDK r30, SDK 6.4.0.1,
HexKL 6.4.0.1 (`source tools/htp/env.sh`, and export `HEXKL_ROOT`
explicitly — its location differs between workstations; LEDGER §3a).

```
W=/local/mnt/workspace/htp_moe/117; mkdir -p $W
git fetch origin && git worktree add /tmp/n117 3c4912fa && cd /tmp/n117
source tools/htp/env.sh; export HEXKL_ROOT=$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon HEXKL_SDK_VER=6.4.0.1
git submodule update --init --depth 1                       # a fresh worktree starts with subprojects/ empty
cp <another worktree>/Applications/CausalLM/lib/libtokenizers_android_c.a Applications/CausalLM/lib/ \
  || ./Applications/CausalLM/build_tokenizer_android.sh     # per-checkout, needs Rust
./test/htp/build.sh && cp test/htp/build/libnntr_hvx_skel.so $W/
                                                            # prints UNDEFINED SYMBOLS OK (46 runtime imports)
(cd Applications/CausalLM && ./build_android.sh --htp)      # fresh builddir fails in `ninja install` on /usr/local:
#   cd /tmp/n117/builddir && meson configure -Dprefix=$PWD/android_build_result && ninja install
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
cd /home/j2z0-lee/nntrainer-117 && git fetch && git checkout htp/117-m1-gemv-vtcm-feed && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/117; mkdir -p $W/logs
D=/data/local/tmp/nntrainer/causallm; T=/data/local/tmp/htp_u8i4_layer_test
therm() { adb shell dumpsys battery | grep -E 'level|temperature'; adb shell cat /sys/class/thermal/thermal_zone0/temp; }
# One skel for every variant; the cell is env, not a binary.
opts() { case $1 in A) echo "";;
                    A0) echo "NNTR_MOE_HTP_GEMV_LEAD_KB=0 NNTR_MOE_HTP_GEMV_ROWS1=0";;
                    B) echo "NNTR_MOE_HTP_GEMV_FEED=1";; esac; }
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
adb shell "md5sum $D/libnntr_hvx_skel.so $T/libnntr_hvx_skel.so" | tee $W/logs/skel.log   # both 5cea0a5a35c72c72ce2daffa8e7588e5
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

### 2. Anchor, once, cold (≈ 2 min) — rules 30 / 32

Run this **before** anything warms the phone.

```
adb shell cat /sys/class/thermal/thermal_zone0/temp | sed 's/^/therm before dma probe: /' | tee -a $W/logs/therm.log
adb shell "cd $T && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='*MoeChunkReplay*'" \
  2>&1 | tee $W/logs/dma_probe.log | grep -E 'DMA_REPLAY |DMA_REPLAY_NOTE|DMA_REPLAY_TRACE|PASSED|FAILED'
```

Record the `DMA_REPLAY workers=1 load=0 pace=0 …` line verbatim in the
results. ≈ 724 µs / 31.2 GB/s is the expected band (#113, #100); the
`f2` / `f2_load` lines beside it are the shape B's kernel issues. (The
`checksum_ok=n` on the `DMA_REPLAY workers=` lines and the resulting
`FAILED` are #99's known pre-existing condition, 11 lines, accepted the
same way in #100 and #113.) The step-0 `DMA_REPLAY_TRACE` line runs the
HMX path's `mm_u8i4_moe_layer_timed` (flags 0) with 31 stage slots; a
`DMA_REPLAY_NOTE moe_layer_timed err=` line instead means a stale skel —
stop and re-push.

### 3. Bit-identity + the matrix with the `vtcm` cell (≈ 6 min)

One gtest run, one skel, no env: the tests drive `moe_set_opts` themselves.

```
adb shell cat /sys/class/thermal/thermal_zone0/temp | sed 's/^/therm before matrix: /' | tee -a $W/logs/therm.log
adb shell "cd $T && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_mm_u8i4 \
  --gtest_filter='*MoeLayerM1GemvMatchesHmx*:*MoeM1GemvFeedVsCompute*'" \
  2>&1 | tee $W/logs/matrix.log | grep -E 'U8I4_FIELD|PASSED|FAILED|SKIPPED'
adb shell cat /sys/class/thermal/thermal_zone0/temp | sed 's/^/therm after matrix: /' | tee -a $W/logs/therm.log
grep 'cell=vtcm\|cell=arena.*rows1=1\|cell=hot.*rows1=1' $W/logs/matrix.log | grep -E 'lead_kb=(0|192) ' \
  | sed -E 's/.*cell=([a-z]+) lead_kb=([0-9]+) rows1=([01]) feed=([01]).*mm_us=([0-9.]+) mm_min_us=([0-9.]+).*ns_per_tile=([0-9.]+) gbps=([0-9.]+).*/\1 lead=\2 rows1=\3 feed=\4 mm=\5 min=\6 ns_per_tile=\7 gbps=\8/'
```

Expected in `matrix.log`:

* `U8I4_FIELD path=moe_m1_gemv field=bad_elems_M1 value=0 of 40960`,
  `field=bad_elems_M4 value=0 of 163840`, `field=bit_identical value=yes`
  (twice #113's denominators: the ten pairs with the arena read and with
  the feed; a single non-zero **voids that cell** — note which
  `bad_cell_M* … feed=` line appeared);
* **25 `U8I4_FIELD path=m1_bench` lines**: #113's 22 (`cell=arena` and
  `cell=hot` × 5 leads × 2 loops, `cell=heap` × 2), every one now with
  `feed=0`, plus **three `cell=vtcm … feed=1` lines**: `lead_kb=0 rows1=1`,
  `lead_kb=192 rows1=1` (must agree with each other: the kernel forces the
  lead off under the feed) and `lead_kb=0 rows1=0`. Each carries
  `mm_us= mm_min_us= lane_us= lanes= tiles= ns_per_tile= gbps=`. No
  `INVALID`;
* `[  PASSED  ] 2 tests.`

Read the `vtcm` one-row line against `arena lead_kb=192 rows1=1` (#113:
129.05 ns/tile ≈ 937 µs) and `hot lead_kb=192 rows1=1` (31.25 ns/tile,
the VTCM-read compute proxy): `mm_min_us` of the `vtcm` line is the
microbench's verdict number, and `mm_min_us − 22 020 KB / (anchor GB/s)`
is the compute tail + the six fork/joins (risk table, plan §5: ≫ 40 µs
files the §3.2 upgrade).

### 4. Profiles (≈ 4 min) — level-2 runs at G = 64, never read for tok/s

```
prof() { # prof <A|A0|B>
  adb shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/q40-qs4cx-wh/nntr_config.json && \
    NNTR_MOE_HTP_M1_GEMV=1 $(opts $1) NNTR_HTP_PROFILE=2 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" \
    2>&1 | tee $W/logs/prof_$1.log | grep -E 'moe m1 gemv|level=|K=2048 N=2048|weight DMA|DMA ring' | cut -c1-480
}
therm                                  # checkpoint 1
prof A; prof B
prof A0                                # optional (≈ 1.5 min); A vs A0 at level 2 is #113's mm 1000.2 vs 937.0 reproduced
therm                                  # checkpoint 2
```

Expected per variant: one `[HTP] moe m1 gemv: on (applied=0x…) … feed=…
source=env` with the variant's word; the header `[HTP-PROFILE] level=2
qos_mode=2 …` (**`qos_mode=2` in all**; a `1` voids that profile); the
M==1 row `K=2048 N=2048 M==1 calls=1408 … mm <µs> … blocks=0
m1_gemv=1408/1408 feed=<n>/1408]`; the M>1 row with `m1_gemv=0/23
feed=0/23`.

* **A / A0**: `feed=0/1408`; the line after the row reads `weight DMA:
  n/a (direct arena read inside mm, no ring; …)`; `DMA ring: desc=2/call
  waits=2 …` (the two `moe_dma_copy` pieces).
* **B**: `feed=1408/1408`; `weight DMA: 21504 KB/call, first 3584 KB took
  <µs> = <GB/s>; averaged over the call <GB/s>` (the averaged number
  divides by `dsp` and reads ≈ 27 at the gate — not the gate's number);
  `DMA ring: desc=10/call waits=10 (blocked …) wait=<µs> us [act 0.0 gu
  0.0+dn 0.0] busy=<lo>..<hi> us -> engine <lo>..<hi> GB/s depth max=2..3
  first expert ready at <µs> last issue at <µs> of <dsp>`. **`engine`'s
  lower bound is the in-situ feed rate the gate reads (≥ 28).** `gu` / `dn`
  read 0.0 on purpose: the waits sit inside `mm`, not in the drain slots.
  `busy ≈ mm` says the engine never idled; `busy ≪ mm` names the gap.

### 5. E2E cells (≈ 30 min) — mirrored order per G

```
run() { # run <A|A0|B> <G> <run#>
  adb shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' models/q40-qs4cx-wh/nntr_config.json && \
    grep num_to_generate models/q40-qs4cx-wh/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    NNTR_MOE_HTP_M1_GEMV=1 $(opts $1) NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" \
    2>&1 | tee $W/logs/$1_G$2_r$3.log | grep -E '^(prefill|generation|total|peak memory)|moe m1 gemv|libnntr_hvx_skel'
}
for g in 64 512 1024; do
  [ $g = 1024 ] && sleep 60                # cooldown before the G = 1024 block
  run A $g 1; run A0 $g 1; run B $g 1
  run B $g 2; run A0 $g 2; run A $g 2
  therm                                    # checkpoints 3, 4, 5
done
```

Expected in every log: `prefill: 512 tokens, … TPS`, `generation: <G>
tokens, … TPS`, `total: … ms`, `peak memory: … KB`, exactly one
`[HTP] moe m1 gemv: on (applied=0x…)` with the **variant's** word, and
**no** `[HTP-PROFILE]` block. `prefill: N` with N ≠ 512 voids the prefill
column; `generation: N` ≠ G is a failed config edit.

**Short form (≈ 40 min total), if time runs out:** steps 2–4 whole (they
are what the verdict is read from); E2E with one mirrored pair per G
(A, B at 64; B, A at 512; A, B at 1024) plus a second pair at G = 64, and
A0 at G = 64 only (× 2, mirrored with A). Say so under Notes.

### 6. Checks on the workstation (1 min)

```
grep -c 'moe m1 gemv: on' $W/logs/*_G*_r*.log $W/logs/prof_*.log          # 1 each
grep -h -o 'applied=0x[0-9a-f]* lead=[0-9]*KB rows1=[01] feed=[a-z]*' $W/logs/prof_*.log   # A 0x103c1 … feed=default, B 0x303e1 … feed=vtcm, A0 0xc1
grep -l 'HTP-PROFILE' $W/logs/*_G*_r*.log                                 # nothing
for v in A A0 B; do [ -f $W/logs/prof_$v.log ] && { printf '%s: ' $v; grep 'M==1' $W/logs/prof_$v.log | grep -o 'mm [0-9.]*\|blocks=[0-9]* m1_gemv=[0-9/]* feed=[0-9/]*' | tr '\n' ' '; echo; }; done
grep -h 'DMA ring' $W/logs/prof_B.log
for g in 64 512 1024; do for r in 1 2; do for v in A0 B; do
  [ -f $W/logs/${v}_G${g}_r${r}.log ] || continue
  printf '%s vs A G=%s r%s: ' $v $g $r
  diff <(sed -n '/^=====/q;p' $W/logs/A_G${g}_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel') \
       <(sed -n '/^=====/q;p' $W/logs/${v}_G${g}_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel') >/dev/null && echo same || echo DIFFERENT
done; done; done
cat $W/logs/skel.log $W/logs/therm.log
```

Then fill the tables below, commit this file on the same branch, push, set
#117 to `state:measured`.

## Results (fill in)

Unit (serial from `adb devices`): ______. Battery / charger / screen:
______. `thermal_zone0` at the checkpoints 0 … 5: ______.

### Anchor (step 2)

| cell | µs/call | GB/s | reads as |
|---|---|---|---|
| `DMA_REPLAY workers=1 load=0 pace=0` | | | (#113: 724.0 / 31.2; #100: 719.6 / 31.4) |
| `DMA_REPLAY … f2` / `f2_load` (if printed) | | | (#100: 695.5 / 31.7 and 697.6 / 31.6) |

### The `vtcm` cell beside `arena` and `hot` (step 3) — `mm_min_us`, `ns_per_tile`

| cell | lead 0, rows1=1 | 192, rows1=1 | 0, rows1=0 | #113 reference (ns/tile) |
|---|---|---|---|---|
| vtcm (feed=1) | | | | — |
| arena (feed=0) | | | | 154.34 / **129.05** / 133.51 |
| hot (feed=0) | | | | 42.74 / **31.25** / 72.15 |

`vtcm` (192, rows1=1) `mm_min_us` − 22 020 KB / anchor GB/s = ______ µs
(the compute tail + 6 fork/joins; ≫ 40 files plan §3.2's upgrade).

Accuracy: `bad_elems_M1` ______ of 40960 · `bad_elems_M4` ______ of
163840 · `bit_identical` ______ · `bad_cell_M*` lines: ______ · 25
`m1_bench` lines, `INVALID`: ______ · `[  PASSED  ] 2 tests`: ______.

### Profiles (step 4), `[HTP-PROFILE]` level 2, G = 64, µs/call

| variant | applied | qos_mode | M==1 `mm` | M==1 `dsp` | host / transport | blocks / m1_gemv / feed | `DMA ring:` desc, waits, busy, engine lo..hi GB/s, depth max | `weight DMA:` line | M>1 `dsp` | M>1 m1_gemv / feed |
|---|---|---|---|---|---|---|---|---|---|---|
| A (D192, arena) | `0x103c1` | | | | | 0 / 1408⁄1408 / 0⁄1408 | desc=2 … | n/a (direct arena read …) | | 0/23 / 0/23 |
| B (D192 + VTCM feed) | `0x303e1` | | | | | 0 / 1408⁄1408 / 1408⁄1408 | desc=10 … | 21504 KB/call, first 3584 KB took … | | 0/23 / 0/23 |
| A0 (four-row, lead 0) — optional | `0xc1` | | | | | 0 / 1408⁄1408 / 0⁄1408 | desc=2 … | n/a | | 0/23 / 0/23 |

Reference: #113's sitting (same unit `R3CY205ZMND`): A0-equivalent
**1000.2** / dsp 1059.4, D192 **937.0** / dsp 995.5; M>1 `dsp`
16233–16247. Gate: B `mm` ≤ **760.0**; B `mm` scaled by (31.2 / anchor
GB/s) = ______.

### E2E (NPU model, prompt 512, 8 threads)

| variant | gen | run | prefill tok/s | decode tok/s | peak RSS (KB) | text = A (same G, run)? | applied (from the log) |
|---|---|---|---|---|---|---|---|
| A | 64 | 1 | | | | reference | `0x103c1` |
| A0 | 64 | 1 | | | | | `0xc1` |
| B | 64 | 1 | | | | | `0x303e1` |
| B | 64 | 2 | | | | | `0x303e1` |
| A0 | 64 | 2 | | | | | `0xc1` |
| A | 64 | 2 | | | | reference | `0x103c1` |
| A | 512 | 1 | | | | reference | `0x103c1` |
| A0 | 512 | 1 | | | | | `0xc1` |
| B | 512 | 1 | | | | | `0x303e1` |
| B | 512 | 2 | | | | | `0x303e1` |
| A0 | 512 | 2 | | | | | `0xc1` |
| A | 512 | 2 | | | | reference | `0x103c1` |
| A | 1024 | 1 | | | | reference | `0x103c1` |
| A0 | 1024 | 1 | | | | | `0xc1` |
| B | 1024 | 1 | | | | | `0x303e1` |
| B | 1024 | 2 | | | | | `0x303e1` |
| A0 | 1024 | 2 | | | | | `0xc1` |
| A | 1024 | 2 | | | | reference | `0x103c1` |

Decode means (of the two mirrored runs) and deltas:

| G | A | A0 | A vs A0 (%) | B | B vs A (%) |
|---|---|---|---|---|---|
| 64 | | | (#113: +4.11) | | |
| 512 | | | (#113: +3.29) | | |
| 1024 | | | (#113: +4.26) | | |

Prefill means (G = 64 / 512 / 1024), rule 27's mirrored-order caveat:
A ______, A0 ______, B ______ → B vs A ______ % (budget −5 %); M>1 `dsp`
B vs A ______ % (budget 2.5 %).

Reference: #113's sitting, decode A (= this A0) **27.444 / 26.829 /
25.930**, D192 (= this A) **28.572 / 27.711 / 27.035**; prefill A
536.63 / 516.98 / 420.24, D192 532.50 / 517.25 / 435.93. Goal ≥ 50
decode, prefill ≥ 497.

## Verdict (fill in)

* **Gate** (B M==1 `mm` ≤ 760.0 **and** decode ≥ A at all three G **and**
  text = A): ______.
* **Accuracy gate (a)**: ______.
* **Prefill (standing)**: ______.
* **Anchor** (rules 30 / 32) and **in-situ engine ≥ 28 GB/s**: ______.
* **A vs A0** (#113's confirmation): ______ → BENCHMARK's "now" moves to A.

## Notes from the run

<thermal, first-run page faults, FARF/AEE errors, anything stale, the
rebuilt md5s if `$W` was absent (rule 22), deviations from the steps>
