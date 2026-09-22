# Measurement 105: the M=1 GEMV's compute side — one-row loop and l2fetch lead (LEDGER ㉒)

Branch `htp/105-m1-gemv-compute` (code head `a0ae8b29`, this file on the
same branch) — estimated device time: **≈ 50 min** (≈ 40 min if the
short form in step 3 is taken)

Plan: `docs/plans/105-m1-gemv-compute.md` §4 step 5. Issue #105 (tracker
#76; LEDGER ㉒ compute half, ⑯). Contract
`docs/plans/0001-htp-moe-decode-agent-system.md`.

**Run right after the handoff-88 sitting, same phone: the NPU model it
installed is reused** (step 1 checks its md5 and skips the 4.3 GB push).
Only the NPU model `q40-qs4cx-wh` runs here; no CPU `q40` cell.

## Why

On the M=1 GEMV path (#94 sitting 2 variant C) the level-2 M==1 row's
`mm` is 975 µs/call: 22.0 MB of arena weight at 22.6 GB/s, with the pool's
lanes busy ≈ 96 % of it. At m = 1 the kernel ran the four-row loop — 32
`vrmpy` per k-tile for 8 that are stored, over four rotating accumulators
(the V79 HVX PRM §5.6 accumulator stall) — and each column's `l2fetch`
went out right before its own loads. **Decision that hangs on this
sitting:** B's `mm` below A's by more than 5 % with decode ≥ A at every G
= pass; `mm` ≤ 600 µs closes ㉒; the microbench says how much of what is
left is feed (㉒'s DMA/VTCM half, #100) versus compute.

Variants (4; one app set, **only the skel changes**):

* **A** — skel from `htp_moe` @ `48fd2420` (the head this branch forks
  from; its DSP sources = the #94 C path). Run first.
* **B1** — one-row loop (`gemm_row1`), per-column `l2fetch` as before
  (`-DHVX_GEMV_PF_LEAD_KB=0u`).
* **B2** — B1 + the next block's `l2fetch` box issued a block ahead, 64 KB
  lead (the branch default).
* **B3** — B1 + 192 KB lead (where L2 eviction would show, as B3 < B2).

`NNTR_MOE_HTP_M1_GEMV=1` is set **explicitly in every cell, A included**
(it resolves to on whether or not #101 has landed). Every log must print
`[HTP] moe m1 gemv: on (applied=0x1)` (a `source=…` suffix is fine) and
every level-2 M==1 row `blocks=0 m1_gemv=1408/1408`, or the run is void.

The ARM side is untouched: the branch's diff against `htp_moe` is
`nntrainer/tensor/htp_backend/{hvx/hvx_gemm_u8i4_wh.{c,h},
hmx/hexkl_mm_u8i4_moe.{c,h}}` (DSP only; `htp_backend/meson.build`
compiles neither into `libnntrainer.so`), `test/htp/host/*` and the
gtest. So one app set serves all four variants, and the generated text
must be byte-identical across A/B1/B2/B3 per G and run (the int32 is
unchanged by construction; see Gates). `NNTR_L2_DIFF` n/a for
`QS4CX_WH`; text = CPU `q40` n/a (different weights) — the accuracy
column is "= A".

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

Staged under **`W=/local/mnt/workspace/htp_moe/105/`**; `$W/md5.txt` is the
`md5sum` of every staged file and is reproduced here. The B skels and the
app were built in `/home/j2z0-lee/nntrainer-88-A` on this branch (skels
@ `a0ae8b29`), A's skel in the detached worktree
`/home/j2z0-lee/nntrainer-105-A` @ `48fd2420`. Skel builds are not
bit-reproducible here (two builds of B2's source gave two md5s), so the
**staged file's md5 is the identity**; the variant flag was confirmed in
each binary by disassembly (B1: workers call `hvx_gemm_u8i4_wh_col`; B2/B3:
`_col_nopf` + `_prefetch`, lead constant `0x10000` vs `0x30000`).

| file (staged as) | md5 | built from |
|---|---|---|
| `$W/libnntr_hvx_skel.A.so` | `9bcb81e496800215a2c73a7f865328b8` | `htp_moe` @ `48fd2420`, `HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh`; `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK (46 runtime imports)` |
| `$W/libnntr_hvx_skel.B1.so` | `a3f5d43175177eb19abf70a19d5a52b2` | `a0ae8b29`, `HEX_EXTRA_CFLAGS=-DHVX_GEMV_PF_LEAD_KB=0u`; clean, `UNDEFINED SYMBOLS OK (46)` |
| `$W/libnntr_hvx_skel.B2.so` | `18eeb1c84fe8d068368a623372195dea` | `a0ae8b29`, default (64 KB); clean, `UNDEFINED SYMBOLS OK (46)` |
| `$W/libnntr_hvx_skel.B3.so` | `1e08fd1032e6998c94b02e1c3f548738` | `a0ae8b29`, `HEX_EXTRA_CFLAGS=-DHVX_GEMV_PF_LEAD_KB=192u`; clean, `UNDEFINED SYMBOLS OK (46)` |
| `$W/app/nntrainer_causallm` | `01bd38f6af041e7d3d028b485bf40193` | this branch, `build_android.sh --htp` (+ `ninja -C builddir install`, then `--htp --cache`); `jni/libs/arm64-v8a/`. Same md5 as #88's A file (unchanged sources, deterministic ndk-build) |
| `$W/app/libcausallm_core.so` | `0a700c028c3fede5990a95e245716b51` | same; == #88 A |
| `$W/app/libnntrainer.so` | `ddc61f2945d4e331376f3e991bac67f0` | same (`jni/obj/local/arm64-v8a/`); `readelf -d` NEEDED `libsdkl.so`, `libcdsprpc.so`. Carries PR #103 (it is `htp_moe` post-merge), so it differs from #88's A `341d5762…` and, by build path (rule 14), from #88's B `4f6c90dd…` |
| `$W/app/libccapi-nntrainer.so` | `571c0ff2168dc6b82381a1e393895981` | same; == #88 A |
| `$W/app/libc++_shared.so`, `$W/gtest/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot (== #77 / #88 / #94) |
| `$W/app/libsdkl.so`, `$W/gtest/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` (== #77 / #88 / #94) |
| `$W/gtest/unittest_hvx_mm_u8i4` | `e498d6c2a283b7bb2f570bb2c32d10dc` | `ndk-build … unittest_hvx_mm_u8i4` on this branch (`test/jni/obj/local/arm64-v8a/`); has `MoeLayerM1GemvMatchesHmx` and the new `MoeM1GemvFeedVsCompute` |
| `$W/prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | copied from `/local/mnt/workspace/htp_moe/94/prompt512.txt` (the #88 / #94 prompt) |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (4316133120 B) | `7b7867fab51845664c0050c0a837073e` | #78, `--moe_dtype QS4CX_WH` (NPU model; on the phone since the #88 sitting) |
| `tokenizer.json` (`q40-qs4cx-wh/`) | `7b8067a580173d3eb1697afae3b456f5` | HF |

Not staged, never pushed: `builddir/.../libcdsprpc.so` (rule 5).

Workstation sanity before pushing:

```
W=/local/mnt/workspace/htp_moe/105
(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')           # 0
strings $W/app/nntrainer_causallm | grep -c 'per-layer-type totals'  # 0 (not a profile binary)
find $W -name 'libcdsprpc*' | wc -l                                 # 0
```

### If `$W` is not on the workstation you measure from

The #88 sitting ran on a workstation without the staging directory. Rebuild
the set from the commits instead; skel builds are not byte-reproducible, so
the md5s will differ from the table. Record the ones you push under "Notes
from the run" (rules 14 and 21). Build from a clean checkout, NDK r30,
SDK 6.4.0.1, HexKL 6.4.0.1 (`source tools/htp/env.sh` or the same exports).

```
W=/local/mnt/workspace/htp_moe/105; mkdir -p $W/app $W/gtest
git fetch origin && git worktree add /tmp/n105A 48fd2420 && git worktree add /tmp/n105B a0ae8b29
(cd /tmp/n105A && HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh && cp test/htp/build/libnntr_hvx_skel.so $W/libnntr_hvx_skel.A.so)
for v in B1:0u B2:64u B3:192u; do
  (cd /tmp/n105B && HEX_EXTRA_CFLAGS=-DHVX_GEMV_PF_LEAD_KB=${v#*:} HEXKL_SDK_VER=6.4.0.1 ./test/htp/build.sh \
     && cp test/htp/build/libnntr_hvx_skel.so $W/libnntr_hvx_skel.${v%%:*}.so)
done                                                    # each build prints UNDEFINED SYMBOLS OK (46 runtime imports)
# One app set serves A and every B (this PR changes no ARM source). Build it from a0ae8b29:
(cd /tmp/n105B/Applications/CausalLM && ./build_android.sh --htp)   # fresh builddir: see the hexagon-gates skill, rung 3
cp /tmp/n105B/Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so} $W/app/
cp /tmp/n105B/Applications/CausalLM/jni/obj/local/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so,libc++_shared.so} $W/app/
strings $W/app/libnntrainer.so | grep -c 'staging: act'           # 1: PR #103 is in the app (else stale --cache: ninja install, rebuild)
# gtest: ndk-build unittest_hvx_mm_u8i4 in /tmp/n105B/test/jni (rung 3), copy it and libc++_shared.so to $W/gtest/
# libsdkl.so into both dirs from $HEXKL_ROOT/lib/6.4.0.1/armv8_android26/; prompt512.txt = docs/measurements/77-prompt512.txt
(cd $W && md5sum libnntr_hvx_skel.*.so app/* gtest/* prompt512.txt > md5.txt)
```

## Steps (workstation, phone on USB)

Shell setup once:

```
cd /home/j2z0-lee/nntrainer-88-A && git fetch && git checkout htp/105-m1-gemv-compute && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/105; mkdir -p $W/logs
D=/data/local/tmp/nntrainer/causallm; T=/data/local/tmp/htp_u8i4_layer_test
therm() { adb shell dumpsys battery | grep -E 'level|temperature'; adb shell cat /sys/class/thermal/thermal_zone0/temp; }
sk() { # sk <A|B1|B2|B3>: the variant's skel into both dirs, device md5 logged
  adb push $W/libnntr_hvx_skel.$1.so $D/libnntr_hvx_skel.so >/dev/null && \
  adb push $W/libnntr_hvx_skel.$1.so $T/libnntr_hvx_skel.so >/dev/null && \
  adb shell "md5sum $D/libnntr_hvx_skel.so $T/libnntr_hvx_skel.so" | sed "s/^/skel $1: /" | tee -a $W/logs/skel_push.log
}
```

### 0. Device state (1 min)

```
adb devices          # exactly one device; record the serial under Notes (any S25 Ultra, contract §4.2)
therm                # checkpoint 0: battery %, temperature (tenths of °C), thermal_zone0 (m°C)
```

Screen off, charger in. If the phone is warm from the #88 sitting, wait
until `thermal_zone0` is back near the value #88's checkpoint 1 read.

### 1. Install (≈ 3 min) — model reused from #88

```
adb shell ls -l $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin     # 4316133120
adb shell md5sum $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin    # 7b7867fab51845664c0050c0a837073e -> no push
# only if either line differs:  adb push /local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh $D/models/q40-qs4cx-wh
adb shell mkdir -p $D/models $T
adb push $W/app/nntrainer_causallm $W/app/libcausallm_core.so $W/app/libnntrainer.so $W/app/libccapi-nntrainer.so $W/app/libc++_shared.so $W/app/libsdkl.so $W/prompt512.txt $D/
adb push $W/gtest/unittest_hvx_mm_u8i4 $W/gtest/libc++_shared.so $W/gtest/libsdkl.so $T/
adb shell "chmod 755 $D/nntrainer_causallm $T/unittest_hvx_mm_u8i4"
sk A
```

(#88 leaves its own A or B app on the phone: the push above replaces all
four files, which is why the md5 line below matters.)

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
adb shell "md5sum $D/nntrainer_causallm $D/libnntrainer.so $D/libcausallm_core.so $D/libccapi-nntrainer.so $T/unittest_hvx_mm_u8i4 $D/prompt512.txt $D/models/q40-qs4cx-wh/tokenizer.json"
```

### 2. Proof + microbench per skel (≈ 12 min) — one level-2 run and the two gtests, A first

One helper per variant: skel swap, a level-2 profile run at G = 64 (never
read for tok/s), then the gtests with the thermal zone around them (rule
18).

```
prof() { # prof <A|B1|B2|B3>
  sk $1
  adb shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/q40-qs4cx-wh/nntr_config.json && \
    NNTR_MOE_HTP_M1_GEMV=1 NNTR_HTP_PROFILE=2 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/prof_$1.log | grep -E 'moe m1 gemv|level=|K=2048 N=2048' | cut -c1-420
  adb shell cat /sys/class/thermal/thermal_zone0/temp | sed "s/^/therm before gtest $1: /" | tee -a $W/logs/therm.log
  adb shell "cd $T && md5sum libnntr_hvx_skel.so unittest_hvx_mm_u8i4 && NNTR_MOE_HTP_M1_GEMV=1 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_mm_u8i4 --gtest_filter='*MoeLayerM1GemvMatchesHmx*:*MoeM1GemvFeedVsCompute*'" 2>&1 | tee $W/logs/gtest_$1.log | grep -E 'libnntr_hvx_skel|U8I4_FIELD|PASSED|FAILED|SKIPPED'
  adb shell cat /sys/class/thermal/thermal_zone0/temp | sed "s/^/therm after gtest $1: /" | tee -a $W/logs/therm.log
}
therm                                                          # checkpoint 1
prof A; prof B1; prof B2; prof B3
therm                                                          # checkpoint 2
```

Expected per variant:

* `prof_<v>.log`: one `[HTP] moe m1 gemv: on (applied=0x1)`; the header
  `[HTP-PROFILE] level=2 qos_mode=2 …` (**`qos_mode=2` in all four**; a
  `1` voids that profile); the M==1 row `K=2048 N=2048 M==1 calls=1408 …
  mm <µs> … blocks=0 m1_gemv=1408/1408]`; the M>1 row with
  `m1_gemv=0/<calls>` (prefill stays on the HMX loop).
* `gtest_<v>.log`: the md5 line of the skel equals the table's row for
  `<v>`; `U8I4_FIELD path=moe_m1_gemv field=bad_elems_M1 value=0 of 2048`,
  `field=bad_elems_M4 value=0 of 8192`, `field=bit_identical value=yes`;
  three `U8I4_FIELD path=m1_bench cell=arena|heap|hot mm_us=… mm_min_us=…
  lane_us=… lanes=… tiles=… ns_per_tile=… gbps=…` lines (no `INVALID`);
  `[  PASSED  ] 2 tests.`

A `bad_elems` ≠ 0 under any B skel is a failed accuracy gate: stop, note
it, skip that variant's E2E cells. A `MoeM1GemvFeedVsCompute` failure on
the `hot` cell alone (it routes one weight pair to all four experts, a
first for this call) does not void the variant: note the message and
continue.

### 3. E2E cells (≈ 30 min) — mirrored order per G

```
run() { # run <A|B1|B2|B3> <G> <run#>   (the variant's skel must already be on the device)
  adb shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' models/q40-qs4cx-wh/nntr_config.json && \
    grep num_to_generate models/q40-qs4cx-wh/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    NNTR_MOE_HTP_M1_GEMV=1 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/$1_G$2_r$3.log | grep -E '^(prefill|generation|total|peak memory)|moe m1 gemv|libnntr_hvx_skel'
}
cell() { sk $1 >/dev/null; run $1 $2 $3; }
for g in 64 512 1024; do
  cell A $g 1; cell B1 $g 1; cell B2 $g 1; cell B3 $g 1
  cell B3 $g 2; cell B2 $g 2; cell B1 $g 2; cell A $g 2
  therm                                                        # checkpoints 3, 4, 5
done
```

Each log carries the device md5 of the skel it ran (the `md5sum` in
`run`); a cell whose md5 is not its variant's table row is void (rule
21). Expected lines in every log: `prefill: 512 tokens, … TPS`,
`generation: <G> tokens, … TPS`, `total: … ms`, `peak memory: … KB`, one
`[HTP] moe m1 gemv: on (applied=0x1)`, and **no** `[HTP-PROFILE]` block.
`prefill: N` with N ≠ 512 voids the prefill column; `generation: N` ≠ G
is a failed config edit.

**Short form (≈ 40 min total), if time runs out:** keep A and B2 at all
three G; B1 and B3 at G = 64 only (the profile + gtest from step 2 stay).

### 4. Checks on the workstation (1 min)

```
grep -c 'moe m1 gemv: on (applied=0x1)' $W/logs/*_G*_r*.log $W/logs/prof_*.log   # 1 each
grep -l 'HTP-PROFILE' $W/logs/*_G*_r*.log                                        # nothing
for v in A B1 B2 B3; do printf '%s: ' $v; grep 'M==1' $W/logs/prof_$v.log | grep -o 'blocks=[0-9]* m1_gemv=[0-9/]*'; done   # blocks=0 m1_gemv=1408/1408 x4
for g in 64 512 1024; do for r in 1 2; do for v in B1 B2 B3; do
  printf '%s vs A G=%s r%s: ' $v $g $r
  diff <(sed -n '/^=====/q;p' $W/logs/A_G${g}_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel') \
       <(sed -n '/^=====/q;p' $W/logs/${v}_G${g}_r${r}.log | grep -v 'moe m1 gemv\|libnntr_hvx_skel') >/dev/null && echo same || echo DIFFERENT
done; done; done
cat $W/logs/skel_push.log $W/logs/therm.log
```

Then fill the tables below, commit this file on the same branch, push,
set #105 to `state:measured`.

## Results (fill in)

Unit (serial from `adb devices`): ______  Battery / °C / thermal_zone0 at checkpoints 0–5: ______

Proof and kernel columns (`[HTP-PROFILE]` level 2, G = 64, one run each;
µs/call; the verdict reads `mm` and `dsp`):

| variant | skel md5 (device) | qos_mode | M==1 `mm` | M==1 `dsp` | M==1 `swiglu` | M==1 host / transport | blocks / m1_gemv | M>1 `dsp` | M>1 m1_gemv |
|---|---|---|---|---|---|---|---|---|---|
| A | | | | | | | | | |
| B1 | | | | | | | | | |
| B2 | | | | | | | | | |
| B3 | | | | | | | | | |

Gtests (per skel; `thermal_zone0` before → after):

| variant | bad_elems_M1 | bad_elems_M4 | bit_identical | therm (m°C) | arena mm / min / lane_us / ns_per_tile / gbps | heap mm / min / lane_us / ns_per_tile / gbps | hot mm / min / lane_us / ns_per_tile / gbps |
|---|---|---|---|---|---|---|---|
| A | | | | | | | |
| B1 | | | | | | | |
| B2 | | | | | | | |
| B3 | | | | | | | |

E2E (NPU model, `NNTR_MOE_HTP_M1_GEMV=1`, prompt 512, 8 threads):

| variant | gen | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS (KB) | text = A (same G, run)? | skel md5 (device, from the log) |
|---|---|---|---|---|---|---|---|---|
| A | 64 | 1 | | | n/a (#89) | | reference | |
| B1 | 64 | 1 | | | n/a | | | |
| B2 | 64 | 1 | | | n/a | | | |
| B3 | 64 | 1 | | | n/a | | | |
| B3 | 64 | 2 | | | n/a | | | |
| B2 | 64 | 2 | | | n/a | | | |
| B1 | 64 | 2 | | | n/a | | | |
| A | 64 | 2 | | | n/a | | reference | |
| A | 512 | 1 | | | n/a | | reference | |
| B1 | 512 | 1 | | | n/a | | | |
| B2 | 512 | 1 | | | n/a | | | |
| B3 | 512 | 1 | | | n/a | | | |
| B3 | 512 | 2 | | | n/a | | | |
| B2 | 512 | 2 | | | n/a | | | |
| B1 | 512 | 2 | | | n/a | | | |
| A | 512 | 2 | | | n/a | | reference | |
| A | 1024 | 1 | | | n/a | | reference | |
| B1 | 1024 | 1 | | | n/a | | | |
| B2 | 1024 | 1 | | | n/a | | | |
| B3 | 1024 | 1 | | | n/a | | | |
| B3 | 1024 | 2 | | | n/a | | | |
| B2 | 1024 | 2 | | | n/a | | | |
| B1 | 1024 | 2 | | | n/a | | | |
| A | 1024 | 2 | | | n/a | | reference | |

Verdict (plan §1), per B variant against A of this sitting:

* **pass** = level-2 M==1 `mm` < A's × 0.95 **and** decode mean ≥ A's at G
  = 64, 512 and 1024; **closes ㉒** if `mm` ≤ 600 µs/call.
* accuracy: text = A at every G and run, `bit_identical value=yes`.
* prefill: prompt-512 tok/s ≥ −5 % of A; a single cell below that with
  the M>1 `dsp` within 5 % of A's is noise (A's own spread was ±13 % in
  #94 s2).
* microbench (plan §3.4): `hot` ≈ `arena` ns/tile under A → issue-bound,
  B1 is the lever; `hot` ≪ `arena` under B1/B2 → the rest is feed, and
  `arena` − `hot` is ㉒ feed half's budget; `heap` ≪ `arena` → the ION
  mapping itself is the feed problem.
* PR default after read-back: the better of B2/B3, 64 if they are within
  ±5 %, 0 if B1 beats B2 by more than the noise.

Reference (#94 sitting 2, unit `R3CY205ZMND`, `htp_moe` @ `2a75f7d9`,
variant C = this A's path): level-2 M==1 `mm` 974.7, `dsp` 1044.0,
host 1792.2, transport 748.2 µs/call, `blocks=0 m1_gemv=1408/1408`; M>1
`dsp` 16554.3; decode 18.83 / 18.33 / 17.59 tok/s (G 64 / 512 / 1024,
means of two), prefill 389–527 (A of that sitting). Goal ≥ 50 decode
tok/s, prefill ≥ −5 % of A.

## Gates passed on the workstation (host evidence, not device)

* `bash test/htp/host/run_host_checks.sh` (after `source tools/htp/env.sh`):
  `M1 GEMV PATH BIT-IDENTICAL TO HMX PATH (M=1,2,4; tiny+real)` and
  `M1 GEMV PREFETCH LEAD COVERS EVERY COLUMN (lanes=6; lead 0|64|192 KB)`
  with `ALL CHECKS PASS` for each of the three leads; `WORKER POOL LANES
  OK`; `HVX GEMV NATIVE cases=1152 bad=0` / `HVX GEMV NATIVE BIT-IDENTICAL
  (libnative; m=1..16 rows1+rows4)` — the real kernel on x86 against the
  SDK's HVX emulation; mutants caught: `gemm_row1` shift `bad=9216` (only
  the lone-row cases), `gemm_rows4` shift `bad=82944`.
* `ninja -C build`; `unittest_causallm_models --gtest_filter='*Lfm2Moe*'`
  6/6 PASSED (fixture generated); `*qs4cx*` 2/2; `tools/htp_syntax_check.sh`
  exit 0.

## Notes from the run

<serial, thermal at checkpoints, the provenance and skel_push md5 lines,
first-run page faults, FARF/AEE errors, anything stale. A `qos_mode=1`
header, an `AEE_EBADPARM (0x8000040E)`, a `0x80000406` open failure or a
`nntr_hvx_moe_set_opts failed` line goes here with the exact text.>
