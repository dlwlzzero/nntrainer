# Measurement 157 (step 2): the decode MoE call's experts split between the DSP and the CPU, bit for bit

Branch `htp/157-expert-split-impl`, code @ `c515583c` (the set below was
built from that tree; later commits on the branch are docs only). Plan
`docs/plans/157-expert-split.md` (section 4 step 7). Estimated device time:
**≈ 50 min**. Agents never run `adb`; the user (or the orchestrator on the
user's request) runs this top to bottom.

## Why

1. Speed: today a decode MoE call reads four experts (22 MB) on the DSP
   while the CPU waits (≈ 0.73 ms of a ≈ 25 ms token, ×22). With
   `NNTR_MOE_HTP_SPLIT=k` the DSP keeps the first k active experts (in
   ascending id) and the CPU computes the others at the same time from the
   same arena, mapped cached. The model predicts +30 to +42 % decode for
   k = 2 (plan section 3); the lever gate is **S2 ≥ A + 20 % at G = 512**.
2. Accuracy, under the bit-preserving rule: the CPU continues the DSP's
   partial sum in the DSP's order with the DSP's arithmetic
   (`moe_m1_det.h`), so every MoE dump must be `bit_identical=1` against A,
   the text identical to A on all 8 prompts and in every tok/s cell, and the
   `[PPL] decode` lines byte-identical to A's. Any difference is a defect,
   not an approval question. The host shows it on hvx_emu (`MOE SPLIT
   BIT-IDENTICAL k=0..4`), in the in-process build (`E2E eval
   split1-tiny / split0-tiny / split1-lfm25 … bit_identical=1`) and, once,
   against hexagon-sim running the skel sources (0 mismatches at every
   stage); none of that is silicon. The gtest of step 2 is the first
   silicon check and gates everything after it.

### Variants (4, one binary set, environment only; every cell also sets `NNTR_OP_TIME=1 NNTR_NUM_THREADS=8`)

| | env | expected banners (stderr) |
|---|---|---|
| **A** (reference, first) | none | `[HTP] dspq: on …` once; **no** `arena: cached`, **no** `moe split` line; at exit `[HTP] moe m1 split=4 calls=N split_calls=0 …` |
| **C** | `NNTR_HTP_ARENA_CACHED=1` | `dspq: on` once; `[HTP] arena: cached, one D-cache clean per weight (…)` once; no `moe split` line; exit line `split=4 … split_calls=0` |
| **S2** | `NNTR_MOE_HTP_SPLIT=2` | `dspq: on` once; `arena: cached` once; `[HTP] moe split: k=2 cpu_threads=8 arena=cached` once; exit line `split=2 calls=N split_calls=N` |
| **S1** | `NNTR_MOE_HTP_SPLIT=1` | the same with `k=1` / `split=1` |

C isolates the cache attribute (plan section 5's risks: stale lines, a
slower DSP read through the cached mapping). A log without its banners —
or with `[HTP] moe split: k=… asked for, but a call has …` — is **void**,
never "at A's speed" (rule 36). The exit line is the split's timer:
`call_us` / `calls` is the MoE call's wall, `cpu_us` / `split_calls` the
CPU experts, `dsp_wait_us` what the caller then still waited (the DSP's
remaining time plus the merge). `NNTR_OP_TIME` itself (#150, PR #153) is
not in this tree; the variable only switches this line on for A and C.

## Artifacts (`/local/mnt/workspace/htp_moe/157/set/`, `md5.txt` next to it)

| file | md5 | built with |
|---|---|---|
| `libnntr_hvx_skel.so` | `37468a7fdbf2e469589849598ca860ff` | A's, copied from `/local/mnt/workspace/htp_moe/150/set/` (the #150 / #90 set); this PR changes no DSP source |
| `nntrainer_causallm` | `1f5d632ceed02c04b68b0ea618e5d990` | `build_android.sh --htp --cache` (`jni/libs/arm64-v8a/`) |
| `libcausallm_core.so` | `da9f49916d9821c18b617f94372ef3a1` | same (`NNTR_HTP_FORWARD_KINDS` count 2) |
| `libnntrainer.so` | `3bdf872465dca83e495a4b9f28eefe9a` | same (`jni/obj/local/arm64-v8a/`; NEEDED `libsdkl.so`, `libcdsprpc.so`; `htp_moe_cpu.o`: 0 fmla / fmls / fmadd / fmsub, 64 sdot) |
| `libccapi-nntrainer.so` | `231dbd74f03f2d1e09116ab85970a19f` | same (`jni/obj/local/arm64-v8a/`) |
| `unittest_hvx_mm_u8i4` | `2e19d6ef9aaa8e14f118a035d552aadb` | `test/jni` ndk-build (`-march=armv8.2-a+dotprod`; `neon_sdot` string present) |
| `libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 (from `150/set/`) |
| `libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL 6.4.0.1 (from `150/set/`) |
| `prompt512.txt` (p01) | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt`, 512 tokens (from `141b/set/`) |
| `bitset-02-code.txt` … `bitset-08-short.txt` (p02–p08) | see `md5.txt` / `docs/measurements/prompts/README.md` | 207 / 117 / 326 / 207 / 276 / 402 / 24 tokens (from `141b/set/`) |
| model `q40-qs4cx-wh`, `tokenizer.json` | on the phone since #100 | not pushed |

Not staged, never pushed: `builddir/.../libcdsprpc.so` (rule 5). No DSP
source, IDL or stub changed in this PR, so the skel is A's from the #150
set, unchanged.

Workstation sanity before pushing:

```
W=/local/mnt/workspace/htp_moe/157
(cd $W/set && LC_ALL=C md5sum -c ../md5.txt | grep -vc ': OK$')           # 0
strings $W/set/libnntrainer.so | grep -c 'dspq: on'                       # 1
strings $W/set/libnntrainer.so | grep -c 'moe split: k=%d'                # 2 (banner, refusal)
strings $W/set/libcausallm_core.so | grep -c NNTR_HTP_FORWARD_KINDS       # 2 (rule 36)
strings $W/set/nntrainer_causallm | grep -c 'per-layer-type totals'       # 0 (not a profile binary)
find $W -name 'libcdsprpc*' | wc -l                                       # 0
```

Rebuild recipe if `$W` is not on the measuring workstation: `git checkout
c515583c`, `source tools/htp/env.sh`, `export
HEXKL_ROOT=/home/j2z0-lee/Qualcomm/hexkl-1.0-beta.2/hexkl_addon
HEXKL_SDK_VER=6.4.0.1`, `git submodule update --init --depth 1`, copy
`Applications/CausalLM/lib/libtokenizers_android_c.a` from another
worktree, `(cd Applications/CausalLM && ./build_android.sh --htp)` (on a
fresh `builddir`: `cd builddir && meson configure
-Dprefix=$PWD/android_build_result && ninja install`, then `--htp
--cache`), `ln -sfn $PWD/subprojects/googletest/googletest
test/jni/googletest`, `./test/htp/build.sh` (for `test/htp/generated/`,
the gtest's stub; its skel is not staged), then the gtest's `ndk-build`
line from `.claude/skills/hexagon-gates` rung 3. `libc++_shared.so` and
`libsdkl.so` come from `/local/mnt/workspace/htp_moe/150/set/`. A rebuilt
set does not match the table: record the md5s you push. `ninja -C builddir`
re-runs ndk-build only when a listed `.cpp` changed, not a header: after a
change to `moe_m1_det.h` alone, `touch
nntrainer/tensor/htp_backend/htp_moe_cpu.cpp` first (it cost one rebuild
while staging this set).

## Steps (workstation, phone on USB)

Shell setup once. Every `adb` names the serial. Clean run dir `s157`.

```
cd /home/j2z0-lee/nntrainer-157b && git fetch -q && git checkout htp/157-expert-split-impl && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/157; mkdir -p $W/logs $W/dump
S=R3CY10WM83Y      # adb devices: record the serial you use under Notes
D=/data/local/tmp/nntrainer/causallm/s157; M=../models/q40-qs4cx-wh; DD=/data/local/tmp/s157dump
therm() { adb -s $S shell dumpsys battery | grep -E 'level|temperature'; adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp; }
env_of() { case $1 in C) echo "NNTR_HTP_ARENA_CACHED=1";; S2) echo "NNTR_MOE_HTP_SPLIT=2";; S1) echo "NNTR_MOE_HTP_SPLIT=1";; *) echo "";; esac; }
run() { # run <A|C|S2|S1> <G> <log name> <prompt file> [extra env]
  adb -s $S shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    $(env_of $1) $5 NNTR_OP_TIME=1 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat $4)\"" \
    2>&1 | tee $W/logs/$3.log | grep -E '^(prefill|generation|total|peak memory)|dspq: on|arena: cached|moe split|moe m1 split|libnntr_hvx_skel'
}
# The generated text of a log: everything before the summary, minus every
# [HTP] line (the split banner is printed at the first decode call, so it
# can land inside the streamed text; perl removes it with its newline).
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate'; }
```

### 0. Device state (1 min)

```
adb -s $S get-state                   # device (every adb below names -s $S)
therm | tee -a $W/logs/therm.log      # checkpoint 0 (battery %, °C·10, zone0 m°C)
```

Screen off, charger in, cool start.

### 1. Install (≈ 2 min) — model reused

```
adb -s $S shell ls -l /data/local/tmp/nntrainer/causallm/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin   # 4316133120
adb -s $S shell "rm -rf $D $DD; mkdir -p $D $DD"
adb -s $S push $W/set/. $D/
adb -s $S shell "chmod 755 $D/nntrainer_causallm $D/unittest_hvx_mm_u8i4"
adb -s $S shell "cd $D && md5sum *" | tee $W/logs/md5_device.log     # must equal ../md5.txt, line by line
diff <(sort -k2 $W/md5.txt | tr -d '\r') <(sort -k2 $W/logs/md5_device.log | tr -d '\r') && echo MD5 OK
```

Config (as #141: greedy, `init_seq_len: 512`, `moe_engine: htp`,
`moe_htp_layers: ""`), re-applied unconditionally:

```
adb -s $S shell "cd $D/$M && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers' nntr_config.json"
```

Expected echo: `do_sample": false`, `bad_word_ids": [124900]`,
`init_seq_len": 512`, `moe_engine": "htp"`, `moe_htp_layers": ""`.

### 2. Silicon bit check FIRST (≈ 2 min) — stop here if any `bad` ≠ 0

```
adb -s $S shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_mm_u8i4 --gtest_filter='HmxMmU8I4Layer.MoeM1CpuSplitMatchesDsp'" \
  2>&1 | tee $W/logs/gtest_split.log | grep -E 'moe_m1_split|OK \]|FAILED|PASSED'
for m in 0 1; do
  adb -s $S shell "cd $D && NNTR_HTP_ARENA_CACHED=$m LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_mm_u8i4 --gtest_filter='HmxMmU8I4Layer.ArenaMapAndDma:HmxMmU8I4Layer.ArenaUncachedWriteAfterMap:HmxMmU8I4Layer.MoeLayerFromArenaMatchesHeap'" \
    2>&1 | tee $W/logs/gtest_arena_c$m.log | grep -E 'field=(cached|checksum_ok|bad_elems)|OK \]|FAILED|PASSED'
done
```

Expected: `field=cpu_path value=neon_sdot` (`scalar` means the gtest was
built without dotprod: void, rebuild), `bad_elems_k0` … `bad_elems_k4`,
`bad_elems_expert0` … `expert3` and `bad_elems` all `value=0`, `[  PASSED  ]
1 test`. The arena runs: `cached value=n` then `y`, `checksum_ok value=yes`,
`path=arena_moe field=bad_elems value=0 …`, `[  PASSED  ] 3 tests` twice.
**Any `bad` ≠ 0 or a FAILED: stop, record the fields, do not run step 3 on.**
The per-expert fields name the expert, the per-k fields the split point.

### 3. Sanity, one short run each (≈ 2 min)

```
for v in A C S2 S1; do run $v 8 sanity_$v prompt512.txt; done
grep -c 'arena: cached\|moe split' $W/logs/sanity_A.log                                  # 0
grep -h 'arena: cached\|moe split\|moe m1 split' $W/logs/sanity_{C,S2,S1}.log
```

Expected banners as the variants table; the exit lines' `calls`
at G = 8 is 176 (22 per generated token, as #141), S2 / S1
`split_calls` = `calls`.

### 4. tok/s, prompt 512 (≈ 15 min) — mirrored per G

```
for g in 64 512 1024; do
  run A  $g A_G${g}_r1  prompt512.txt; run C  $g C_G${g}_r1  prompt512.txt
  run S2 $g S2_G${g}_r1 prompt512.txt; run S1 $g S1_G${g}_r1 prompt512.txt
  run S1 $g S1_G${g}_r2 prompt512.txt; run S2 $g S2_G${g}_r2 prompt512.txt
  run C  $g C_G${g}_r2  prompt512.txt; run A  $g A_G${g}_r2  prompt512.txt
  therm | tee -a $W/logs/therm.log    # checkpoints 1, 2, 3
done
```

Expected in every log: `prefill: 512 tokens, … TPS`, `generation: <G>
tokens, … TPS`, `generation(last 64): 64 tokens, … TPS`, `peak memory`,
the variant's banners, its exit line with `calls` = 22 × G.

### 5. Dumps, prompt 512, G = 64 (≈ 8 min, ≈ 1 GB on the phone) — gate (c)

```
for v in A1 C S2 S1 A2; do
  adb -s $S shell "rm -rf $DD/$v && mkdir -p $DD/$v"
  run ${v%[12]} 64 dump_$v prompt512.txt NNTR_HTP_DUMP=$DD/$v
  rm -rf $W/dump/$v && adb -s $S pull $DD/$v $W/dump/$v > /dev/null && adb -s $S shell "rm -rf $DD/$v"
done
therm | tee -a $W/logs/therm.log      # checkpoint 4
E="python3 tools/htp/htp_dump_eval.py"
$E --label a2 $W/dump/A1 $W/dump/A2 | tail -1      # null check: must be bit_identical=1
for v in C S2 S1; do $E --label $v $W/dump/A1 $W/dump/$v | tail -1; done
wc -l $W/dump/A1/manifest.txt $W/dump/S2/manifest.txt  # equal (the manifest records the full routing)
```

`a2` not `=1` means the sitting cannot judge the dumps (A not bit-stable
run to run): record it and go on. `C` / `S2` / `S1` `bit_identical=0` is a
defect: record the `first_diff` line and stop before step 6.

### 6. Decode PPL at G = 512 (≈ 3 min)

```
adb -s $S shell "rm -f $D/cont.ids"
run A  512 ppl_A_self prompt512.txt NNTR_PPL_DECODE=cont.ids     # writes A's continuation (source=self)
run A  512 ppl_A      prompt512.txt NNTR_PPL_DECODE=cont.ids     # forced on it (null check)
run C  512 ppl_C      prompt512.txt NNTR_PPL_DECODE=cont.ids
run S2 512 ppl_S2     prompt512.txt NNTR_PPL_DECODE=cont.ids
run S1 512 ppl_S1     prompt512.txt NNTR_PPL_DECODE=cont.ids
for v in C S2 S1; do printf '%s: ' $v
  cmp -s <(grep '^\[PPL\] decode' $W/logs/ppl_A.log) <(grep '^\[PPL\] decode' $W/logs/ppl_$v.log) && echo identical || echo DIFFERENT; done
grep -h '^\[PPL\] decode tokens' $W/logs/ppl_*.log
```

Every `[PPL] decode` line (511 step lines with `%.17g` NLLs and the
summary) of C / S2 / S1 must equal A's forced run; A's self and forced
summaries must agree on `nll_sum`.

### 7. Text set, 8 prompts at G = 64 (≈ 10 min)

```
P="prompt512.txt bitset-02-code.txt bitset-03-math.txt bitset-04-korean.txt bitset-05-json.txt bitset-06-dialogue.txt bitset-07-facts.txt bitset-08-short.txt"
i=0; for p in $P; do i=$((i+1)); for v in A C S2 S1; do run $v 64 text_${v}_p0$i $p; done; done
therm | tee -a $W/logs/therm.log      # checkpoint 5
for i in 1 2 3 4 5 6 7 8; do for v in C S2 S1; do printf 'p0%s %s: ' $i $v
  cmp -s <(strip $W/logs/text_A_p0$i.log) <(strip $W/logs/text_${v}_p0$i.log) && echo identical || echo DIFFERENT; done; done
```

Prefill counts per the prompt README: 512 / 207 / 117 / 326 / 207 / 276 /
402 / 24.

### 8. Profile, not tok/s (≈ 2 min)

```
for v in A C S2 S1; do run $v 64 prof_$v prompt512.txt NNTR_HTP_PROFILE=2; done
grep -hE 'level=|K=2048  N=2048  M==1|staging:' $W/logs/prof_{A,C,S2,S1}.log | cut -c1-400
```

Expected: `level=2 qos_mode=2`; the M==1 row `calls=<n> … dsp= … mm …` —
A and C: 4-expert calls; S2: 2-expert calls; S1: 1-expert calls. C's `mm`
against A's is the cached attribute's cost on the DSP read (risk table).

### 9. Checks on the workstation (1 min)

```
grep -c 'dspq: on' $W/logs/{A,C,S2,S1}_G*_r*.log | grep -v ':1$'                     # nothing (rule 40)
grep -l 'arena: cached\|moe split' $W/logs/A_G*_r*.log $W/logs/text_A_*.log          # nothing
grep -L 'arena: cached' $W/logs/{C,S2,S1}_G*_r*.log                                   # nothing
grep -L 'moe split: k=' $W/logs/{S2,S1}_G*_r*.log                                     # nothing
grep -l 'asked for, but' $W/logs/*.log                                                # nothing
grep -h 'moe m1 split=' $W/logs/{A,S2}_G512_r1.log                                    # the timer table below
grep -l 'HTP-PROFILE' $W/logs/[ACS]*_G*_r*.log                                        # nothing
for v in C S2 S1; do for g in 64 512 1024; do for r in 1 2; do printf '%s G=%s r%s vs A: ' $v $g $r
  cmp -s <(strip $W/logs/A_G${g}_r$r.log) <(strip $W/logs/${v}_G${g}_r$r.log) && echo identical || echo DIFFERENT
done; done; done
cat $W/logs/therm.log
```

Then fill the tables below, commit this file on the branch, push, and set
#157 to `state:measured`.

## Gates (plan section 1)

| # | check | pass |
|---|---|---|
| G0 | step 2 gtests | `bad_elems=0` (all k, all experts, `cpu_path=neon_sdot`); arena gtests pass in both modes |
| G1 | dumps vs A1 (step 5) | `C`, `S2`, `S1` `bit_identical=1` on every file; null `a2` `=1` |
| G2 | text, 8 prompts, G = 64 (step 7) | 8/8 identical for C, S2, S1 |
| G3 | text in every tok/s cell vs A of the same G and run (step 9) | 18/18 identical |
| G4 | `[PPL] decode` lines (step 6) | C, S2, S1 identical to A's forced run |
| G5 | lever: decode tok/s (all), G = 512, mirrored mean | S2 ≥ 1.20 × A |
| G6 | prefill tok/s, per mirrored pair | every variant ≥ 0.95 × A at each G |
| G7 | C alone | decode within ±1 % of A (mirrored mean, each G); M==1 `mm` within ±2 % of A (step 8) |
| G8 | banners (steps 3, 9) | as the variants table; `split_calls = calls` in every S log |

The split stays **off** by default after this sitting; making S2 the
default is a user decision once G0–G8 pass.

## Results (fill in)

Unit: ______ , date / time: ______ , device `md5sum` = the table: ___

### G0 (step 2)

```
(paste the moe_m1_split fields and the two arena runs' fields)
```

### tok/s (prompt 512; mirrored: A C S2 S1 run 1, S1 S2 C A run 2)

| variant | G | prefill r1 / r2 | decode (all) r1 / r2 | decode (last 64) r1 / r2 | decode mean | vs A | peak RSS KB r1 | text = A? |
|---|---|---|---|---|---|---|---|---|
| A | 64 | | | | | — | | (reference) |
| C | 64 | | | | | | | |
| S2 | 64 | | | | | | | |
| S1 | 64 | | | | | | | |
| A | 512 | | | | | — | | (reference) |
| C | 512 | | | | | | | |
| S2 | 512 | | | | | | | |
| S1 | 512 | | | | | | | |
| A | 1024 | | | | | — | | (reference) |
| C | 1024 | | | | | | | |
| S2 | 1024 | | | | | | | |
| S1 | 1024 | | | | | | | |

Reference: #90's control on `htp_moe` (dspq default, `R3CY10WM83Y`,
2026-09-29) decode 39.70 / 39.18 / 36.53 tok/s at G 64 / 512 / 1024; the
plan's model for S2: 56.6 / 55.6 / 50.9 (#90 rates) or 51.7 / 50.8 / 46.9
(conservative). Goal ≥ 50 (contract section 1), prefill ≥ −5 % of this
sitting's A.

### The split's timer (exit line, G = 512 run 1)

| variant | calls | call_us / call | cpu_us / split call | dsp_wait_us / call | cpu_MB / split call |
|---|---|---|---|---|---|
| A | | | — | — | — |
| S2 | | | | | |

### Profile, M==1 row (step 8)

| variant | calls | host us/call | dsp us/call | transport us/call | mm |
|---|---|---|---|---|---|
| A | | | | | |
| C | | | | | |
| S2 | | | | | |
| S1 | | | | | |

### Dumps (G1) and PPL (G4)

```
(paste the four htp_dump_eval lines: a2, C, S2, S1; the three PPL
comparisons and the [PPL] decode tokens lines)
```

### Text set (G2), G = 64

| prompt | tokens (prefill line) | C = A? | S2 = A? | S1 = A? |
|---|---|---|---|---|
| p01 `prompt512.txt` | | | | |
| p02 `bitset-02-code.txt` | | | | |
| p03 `bitset-03-math.txt` | | | | |
| p04 `bitset-04-korean.txt` | | | | |
| p05 `bitset-05-json.txt` | | | | |
| p06 `bitset-06-dialogue.txt` | | | | |
| p07 `bitset-07-facts.txt` | | | | |
| p08 `bitset-08-short.txt` | | | | |

## Notes from the run

Thermal checkpoints (0 start, 1–3 after each G, 4 after the dumps, 5 after
the text set): the CPU now works through the window it used to idle in, so
S2's last-64 column against its all-token column is the DVFS read.

```
(paste therm.log)
```

FARF / AEE errors, anything stale.
