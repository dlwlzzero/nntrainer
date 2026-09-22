# Measurement 88: the MoE call's per-call transport after size-class staging and the 5 ms poll (wall 3)

Branch `htp/88-moe-call-marshalling` (PR #103; code head `d604177c`, this
file on the same branch) — estimated device time: **≈ 45 min**

Plan: `docs/plans/88-moe-call-marshalling.md` §4 step 4. Issue #88 (tracker
#76; LEDGER ⑦). Contract `docs/plans/0001-htp-moe-decode-agent-system.md`.

## Why

#77 B put the M==1 MoE layer call's FastRPC transport at 527.7 µs/call with
the DSP hot (level 3) and called it marshalling; the #88 plan's host
inventory says the 500 µs are the driver's cache maintenance on a 4 MiB
cached ION staging pair that decode's 8 KiB rode after prompt 512, plus
the 100 µs poll window that every ≈ 1.3 ms call outlives. PR #103 takes
upstream's two fixes (`fb0f02b9` size-class staging, `04a2fcc4` +
`b0a384d6` 5 ms poll) and prints the inventory. **Decision that hangs on
this sitting:** B's M==1 `transport` at `NNTR_HTP_PROFILE=2` ≤ 0.1 ms/call
closes ⑦ and prebind (plan §3.3) is not built; ≥ −30 % vs A is progress;
< −15 % sends the question back to #83 §4. C (B with the poll back at 100
µs) says how much of B is the poll and how much the staging.

Variants (3, one skel):

* **A** — `htp_moe` @ `08afbb10` app (the merged head without PR #103), env
  unset. Run first.
* **B** — this branch's app, env unset (poll 5000 µs default, 64 KiB
  staging class for decode).
* **C** — B's binary with `NNTR_HTP_POLL_US=100`: G = 64 only.

The DSP arithmetic is untouched (no file under `test/htp/` or
`htp_backend/{hmx,hvx}` moved; the skel below is built from this branch
and is the same source as `08afbb10`'s), so the text of A, B and C must be
identical per G, and run 1 = run 2. `NNTR_L2_DIFF` n/a for `QS4CX_WH`;
text = CPU `q40` n/a (different weights) — the accuracy column is A = B.

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

Staged under **`W=/local/mnt/workspace/htp_moe/88/`**; `$W/md5.txt` is the
`md5sum` of every staged file and is reproduced here. A was built in the
detached worktree `/home/j2z0-lee/nntrainer-88-A` @ `08afbb10`, B and the
skel in `/home/j2z0-lee/nntrainer-88` @ `d604177c` (both
`build_android.sh --htp`, then the fresh-builddir `meson configure
-Dprefix` + `ninja install` step, then `--htp --cache`; `readelf -d
libnntrainer.so` lists `libsdkl.so` and `libcdsprpc.so` in both).

| file (staged as) | md5 | built from |
|---|---|---|
| `$W/libnntr_hvx_skel.so` (→ device as `libnntr_hvx_skel.so`; serves A, B, C) | `ce85595dc016db93dec68fbec5941dad` | `test/htp/build.sh` @ `d604177c`; `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK (46 runtime imports)` (#97 guard). DSP sources identical to `08afbb10` |
| `$W/A/nntrainer_causallm` | `01bd38f6af041e7d3d028b485bf40193` | `htp_moe` @ `08afbb10` (`jni/libs/arm64-v8a/`) |
| `$W/A/libcausallm_core.so` | `0a700c028c3fede5990a95e245716b51` | same |
| `$W/A/libnntrainer.so` | `341d57621737fa61c44a04782959e6e5` | same (`jni/obj/local/arm64-v8a/`) |
| `$W/A/libccapi-nntrainer.so` | `571c0ff2168dc6b82381a1e393895981` | same |
| `$W/B/nntrainer_causallm` | `5a977ff86addc774b8cb7caa15f54391` | this branch @ `d604177c` |
| `$W/B/libcausallm_core.so` | `db004ee3e3eb9cbbdbb0c5dfb88b3410` | same |
| `$W/B/libnntrainer.so` | `4f6c90dd9963229aa3da639f40dd9498` | same — the one file that carries PR #103 (`htp_backend/`) |
| `$W/B/libccapi-nntrainer.so` | `df7cc241def803d997fd8a0581064936` | same |
| `$W/libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot (== #77 / #94 `b1586b9b…`) |
| `$W/libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` (== #77 / #94 `0ad4e22a…`) |
| `$W/prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | copied from `/local/mnt/workspace/htp_moe/94/prompt512.txt` (= `docs/measurements/77-prompt512.txt` on `origin/htp/77-first-handoff`) |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (4316133120 B) | `7b7867fab51845664c0050c0a837073e` | #78, `--moe_dtype QS4CX_WH` (NPU model; the only model this sitting runs) |
| `tokenizer.json` (`hf/`, `q40-qs4cx-wh/`) | `7b8067a580173d3eb1697afae3b456f5` | HF |

Gtests (rung 3 ride-along, not run in this sitting): `unittest_hvx_mm_u8i4`
`a03625e046e33214827df68d672177de`, `unittest_hvx_softmax` `ba140f80af11ea3053aa9a919238060e`, `unittest_hvx_attn` `9252109dcdbf78c0d7cb7c7a19d60146`,
`unittest_hvx_fc` `0e233ef27578b2229d0c0dbffb094d57` (`test/jni/obj/local/arm64-v8a/`, branch build).

Expected: `nntrainer_causallm`, `libcausallm_core.so` and
`libccapi-nntrainer.so` may or may not be byte-identical between A and B
(same sources, two build paths — LEDGER rule 14); `libnntrainer.so` must
differ. Not staged, never pushed: `builddir/.../libcdsprpc.so` (rule 5).

Workstation sanity before pushing:

```
W=/local/mnt/workspace/htp_moe/88; M=/local/mnt/workspace/models/lfm2.5-8b-a1b
(cd $W && md5sum -c md5.txt)                                        # every line OK
for v in A B; do strings $W/$v/nntrainer_causallm | grep -c 'per-layer-type totals'; done   # 0 0 (not profile binaries)
find $W -name 'libcdsprpc*' | wc -l                                 # 0
md5sum $W/prompt512.txt                                             # fc65c1588dc66dd764c7013fe96cbb75
```

## Steps (workstation, phone on USB)

Shell setup once:

```
cd /home/j2z0-lee/nntrainer-88 && git fetch && git checkout htp/88-moe-call-marshalling && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/88; M=/local/mnt/workspace/models/lfm2.5-8b-a1b; mkdir -p $W/logs
D=/data/local/tmp/nntrainer/causallm
therm() { adb shell dumpsys battery | grep -E 'level|temperature'; adb shell cat /sys/class/thermal/thermal_zone0/temp; }
```

### 0. Device state (2 min)

```
adb devices                     # record the serial that answers -- whichever unit it is, write it under Notes (rule 13)
therm                           # battery %, temperature (tenths of °C), thermal_zone0 (m°C)
adb shell df -h /data
```

Screen off, charger in. Note serial, battery %, °C under "Notes".

### 1. Install (5 min; + 5–8 if the model must be pushed)

Model present? Size must read exactly `4316133120`; if so hash it and skip
the push when it reads `7b7867fab51845664c0050c0a837073e`:

```
adb shell ls -l $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin
adb shell md5sum $D/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin
# otherwise:
adb shell mkdir -p $D/models && adb push $M/q40-qs4cx-wh $D/models/q40-qs4cx-wh
```

Skel, shared libraries, prompt, then **A's binaries** (B is pushed in
step 3 over the same file names — the md5 line is what says which is on
the device):

```
adb shell mkdir -p $D/models
adb push $W/libnntr_hvx_skel.so $D/libnntr_hvx_skel.so
adb push $W/libc++_shared.so $W/libsdkl.so $W/prompt512.txt $D/
adb push $W/A/nntrainer_causallm $W/A/libcausallm_core.so $W/A/libnntrainer.so $W/A/libccapi-nntrainer.so $D/
adb shell "chmod 755 $D/nntrainer_causallm"
```

Config edits, re-applied unconditionally (greedy, exact count; the WH dir
must carry `moe_engine: htp`):

```
adb shell "cd $D/models/q40-qs4cx-wh && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|moe_engine|moe_htp_layers' nntr_config.json"
```

Expected echo: `do_sample": false`, `bad_word_ids": [124900]`,
`init_seq_len": 512`, `moe_engine": "htp"`, `moe_htp_layers": ""`.

Provenance (paste under Notes; must equal the A rows and the skel row):

```
adb shell "md5sum $D/libnntr_hvx_skel.so $D/nntrainer_causallm $D/libnntrainer.so $D/libcausallm_core.so $D/libccapi-nntrainer.so $D/prompt512.txt $D/models/q40-qs4cx-wh/tokenizer.json"
```

### 2. A — 6 cells (≈ 14 min), NPU model, env unset

One helper (the #94 `runA`, variant name added):

```
run() { # run <variant A|B|C> <G> <run#> [extra env]
  adb shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' models/q40-qs4cx-wh/nntr_config.json && \
    grep num_to_generate models/q40-qs4cx-wh/nntr_config.json && \
    $4 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/$1_G$2_r$3.log | grep -E '^(prefill|generation|total|peak memory)'
}
therm                                                          # checkpoint 1
run A 64 1;  run A 64 2
run A 512 1; run A 512 2
sleep 60; run A 1024 1; sleep 60; run A 1024 2
therm                                                          # checkpoint 2
```

Expected lines in every log, exactly: `prefill: 512 tokens, … TPS`,
`generation: <G> tokens, … TPS`, `total: … ms`, `peak memory: … KB`; one
`[HTP] moe m1 gemv: off (applied=0x0)` line; **no** `[HTP-PROFILE]` block
(an env leak voids the run). `prefill: N` with N ≠ 512 voids the prefill
column (tokenizer mismatch); `generation: N` ≠ G is a failed config edit,
not a data point. No `generation(last 64)` line exists on this tree (#89).

### 3. B — push, then the same 6 cells (≈ 15 min)

```
adb push $W/B/nntrainer_causallm $W/B/libcausallm_core.so $W/B/libnntrainer.so $W/B/libccapi-nntrainer.so $D/
adb shell "chmod 755 $D/nntrainer_causallm && md5sum $D/nntrainer_causallm $D/libnntrainer.so $D/libcausallm_core.so $D/libccapi-nntrainer.so"   # must equal the B rows
run B 64 1;  run B 64 2
run B 512 1; run B 512 2
sleep 60; run B 1024 1; sleep 60; run B 1024 2
therm                                                          # checkpoint 3
```

### 4. C — B's binary, poll window back at 100 µs (1 min)

```
run C 64 1 NNTR_HTP_POLL_US=100
```

### 5. Profiles (≈ 6 min; never read for tok/s) — G = 64, one run each

B is on the device now; C and B first, then A is pushed back for its
profile and the thermal re-run.

```
adb shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' models/q40-qs4cx-wh/nntr_config.json"
prof() { # prof <name> <env...>
  adb shell "cd $D && $2 NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm ./models/q40-qs4cx-wh \"\$(cat prompt512.txt)\"" 2>&1 | tee $W/logs/prof_$1.log | grep -E '^\[HTP-PROFILE\]' | grep -E 'level=|K=2048 N=2048|staging:' | cut -c1-400
}
prof B_L2 "NNTR_HTP_PROFILE=2"
prof B_L3 "NNTR_HTP_PROFILE=3"
prof C_L2 "NNTR_HTP_PROFILE=2 NNTR_HTP_POLL_US=100"
adb push $W/A/nntrainer_causallm $W/A/libcausallm_core.so $W/A/libnntrainer.so $W/A/libccapi-nntrainer.so $D/
adb shell "chmod 755 $D/nntrainer_causallm && md5sum $D/libnntrainer.so"      # 341d57621737fa61c44a04782959e6e5
prof A_L2 "NNTR_HTP_PROFILE=2"
```

Paste, per profile: the header line (`level=… qos_mode=…` — **must read
`qos_mode=2` in all four**; a `1` means the driver refused the poll value
and voids that cell, LEDGER doc 51 §2.16), the `K=2048 N=2048 M==1` and
`M>1` rows, and B's / C's `staging:` line. Expected on B / C:
`staging: act 65536 B out 65536 B ion=y  rpc allocs=<n> (session)
non-ION in-args=6/464 B` under the M==1 row and `act 4194304 B out
4194304 B` under the M>1 row; `n` is process-wide (arena chunks + 4 pool
buffers) and must be the **same** in `B_L2` and `B_L3` (level 3 runs each
timed call 5×; a larger `n` at level 3 means something allocates per
call). A has no `staging:` line (the field did not exist).

### 6. A re-run (2 min) — thermal check, env unset

```
run A 64 3
for g in 64 512 1024; do printf 'A vs B G=%s: ' $g; diff <(sed -n '/^=====/q;p' $W/logs/A_G${g}_r1.log) <(sed -n '/^=====/q;p' $W/logs/B_G${g}_r1.log) >/dev/null && echo same || echo DIFFERENT; done
for v in A B; do for g in 64 512 1024; do printf '%s G=%s r1=r2: ' $v $g; diff <(sed -n '/^=====/q;p' $W/logs/${v}_G${g}_r1.log) <(sed -n '/^=====/q;p' $W/logs/${v}_G${g}_r2.log) >/dev/null && echo same || echo DIFFERENT; done; done
printf 'C vs B G=64: '; diff <(sed -n '/^=====/q;p' $W/logs/C_G64_r1.log) <(sed -n '/^=====/q;p' $W/logs/B_G64_r1.log) >/dev/null && echo same || echo DIFFERENT
grep -l 'HTP-PROFILE' $W/logs/[ABC]_G*_r*.log                         # nothing
```

Then fill the tables below, commit this file on the same branch, push, set
#88 to `state:measured`.

## Results (fill in)

Unit (serial from `adb devices`): ______  Battery / °C at checkpoints 1–3: ______

| variant | gen | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS (KB) | text = A run 1? | run1 = run2? | libnntrainer.so md5 (device) |
|---|---|---|---|---|---|---|---|---|---|
| A | 64 | 1 | | | n/a (#89) | | reference | | |
| A | 64 | 2 | | | n/a | | | | |
| A | 512 | 1 | | | n/a | | reference | | |
| A | 512 | 2 | | | n/a | | | | |
| A | 1024 | 1 | | | n/a | | reference | | |
| A | 1024 | 2 | | | n/a | | | | |
| B | 64 | 1 | | | n/a | | | | |
| B | 64 | 2 | | | n/a | | | | |
| B | 512 | 1 | | | n/a | | | | |
| B | 512 | 2 | | | n/a | | | | |
| B | 1024 | 1 | | | n/a | | | | |
| B | 1024 | 2 | | | n/a | | | | |
| C | 64 | 1 | | | n/a | | | — | |
| A | 64 | 3 (after B) | | | n/a | | | | |

Transport (`[HTP-PROFILE]`, G = 64, one run each):

| profile | qos_mode | M==1 calls | M==1 host / dsp / transport (µs/call) | M>1 calls | M>1 host / dsp / transport | staging: line |
|---|---|---|---|---|---|---|
| A level 2 | | | | | | (none) |
| B level 2 | | | | | | |
| B level 3 | | | | | | |
| C level 2 (poll 100) | | | | | | |

Verdict (plan §1): B M==1 transport ≤ 100 µs → **done**; (A − B)/A ≥ 30 %
→ progress; < 15 % → no change (back to #83 §4). C − B = the poll's share;
B − (A − C + B) ≈ the staging's share. Prefill gate: B prompt-512 tok/s ≥
−5 % of A and the M>1 transport not above A's.

Reference (other unit `R3CY205ZMND`, #77 B): M==1 transport 587.7 µs/call
at level 2, 527.7 at level 3; M>1 2231 / 2123. #77 A decode 18.2–21.3
tok/s, prefill 404–541. Author's unit: decode call transport 553 → ≈ 161
(staging, doc 50 §3.7) and 158 → 83 µs (poll, doc 51 §2.20). Goal ≥ 50
decode tok/s, prefill ≥ 497.

## Notes from the run

<serial, thermal at checkpoints, first-run page faults, FARF/AEE errors,
the provenance md5 lines, anything stale. A `qos_mode=1` header, an
`AEE_EBADPARM (0x8000040E)` or a `0x80000406` open failure goes here with
the exact line.>
