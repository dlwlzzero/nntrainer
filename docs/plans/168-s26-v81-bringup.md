# 168 — The MoE decode on the Galaxy S26 Ultra NPU (SM8850, HTP v81), end to end

Issue: dlwlzzero/nntrainer#168 (p0). Read against `htp_moe_v81` @ `f810c45b`
(= `htp_moe` `40e797ed` + the contract commit whose decision-log row lets
agents run `adb -s R5KL20NFRCK` on this branch only). Work branch
**`htp/168-s26-v81-bringup`**, PR into **`htp_moe_v81`** (never `htp_moe`).

## 0. What the planner already checked on the workstation and the phone

| fact | how | result |
|---|---|---|
| The arch is **already a knob** | `test/htp/build.sh:17` `HEX_ARCH="${HEX_ARCH:-v79}"`, used at `:35` (HexKL lib), `:94` (`-m$HEX_ARCH`), `:103-104` (QuRT includes); `tools/htp/env.sh:35` `export HEX_ARCH="${HEX_ARCH:-v79}"`, `:36` lib check | no fork needed |
| **The v81 skel builds today, unchanged tree** | `HEX_ARCH=v81 ./test/htp/build.sh` (run once, v79 artifact restored afterwards) | `-Wall -Werror` clean, `UNDEFINED SYMBOLS OK (51 runtime imports)` (same count as v79, #158), `hexagon-readelf -h` → `Flags: 0x81, V81` (v79 skel: `0x79, V79`), md5 `ad444ab4…` (not reproducible build to build, #158 note) |
| HexKL v81 ABI | `hexagon-nm` on `lib/6.4.0.1/hexagon_toolv19_v{79,81}/libhexkl_micro.a` | defined symbols (76) and undefined imports (63) **identical**; `hexkl_micro.h` is one header for all arches |
| QuRT v81 headers | `diff -rq rtos/qurt/computev79/include rtos/qurt/computev81/include` | **identical** (incl. `qurt_hvx.h`, `qurt_hmx.h`, `qurt_user_dma.h`, `qurt_l2cfg.h`) |
| HVX intrinsics | `Tools/target/hexagon/include/hvx_hexagon_protos.h` | 31 blocks `__HVX_ARCH__ >= 81`, all **additions** (fp8, bf16 convert, `vilog2`, `veqsf`…); nothing removed |
| UDMA descriptor | `hexagon_types.h:2633` `hexagon_udma_descriptor_type1_t` | one ungated layout (incl. `srcbypass`) for every arch in toolchain 19.0.04; our `hexkl_dma_ring.h:24-44` mirrors it |
| SDK feature matrix (`docs/reference/feature_matrix.html`) | Pakala (v79) vs Kaanapali (v81) columns | cDSP v81; Unsigned PD, VTCM APIs, User DMA, FP HVX: **yes on both**; UBWC DMA: no on Kaanapali (unused here) |
| Phone | `getprop`, `/proc/cpuinfo`, `df`, `ls` | SM-S948N, `ro.soc.model=SM8850`, `ro.board.platform=canoe`, Android 16 (SDK 36), 8 cores, `MemTotal` 11.1 GiB (7.0 GiB available when idle), `/data` **50 GB free**, SELinux enforcing, shell uid 2000 (not in group `system`) |
| Phone FastRPC | `ls -la /dev`, `adb pull /vendor/lib64/libcdsprpc.so` (read only) | `/dev/fastrpc-cdsp` `crw-rw-r-- system system`; `libcdsprpc.so` md5 `f15d4489…` exports `dspqueue_*`, `remote_session_control`, `remote_handle64_{open,control}`, `rpcmem_alloc2`, `fastrpc_mmap` |
| Phone layout | `ls /data/local/tmp/nntrainer/causallm{,/models}` | used by other projects (gemma4, gauss4, their own binaries in the top dir); **no `models/q40*` yet**. Do not touch anything that exists |
| USB | `adb devices -l` | three devices: `R3CN80CW3FY`, `R3CY10WM83Y` (an S25 — agents never touch it), `R5KL20NFRCK`. Every command names `-s R5KL20NFRCK` |

Not verified anywhere on disk (the device answers these, §4 step 5):
SM8850's VTCM size (the skel takes it from `hexkl_micro_hw_init` at run time,
`test/htp/hvx_add_f32.c:71-74`), its HVX context count (`qurt_hvx_get_units`,
`:113`), whether `canoe` is the feature matrix's Kaanapali, the v81 L2 /
`l2fetch` limits (the V79 PRM's three outstanding per thread,
`hvx_gemm_u8i4_wh.h:83`), the v81 DMA engine's `src_bypass` behaviour, and
whether Samsung's Android 16 build lets the shell user open an unsigned PD.

**The issue's accuracy wording cannot pass as written, and this plan says so
up front.** The NPU model and the CPU control have **different MoE weights**
(`QS4CX_WH` per-channel int4 vs `Q4_0`). On the S25 the NPU text leaves the CPU
`q40` text at generated word 43 at every G (LEDGER ⑱, #95), and every NPU row
in BENCHMARK.md carries "text = CPU: n/a (different weights)". `NNTR_L2_DIFF`
only hooks the per-expert fused path (`htp_compute_ops.cpp:1873`). The MoE
layer call that this model runs (`:2945` / `:2953`) has no such hook, so on
this path the variable prints nothing. This plan therefore keeps both of the
issue's items as **recorded columns** and makes the decision on a check that
can pass: **v81 reproduces the S25 (v79) NPU run bit for bit.** That check uses
the MoE dumps, the decode nll lines and the text, compared with #158's variant
B. Variant B is the same code as today's default for the MoE path:
`git diff 0106622f HEAD` touches only the default of `htp_moe_opts.h`'s bypass
bit and the resident (off-by-default) ATTN_M1 / ROPE code. The user confirms
this reading in the issue before the verdict is folded (§1 G5).

## 1. Goal and gate

Goal (issue): `nntrainer_causallm` with `q40-qs4cx-wh` (`moe_engine: htp`) on
`R5KL20NFRCK` loads a **v81** skel, runs every MoE layer on the DSP, and is
correct. tok/s are recorded, not gated.

| # | check | pass |
|---|---|---|
| G0 | skel | the device `md5sum` of `libnntr_hvx_skel.so` equals the staged v81 file, whose `hexagon-readelf -h` reads `Flags: 0x81` (`build.sh` prints `ARCH OK (V81)`, §4 step 1) |
| G1 | the DSP ran the MoE, no CPU fallback | every NPU log has, once each: `[HTP] moe m1 gemv: on (applied=0x703e1) lead=192KB rows1=1 feed=vtcm dma_bypass=1 source=default` (`htp_compute_ops.cpp:1807`), `[HTP] dspq: on queue=0x…` (`:2750`), and `[HTP] dspq: close calls=N served=N bad=0` with **N = 22 × G** (`:2666`). None of `dspq: off` (`:2744`), `moe m1 gemv: off` (`:1777`), `HTP arena: cannot register` (`:3679`), and no logcat `nntr_hvx_open failed` (`htp_backend.cpp:56`, tag `nntrainer`). The level-2 profile's `K=2048 N=2048 M==1` row reads `calls=1408 … m1_gemv=1408/1408 feed=1408/1408` at G=64 |
| G2 | MoE dumps, v81 vs v79 | `tools/htp/htp_dump_eval.py --label v81 /local/mnt/workspace/htp_moe/158b/dump/B <v81 dump>` → `bit_identical=1` (G=64, `prompt512.txt`, 2862 files as on the S25) |
| G3 | decode nll, v81 vs v79 | `[PPL] decode step=…` lines of a v81 self-run at G=512 are byte-identical to `158b/logs/ppl_A.log` and `ppl_B.log` |
| G4 | text, v81 vs v79 | stripped text ≡ `158b/logs/B_G{64,512,1024}_r1.log` (p01) and ≡ `158b/logs/text_B_p0{1..8}.log` (8 prompts, G=64) |
| G5 | the issue's columns, recorded | NPU text vs the CPU `q40` run on the same phone, same prompt, same G: y/n and the first differing word index (expected **n**, ≈ word 43 as on the S25). `NNTR_L2_DIFF`: recorded as "n/a — no hook on the MoE-layer path". The user marks in the issue whether G2–G4 stand in for "text identical to CPU" |
| G6 | tok/s recorded | prefill, decode (all) and decode (last 64) for CPU `q40` and NPU, prompt 512, G 64 / 512 / 1024, two runs each, `NNTR_NUM_THREADS=8`, non-profile binary |
| G7 | `htp_moe` unchanged | an unset `HEX_ARCH` still builds a v79 skel (`ARCH OK (V79)`); the app bits do not depend on the arch |

Standing gates. The prefill gate (≥ −5 % of variant A) is **n/a**: this is the
first sitting on v81 and there is no A of the same phone. The v81 NPU prefill
recorded here is that A for every later `htp_moe_v81` issue. Text vs CPU
(the contract's column) is G5.

## 2. Where it lives

| file:line | today | change |
|---|---|---|
| `tools/htp/env.sh:17` | comment "HEX_ARCH (v79, the S25 Ultra)" | comment names v81 = S26 Ultra and says to `export HEX_ARCH=v81` **before** sourcing: `:35` exports the default, so a later bare `./test/htp/build.sh` silently builds v79 |
| `tools/htp/env.sh:35-36,49` | default v79, lib check and banner use `$HEX_ARCH` | none (already the knob) |
| `test/htp/build.sh:9` | "Override the target with: HEX_ARCH=v75" | name v81 |
| `test/htp/build.sh:56` | "HexKL provides v79 only for lib/6.1.1.0 and newer." | arch-neutral wording (the listing above it already says what exists) |
| `test/htp/build.sh:144-146` | `UNDEFINED SYMBOLS OK`, `built: … ($HEX_ARCH …)` | **new guard**: `"$READELF" -h build/libnntr_hvx_skel.so` must show `Flags: 0x${HEX_ARCH#v}`. It prints `ARCH OK (V81)` or exits 1. Both arches write the same file name, which the phone needs, so the ELF flag is the only thing that tells them apart |
| `.claude/skills/hexagon-gates/SKILL.md:85`, §4 | rung 2 "v79"; rung 4 "user only" | add `HEX_ARCH=v81` and the `ARCH OK` pass line; rung 4 names the `htp_moe_v81` / `R5KL20NFRCK` exception and points at this plan's §4 step 5 |
| `.claude/skills/hexagon-handoff/SKILL.md:49` | "Artifacts (… v79)" | "v79 or v81" |

Not changed, and why. The Android app (`Applications/CausalLM/build_android.sh`
`--htp`, `:147-203`) is ARM64 only. `libsdkl.so` is `armv8_android26` (meson
option `hexkl-lib-subdir`, `meson_options.txt:64`), is only linked, and is never
called (`htp_backend.h:21`). The ARM stub (`generate_stub.sh`) comes from the
same IDL for both arches. The session open (`htp_backend.cpp:41-58`: unsigned
PD on `CDSP_DOMAIN_ID`, `nntr_hvx_URI&_dom=cdsp`) has no arch in it. The DSP
code reads the VTCM size and the HVX context count at run time
(`hvx_add_f32.c:71-74`, `:113-118`). The only 8 MiB literal on the production
side is `fcMaxRows` (`htp_compute_ops.cpp:1064-1065`, already a `ponytail`).
It is used only by the FC-on-HTP path, which the default config does not run.
It matters only if v81's VTCM turns out smaller than 8 MiB. The host stand-ins
(`test/htp/host/inproc/hexkl_micro_standin.c:13`, `dma_probe_host_check.c:45`,
`stub/qurt.h:100`) model v79 and stay that way: the host gates do not change.
No IDL, `HtpComputeOps` contract, quantizer format tag, loader check,
`NNTR_HTP_PROFILE` stage or `tools/htp_fc_report.py` column moves.

## 3. Design

**Chosen: the existing `HEX_ARCH` knob plus one ELF-flag guard.** v79 stays the
default everywhere, so `htp_moe` behaviour is byte-for-byte unchanged. The
S26 run selects v81 with `export HEX_ARCH=v81` before `source tools/htp/env.sh`.
The only new code is the guard (≈ 6 lines of shell), because a v79 skel and a
v81 skel share one file name and one md5 table row. No kernel code changes: the
kernels read VTCM / HVX counts at run time. The M=1 GEMV's bytes do not depend
on the lane count (int32 sums, per-column units, scatter on the caller in fixed
order, `hexkl_mm_u8i4_moe.c:535-560`). So a different v81 context count moves
speed, not bits, and G2 checks that on silicon.

**Rejected: a v81 fork of `env.sh` / `build.sh` (or a v81 default on this
branch).** It duplicates a script whose knob already works and it would make
`htp_moe_v81` diverge from `htp_moe` in tooling. It also buys nothing that
`export HEX_ARCH=v81` does not.

**Rejected: running the v79 skel on the v81 DSP as a "variant A".** HexKL links
a per-arch HMX library. A mismatched HMX configuration can crash the cDSP PD
(subsystem restart) on a phone that other projects share, and the issue asks
for a v81 skel.

Contract §2 / doc 45 §3 rules are untouched. No new DSP allocation. The arena
budget is unchanged (3840 MiB arena, but the v81 PD's address-space layout is
unverified, §5). `QS4CX_WH` keeps no CPU fallback: G1 is exactly the check
that there was none. No kernel or quantizer input changes, so no new `_det`
spec is needed, and bit identity is gated on silicon by G2 / G3.

## 4. Steps

Every shell starts with:

```
cd /home/j2z0-lee/nntrainer && git checkout -b htp/168-s26-v81-bringup origin/htp_moe_v81
export HEX_ARCH=v81 && source tools/htp/env.sh     # banner must end "arch v81"
export HEXKL_ROOT HEXKL_SDK_VER                    # ~/Qualcomm/hexkl-1.0-beta.2/hexkl_addon, 6.4.0.1
```

**Step 1 — guard and comments** (§2 table). Two commits: the kernel/app-side
`[HTP] build.sh: check the skel's ELF arch; env.sh names v81` (tools/, test/htp/),
and separately the agent-system `.claude/skills` edits. Gate, rung 0 is n/a
(shell and markdown only). Rung 2 for **both** arches:

```
HEX_ARCH=v79 ./test/htp/build.sh | grep -E 'UNDEFINED SYMBOLS OK|ARCH OK'   # (51 runtime imports), ARCH OK (V79)
HEX_ARCH=v81 ./test/htp/build.sh | grep -E 'UNDEFINED SYMBOLS OK|ARCH OK'   # (51 runtime imports), ARCH OK (V81)
md5sum test/htp/build/libnntr_hvx_skel.so
```

Show that the guard catches a mismatch once (for example by running the check
line by hand against the other arch's skel) and paste the exit 1 in the PR.

**Step 2 — host rung, once before the PR** (rung 1 of the skill, unchanged).
`ninja -C build`, the `*qs4cx*` / `*Lfm2Moe*` gtests (6 passed), `run_host_checks.sh`
(`ALL CHECKS PASS`, `WORKER POOL LANES OK`), `tools/htp_syntax_check.sh`,
`run_inproc_e2e.sh` (`INPROC E2E PASS`). Nothing host-visible changed, so this
must be green as on `htp_moe`.

**Step 3 — app** (rung 3). `(cd Applications/CausalLM && ./build_android.sh --htp --cache)`.
The existing `builddir` is `enable-htp=true`, `hexkl-lib-subdir=6.4.0.1/armv8_android26`.
Run `ninja -C builddir install` first if the `strings` check complains. Pass:
`readelf -d` NEEDED `libsdkl.so` and `libcdsprpc.so`, `NNTR_HTP_FORWARD_KINDS`
count ≥ 1, and the md5s recorded.

**Step 4 — stage** `/local/mnt/workspace/htp_moe/168/` with `md5.txt`:
`libnntr_hvx_skel.so` (the **v81** one from step 1, re-checked with
`hexagon-readelf -h | grep 'Flags:.*0x81'`), `nntrainer_causallm`,
`libcausallm_core.so` (from `jni/libs/arm64-v8a/`), `libnntrainer.so` and
`libccapi-nntrainer.so` (from `jni/obj/local/arm64-v8a/`), `libc++_shared.so`
(NDK r30, `b1586b9b…`), `libsdkl.so` (HexKL `6.4.0.1/armv8_android26`),
`docs/measurements/77-prompt512.txt` as `prompt512.txt`, and
`docs/measurements/prompts/bitset-0{2..8}-*.txt`. Never stage `libcdsprpc*`.
The phone has its own (`/vendor/lib64/libcdsprpc.so`).

**Step 5 — device, agent-run on `R5KL20NFRCK` (the unavoidable silicon step;
this replaces the handoff for this issue per the contract's 2026-09-29 row).**
One sitting, ≈ 45 min plus ≈ 5 min of model push. Record it as
`docs/measurements/168-s26-v81-bringup.md` in the handoff skill's shape (commit,
md5s, serial, thermal, filled tables). The two "variants" are the CPU control
(`q40`) and the NPU (`q40-qs4cx-wh`): one binary set, switched by the model's
config (rule 21). The v79 reference is the S25's #158 B sitting, read from disk
and not re-run.

```
S=R5KL20NFRCK; W=/local/mnt/workspace/htp_moe/168; L=$W/logs; mkdir -p $L $W/dump
D=/data/local/tmp/nntrainer/causallm/s168; MR=/data/local/tmp/nntrainer/causallm/models; DD=/data/local/tmp/s168dump
R=/local/mnt/workspace/htp_moe/158b     # S25 v79 reference: logs/, dump/B
therm() { adb -s $S shell "dumpsys battery | grep -E '^  (level|temperature)'; cat /sys/class/thermal/thermal_zone0/temp"; }
run() { # run <q40|q40-qs4cx-wh> <G> <log> <prompt> [env]
  adb -s $S logcat -c
  adb -s $S shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' $MR/$1/nntr_config.json && \
    md5sum libnntr_hvx_skel.so nntrainer_causallm libnntrainer.so && \
    NNTR_NUM_THREADS=8 $5 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./nntrainer_causallm $MR/$1 \"\$(cat $D/$4)\"; echo EXIT=\$?" \
    2>&1 | tee $L/$3.log | grep -E '^(prefill|generation|total|peak memory)|dspq:|moe m1 gemv|EXIT='
  adb -s $S logcat -d -v time > $L/$3.logcat
  grep -E 'nntrainer|adsprpc|fastrpc|CDSP|nntr_hvx|hexkl' $L/$3.logcat | grep -iE 'fail|err|warn|denied|unsigned' | head -20
}
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP\] dspq: [^\n]*\n//g; s/\[OP-TIME\][^\n]*\n//g; s/\[PPL\][^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|libnntrainer.so\|num_to_generate\|^EXIT='; }
```

(`strip` is #158's function, plus the `EXIT=` line. Check that it strips the
158b logs to their text before trusting a comparison.)

5.0 **Preconditions (read-only, 1 min).** `adb devices` (record every serial).
`adb -s $S shell df -h /data` must show ≥ 12 GB available: 9.1 GB of models,
≈ 0.2 GB of binaries and ≈ 0.2 GB of dumps at a time. Otherwise stop and
report; do not delete other projects' files. `therm` is checkpoint 0. The screen
is off and the phone is on USB power.

5.1 **Push (≈ 5 min).** Only new paths are written:
```
adb -s $S shell "ls $MR/q40 $MR/q40-qs4cx-wh $D 2>&1 | head -3"          # must all be 'No such file'
adb -s $S push $NNTR_MODEL_DIR/q40 $MR/q40
adb -s $S push $NNTR_MODEL_DIR/q40-qs4cx-wh $MR/q40-qs4cx-wh
adb -s $S shell "md5sum $MR/q40/nntr_lfm2_8b_a1b_q40_arm.bin $MR/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin"
#   d28f55c5bd7adeb8bf73b02de582eb88 / 7b7867fab51845664c0050c0a837073e
adb -s $S shell "mkdir -p $D $DD" && adb -s $S push $(ls -p $W | grep -v / | grep -v md5.txt | sed "s|^|$W/|") $D/
adb -s $S shell "chmod 755 $D/nntrainer_causallm; cd $D && md5sum \$(ls -p | grep -v / | sort)" | tee $L/md5_device.log
diff <(sort -k2 $W/md5.txt) <(sort -k2 $L/md5_device.log | tr -d '\r') && echo MD5 OK
```
Config, both models, the same as the S25 sittings (greedy, EOS banned). The
tokenizer paths in the shipped `nntr_config.json` already point at `$MR/<model>/`:
```
for m in q40 q40-qs4cx-wh; do adb -s $S shell "cd $MR/$m && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|init_seq_len|_engine|_htp_layers|moe_layer_dtype|tokenizer_file' nntr_config.json"; done
```
Expected: `q40` has no `_engine` key (so `moe_engine` is cpu) and
`moe_layer_dtype: Q4_0`. `q40-qs4cx-wh` has `moe_engine: htp`,
`moe_htp_layers: ""` and `QS4CX_WH`.

5.2 **Bring-up at G=8 (2 min). Stop here on any failure and diagnose (§5
table).**
```
run q40 8 sanity_cpu prompt512.txt
run q40-qs4cx-wh 8 sanity_npu prompt512.txt
grep -c 'dspq: on' $L/sanity_npu.log; grep 'moe m1 gemv\|dspq: close' $L/sanity_npu.log   # 1; applied=0x703e1 … dma_bypass=1; calls=176 served=176 bad=0
```

5.3 **tok/s, prompt 512, CPU first, then NPU (≈ 12 min).**
```
for g in 64 512 1024; do for r in 1 2; do run q40 $g cpu_G${g}_r$r prompt512.txt; done; done; therm
for g in 64 512 1024; do for r in 1 2; do run q40-qs4cx-wh $g npu_G${g}_r$r prompt512.txt; done; therm; done
```

5.4 **v81 vs v79 bits (≈ 6 min).**
```
adb -s $S shell "rm -rf $DD/v81 && mkdir -p $DD/v81"
run q40-qs4cx-wh 64 dump_npu prompt512.txt NNTR_HTP_DUMP=$DD/v81
adb -s $S pull $DD/v81 $W/dump/v81 >/dev/null && adb -s $S shell "rm -rf $DD/v81"
python3 tools/htp/htp_dump_eval.py --label v81 $R/dump/B $W/dump/v81 | tail -1               # G2: bit_identical=1
adb -s $S shell "rm -f $DD/cont.ids"
run q40-qs4cx-wh 512 ppl_npu prompt512.txt NNTR_PPL_DECODE=$DD/cont.ids
nll() { grep -o '\[PPL\] decode step=.*' "$1"; }
cmp <(nll $L/ppl_npu.log) <(nll $R/logs/ppl_A.log) && cmp <(nll $L/ppl_npu.log) <(nll $R/logs/ppl_B.log) && echo "G3 nll v81 == v79"
```
If G2 fails, first look at `htp_dump_eval`'s `first_diff`. If the first
differing file is a MoE **input**, the ARM side produced it and the DSP is not
at fault (the S26 CPU has SVE2/SME, and nntrainer has no runtime CPU dispatch:
`grep getauxval|HWCAP` finds nothing in `nntrainer/` or `Applications/`, but
`subprojects/` is unverified). If the first differing file is an output, the
difference is in v81 arithmetic or DMA.

5.5 **Text set, 8 prompts at G=64, CPU and NPU (≈ 8 min).**
```
P="prompt512.txt bitset-02-code.txt bitset-03-math.txt bitset-04-korean.txt bitset-05-json.txt bitset-06-dialogue.txt bitset-07-facts.txt bitset-08-short.txt"
i=0; for p in $P; do i=$((i+1)); run q40 64 text_cpu_p0$i $p; run q40-qs4cx-wh 64 text_npu_p0$i $p; done; therm
```

5.6 **Diagnostics and one level-2 profile (2 min, not tok/s).**
`run q40-qs4cx-wh 64 prof_npu prompt512.txt NNTR_HTP_PROFILE=2`. Record
`level=2 qos_mode=` and the `K=2048 N=2048 M==1` row (`calls=1408 …
m1_gemv=1408/1408 feed=1408/1408`, `dsp`, `mm`, `transport`), plus its
`weight DMA:` and `DMA ring: … engine … GB/s` lines. These are the first v81 DMA
readings; the S25 B read `engine 56.3..57.9 GB/s`, `dsp` 417.5 µs.

5.7 **Checks on the workstation (1 min).**
```
grep -H -E '^(prefill|generation)|peak memory|EXIT=' $L/{cpu,npu}_G*_r?.log
grep -h 'moe m1 gemv' $L/*npu*.log | sort | uniq -c           # one kind: applied=0x703e1 … dma_bypass=1 source=default
grep -c 'dspq: on' $L/*npu*.log | grep -v ':1$'                # nothing
grep -l 'dspq: off\|moe m1 gemv: off\|HTP arena: cannot' $L/*.log   # nothing
grep -h 'dspq: close' $L/npu_G*_r?.log                         # calls=served=22·G, bad=0
grep -l 'nntr_hvx_open failed' $L/*.logcat                     # nothing
grep -h -E 'moe m1 gemv|dspq' $L/*cpu*.log                     # nothing: the CPU runs never call the MoE on the DSP
for g in 64 512 1024; do printf 'G4 G=%s: ' $g; cmp -s <(strip $L/npu_G${g}_r1.log) <(strip $R/logs/B_G${g}_r1.log) && echo identical || echo DIFFERENT; done
for i in 1 2 3 4 5 6 7 8; do printf 'p0%s npu≡v79:' $i; cmp -s <(strip $L/text_npu_p0$i.log) <(strip $R/logs/text_B_p0$i.log) && echo y || echo n
  printf '      npu≡cpu:'; cmp -s <(strip $L/text_npu_p0$i.log) <(strip $L/text_cpu_p0$i.log) && echo y || \
  { echo n; diff <(strip $L/text_npu_p0$i.log | tr -s ' \n' '\n\n') <(strip $L/text_cpu_p0$i.log | tr -s ' \n' '\n\n') | head -1; }; done   # G5 first differing word
```
Fill the measurement doc and commit it on the work branch. Open the PR into
`htp_moe_v81`, set `state:review`, and comment the G5 question to the user.

**Clean-up** after the PR is up. Keep `$MR/q40*` on the phone for later
`htp_moe_v81` issues, delete `$DD`, and never delete anything outside `s168`,
`s168dump` and `models/q40*`.

## 5. Risks and failure modes (host vs device)

| symptom | where to look | meaning / next step |
|---|---|---|
| no `[HTP]` lines; logcat `nntr_hvx_open failed (err=…)` | `$L/*.logcat`, tags `nntrainer`, `adsprpc`/`CDSP` | `0x80000406` AEE_EUNABLETOLOAD: skel not found (`ADSP_LIBRARY_PATH=.`, file name) or an import the v81 image lacks (build.sh's guard covers the project symbols only). `0x80000481` AEE_EUNSIGNEDMOD / `0x80000438` EINVALIDSIGNATURE / `0x80000415` EPRIVLEVEL: this Android 16 build refuses the unsigned PD for shell. Check logcat for `remote_session_control(unsigned PD) failed` (`htp_backend.cpp:45`). That case needs a user decision (test-signed skel), so stop. `0x80000414` AEE_EUNSUPPORTED: `qurt_hvx_get_units()` reported 0 HVX contexts (`hvx_add_f32.c:52`). A `hexkl_micro_hw_init failed` FARF ERROR line (`:78`): HexKL v81 cannot claim VTCM/HMX |
| NPU run gives text but no banner, or throws in the MoE layer | `HtpContext::initialize` degraded to CPU ops (`htp_context.cpp:58-65`) | **the silent-fallback case**: `QS4CX_WH` on CPU ops. G1 voids the run whatever the text says |
| `nntr_hvx_moe_set_opts failed` / `moe m1 gemv: off (moe_set_opts err=0x8000040e)` | log | stale or wrong skel, or an IDL mismatch: compare the device md5 with `md5.txt`; `ARCH OK (V81)` |
| `HTP arena: cannot register … rpcmem_alloc(…)` / `nntr_hvx_arena_attach(…)`, `mapped=… MiB` | log (the message names which call refused and the RSS) | the v81 PD's 32-bit address budget or the dma-heap. 11.1 GiB RAM, 7.0 GiB free at rest; a kill (`EXIT=137`, logcat `lowmemorykiller`) is RAM, not the DSP |
| `AEE_ERPC 0x80000600` in the first MoE call / PD restart in logcat (`subsystem restart`, `cdsp`) | logcat | a DSP crash inside a kernel: rebuild with `HEX_EXTRA_CFLAGS=-DFARF_HIGH=1` (HIGH is compiled out today, `HAP_farf.h:160`) as a **diagnostic** skel and push `nntrainer_causallm.farf` (`0x1f`) next to it to read `nntr_hvx_open: qurt_hvx_get_units()=… n_hvx=…` and the kernel's own FARF lines |
| `dspq: off (…)` | log | dspqueue refused on this PD. MoE calls then stay on FastRPC (correct, slower). G1 fails as written: record the reason and ask the user |
| G2 fails, G1 passes | `htp_dump_eval` `first_diff` | see 5.4: input differs → ARM side; output differs → v81 arithmetic or DMA (`src_bypass` on v81 is unverified; re-run once with `NNTR_MOE_DMA_BYPASS=0` to split it) |
| tok/s oddities | `qos_mode`, ring `engine` GB/s, thermal checkpoints | DVFS votes are best effort and their rejections are FARF HIGH, i.e. invisible. `qos_mode=1` is visible in the profile. Thermal drift: CPU first, checkpoints after each G. No A/B verdict is drawn from this sitting's tok/s |

Host-vs-device gaps this plan is exposed to: the host checks model v79 (8 MiB
VTCM, 6 prefetch lanes, emulated HVX), so nothing on the host speaks for v81.
Correctness on v81 rests on G2–G4 alone. The DMA rate, `l2fetch` limits and
HVX count on v81 are unknown and affect only speed; the level-2 profile shows
them. Thermal drift between sittings does not matter for the verdict, which is
bits, not rates. A stale skel is caught by the device md5 plus `ARCH OK` plus
`applied=0x703e1`.

## 6. Docs to update (supervisor, from the filled measurement doc)

* `docs/htp_moe/BENCHMARK.md`: a new block, **"Galaxy S26 Ultra (SM8850, v81),
  branch `htp_moe_v81`"**, with rows tagged `R5KL20NFRCK`: CPU `q40` and NPU
  `q40-qs4cx-wh` × G 64 / 512 / 1024 × r1 / r2 (prefill, decode all, last 64,
  peak RSS), text = CPU (G5), text ≡ S25 #158 B (G4). These are never compared
  unscaled with the S25 rows (different unit, LEDGER rule 13 / 34). The first
  v81 prefill becomes that block's A.
* `docs/htp_moe/LEDGER.md`: a rule for what v81 turned out to be (VTCM,
  `n_hvx`, ring `engine` GB/s, bits ≡ v79 y/n). An open item for G5 (the user's
  reading of "text identical to CPU" for `QS4CX_WH`, which applies to every
  NPU row). A note that `NNTR_L2_DIFF` has no hook on the MoE-layer path.
* `.claude/skills/hexagon-gates` / `hexagon-handoff`: step 1's edits.
