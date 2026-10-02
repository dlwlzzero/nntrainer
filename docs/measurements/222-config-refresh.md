# Measurement 222: the LFM2.5 closing table on the config of record (old vs new `nntr_config.json`, hybrid and one PD)

Branch `htp/222-config-refresh` @ `f92ec3596` (code; this file and the
runner sit on top of it) — estimated device time: **55 min** (reboot +
5 min idle, 34 runs, cool waits), plus ≈ 10 min of workstation steps.

## Why

The user adopted #222's proposed `nntr_config.json` in full (issue
comment 2026-10-02) to close out the LFM2.5-8B-A1B prefill / decode tok/s
table on it. The sitting measures the new config against the one every
sitting used so far, on the hybrid path and on one PD (Q28), and reads
the accuracy cost the user accepted (doc 51: `attn_proj` +4.5 %,
`dense_ffn` +4.2 % prefill PPL) on this tree.

**What the keys do (host-verified, `run_inproc_e2e.sh` `E2E keys lfm25
… ok`):** `conv_block_engine` / `dense_ffn_engine` / `attn_proj_engine`
= `htp` move the **prefill** (M > 1) matmuls onto the HMX FC kernels;
the decode row is unchanged by construction (hybrid: CPU kernels, every
FC gate declines M == 1; Q28: the DSP runs the whole row). Under Q28 the
keys' one-layer forms hand the DSP the same FC weights (23 handles on the
fixture; 67 weights / 74 handles expected on the 8B, as before). Two code
fixes ship with this branch: the `dense_ffn` layer now skips its CPU FFN
on a resident row (it used to recompute layers 0–1's FFN every Q token
and discard it), and the in-process host build re-lays its x86 Q4_0
repack before the qs4cx requantization (host only; device bytes
unchanged). **`init_seq_len 1024`:** compute runs over the prompt, so
prefill work is unchanged; activations double (RSS column). Its visible
effect: `causal_lm.cpp:715` registers the prefill's token only when the
prompt is **shorter** than `init_seq_len`. The prompt here is 512
tokens, so the **old config's text lacks its first generated token and
the new config's has it** — the runner marks a new text that is the old
one plus a prefix as `same+1st`; the `[PPL] decode step=` lines are
position-indexed and unaffected.

## Address budget: Qnew may not load (read in this sitting)

The keys register a second, qs4cx-WH copy of the FC weights for the HMX
prefill kernels: in_proj 108 + out_proj 36 + q/k/v/o 30 + dense FFN 42
≈ **216 MiB** (doc 50 §3; 7168-wide dense FFN). They go into already
mapped arena room first, then the DSP heap (`registerRm`; doc 50 §3.3:
≈ 100 MiB of heap for the loaded app). **Anew** (hybrid): 3696 MiB of
experts in 3840 mapped leaves ≈ 144 MiB of slack + the heap — the PR
author's all-keys run loaded this way (doc 50 §3.5, 1528 registrations).
**Qnew** (one PD): the pool is 22 × 28 × 5.25 = 3234 MiB in 13 chunks
(3328 mapped, ≈ 94 MiB of slack in chunk tails), the FC set maps 448 more
(3776 of 3840), and the token driver already holds ≈ 57 MiB of heap
(`s1_heap_kib=58578`, #216 logs) — ≈ 200 MiB of room for 216, before
fragmentation. The host cannot check the 32-bit budget, so this sitting
reads it: a new-config variant that does not load is recorded as
**VOID** with its error line (`logs/void_<v>`, the arena / registration
lines under it), its other cells are skipped and the sitting goes on; it
does not stop. A VOID Qnew is the finding that the config of record and
Q28 do not fit one PD together (the follow-up is then a code change:
the keys' copies in the FC set's arena, or fewer keyed layers via
`*_htp_layers`), not a fault of the sitting.

## Variants (one binary set; config and env only, ≤ 4 + the CPU control)

| variant | model dir config | env |
|---|---|---|
| **Aold** (reference, first) | `config/q40-qs4cx-wh.nntr_config.2026-09-21.json` (`init_seq_len 512`, no engine key) | nothing |
| **Anew** | `config/q40-qs4cx-wh.nntr_config.json` (the config of record) | nothing |
| **Qold** | as Aold | `NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28` (model file pre-read + `page_cache_evict -1`, as #201 / #216) |
| **Qnew** | as Anew | the same |
| CPU (control) | `models/q40`, its own config | nothing |

Both NPU configs get the sittings' overlay (the runner applies it):
`num_to_generate` = G, `bad_word_ids [124900]`; `do_sample false` in
`generation_config.json`. The runner restores the device's generation
configs and the `q40` config at the end and leaves the config of record
(pristine) in `models/q40-qs4cx-wh`.

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, from `f92ec3596`)

Staged at `/local/mnt/workspace/htp_moe/222/` by `222-stage.sh`
(`md5.txt` there; the runner checks it on both ends).

| file | md5 | built with |
|---|---|---|
| `test/htp/build/libnntr_hvx_skel.v79.so` (S25) | `67e30b2adc990ae5a099a8d90acc0490` | `HEX_ARCH=v79 ./test/htp/build.sh` (`ARCH OK (V79)`, `UNDEFINED SYMBOLS OK (62 runtime imports)`) |
| `test/htp/build/libnntr_hvx_skel.v81.so` (S26) | `03ac32d1dcc79487ed8366f8a4731bc6` | `HEX_ARCH=v81 ./test/htp/build.sh` (`ARCH OK (V81)`) |
| `Applications/CausalLM/jni/libs/arm64-v8a/nntrainer_causallm` | `9be6afc9c0b3e0d6201acaaa6c4ce4e5` | `build_android.sh --htp` |
| `Applications/CausalLM/jni/libs/arm64-v8a/libcausallm_core.so` | `67916891c22818754e903e0be4830166` | (`NNTR_HTP_FORWARD_KINDS` strings: 2) |
| `Applications/CausalLM/jni/obj/local/arm64-v8a/libnntrainer.so` | `2b1370ed1eaa62e5ed9bd2e536ec98e9` | NEEDED `libsdkl.so`, `libcdsprpc.so`; `graph: forward calls` strings: 1 |
| `Applications/CausalLM/jni/obj/local/arm64-v8a/libccapi-nntrainer.so` | `5dd245802c51a5a5aab31b6e2622647a` | |
| `libc++_shared.so` (NDK r30) / `libsdkl.so` (HexKL 6.4.0.1) | `b1586b9b512712800fd36a24abac1c0a` / `0ad4e22a70e4f135bce38ad8fd1e001b` | |
| `page_cache_evict` | `42595651ef514e155887eb31f421b2dc` | NDK clang from `tools/htp/page_cache_evict.c` |
| `cfg_new.json` = `config/q40-qs4cx-wh.nntr_config.json` | `110cb8bc3de536bf6458305c3ccfe398` | the #222 issue body verbatim |
| `cfg_old.json` = `config/q40-qs4cx-wh.nntr_config.2026-09-21.json` | `65056725dd9e9225d4aa03d7325dd50f` | the workstation copy until now |
| `p01.txt` … `p08.txt` (`77-prompt512.txt`, `prompts/bitset-0*.txt`) | `fc65c158…` … `67b657c1…` (`prompts/README.md`) | |
| `run_222.sh` (= `222-run.sh`) | `aa1798210674230e9efc1d9d7ad31b35` | |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` | `7b7867fab51845664c0050c0a837073e` | NPU model (#78); `nntr_lfm2.5_8b_a1b_q40_arm.bin` = a symlink to it |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40/nntr_lfm2_8b_a1b_q40_arm.bin` | `d28f55c5bd7adeb8bf73b02de582eb88` | CPU control (#78) |

## Steps (workstation, unit on the device farm)

1. **The new model file name (user step, once, workstation and device).**
   The config of record names `nntr_lfm2.5_8b_a1b_q40_arm.bin`, which
   does not exist; it is the same bin under a new name:
   ```
   cd /local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh
   ln -s nntr_lfm2_8b_a1b_q40_arm.bin nntr_lfm2.5_8b_a1b_q40_arm.bin
   md5sum -L nntr_lfm2.5_8b_a1b_q40_arm.bin    # 7b7867fab51845664c0050c0a837073e
   cp /home/j2z0-lee/nntrainer-222/docs/measurements/config/q40-qs4cx-wh.nntr_config.json nntr_config.json   # optional: make the workstation copy the config of record
   ```
   On the device the runner creates the same symlink in
   `/data/local/tmp/nntrainer/causallm/models/q40-qs4cx-wh/` (idempotent)
   and prints both names' md5 into `logs/md5_models.log`; the old name
   stays, so the old config and every earlier runner keep working.
2. Reuse the staged set or rebuild it: `git fetch && git checkout
   htp/222-config-refresh`; `source tools/htp/env.sh`; export
   `HEXKL_ROOT=$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon
   HEXKL_SDK_VER=6.4.0.1`; `HEX_ARCH=v79 ./test/htp/build.sh && cp
   test/htp/build/libnntr_hvx_skel.so test/htp/build/libnntr_hvx_skel.v79.so`,
   the same for v81; `(cd Applications/CausalLM && ./build_android.sh
   --htp)` (fresh worktree: `git submodule update --init --depth 1`, copy
   `Applications/CausalLM/lib/libtokenizers_android_c.a`, and on a fresh
   `builddir` `meson configure -Dprefix=$PWD/android_build_result && ninja
   install` then `--htp --cache`); then `bash
   docs/measurements/222-stage.sh`. The staged `md5.txt` must equal the
   table above.
3. The unit must already hold `models/q40-qs4cx-wh/` and `models/q40/`
   under `/data/local/tmp/nntrainer/causallm/` (the #207 / #208 farm units
   do); otherwise `(cd Applications/CausalLM && ./install_android.sh
   --model=<model dir>)` for both. `adb devices` shows exactly one unit;
   note its serial and SoC (S25 = v79, S26 = v81).
4. **Reboot the unit, wait 5 min idle** (LEDGER rule 61), then
   ```
   bash /local/mnt/workspace/htp_moe/222/run_222.sh <serial> <v79|v81>
   ```
   It pushes the set (`MD5 OK`), runs G 64 (A/Q old/new, mirrored, two
   runs), G 512 and G 1024 (one run each), the CPU control per G, two
   `NNTR_HTP_PROFILE=2` runs (not speed cells), and the PPL block (8
   prompts at G 256, `NNTR_PPL=1` + `NNTR_PPL_DECODE`), with a cool start
   (zone0 ≤ 35 °C) per block. On `STOP`: reboot and run the same command;
   it resumes. If Q28 does not load (`cannot generate`), rerun with a
   third argument `24` and say so.
5. Expected lines per run: `prefill: 512 tokens, <ms> ms, <tps> TPS`,
   `generation: <G> tokens, …`, `generation(last 64): …`, `Max Resident
   Set Size`, the text. The runner's own checks (`OK` / `BAD` lines):
   one `moe m1 gemv: on (applied=…` banner per NPU run; A: `dspq: on`,
   no token driver; Q: close clean (`hops/token=0.00 … timeouts=0/0
   stale=0/0 … id_mismatch=0`), `unmap_fail=0 detach_fail=0`,
   `calls/token=1.00`, **`cpu fc skipped` = 36 × tokens (old) / 32 ×
   tokens (new)**; profiles: `M>1` rows for `conv`, `dense` and `FC` in
   both new-config runs (the keys acted); every Qold text = Aold r1 of its
   G; every Qnew text = Anew of the same G and run.
6. Paste `logs/speed.txt`, `logs/ppl.txt`, the `M>1` profile rows and
   `logs/texts.txt` below, fill the approval column, commit this file on
   the branch, push, and comment on #222 (`state:measured`).

## Results (fill in)

Reference (other sittings, not comparable cell by cell): record sitting
2026-09-30 on `R3CY10WM83Y` (S25), old config, cool: NPU A decode
**53.97 / 52.16 / 51.41**, prefill 528 / 534 / 503 (means), CPU **52.18 /
51.31 / 50.12**; Q28 one PD 43 (G 64 / 512, same unit) and 40–47 (farm
`R3CY205ZMND`, G 64–1024); S26 `R5KL20NFRCK` (#185) DQ 52.09 / 51.52 /
48.61. Goal: decode ≥ 50 and above the CPU of the same sitting; prefill
≥ −5 % of Aold.

| variant | G | run | prefill tok/s | decode tok/s (all) | decode tok/s (last 64) | peak RSS (KB) | text vs Aold r1 | calls/token | cpu fc skipped / token |
|---|---|---|---|---|---|---|---|---|---|
| Aold | 64 | 1 | | | | | (ref) | — | — |
| Anew | 64 | 1 | | | | | | — | — |
| Qold | 64 | 1 | | | | | | | 36 |
| Qnew | 64 | 1 | | | | | | | 32 |
| Qnew | 64 | 2 | | | | | | | |
| Qold | 64 | 2 | | | | | | | |
| Anew | 64 | 2 | | | | | | — | — |
| Aold | 64 | 2 | | | | | | — | — |
| CPU | 64 | 1 | | | | | | — | — |
| Aold | 512 | 1 | | | | | (ref) | — | — |
| Anew | 512 | 1 | | | | | | — | — |
| Qold | 512 | 1 | | | | | | | |
| Qnew | 512 | 1 | | | | | | | |
| CPU | 512 | 1 | | | | | | — | — |
| Aold | 1024 | 1 | | | | | (ref) | — | — |
| Anew | 1024 | 1 | | | | | | — | — |
| Qold | 1024 | 1 | | | | | | | |
| Qnew | 1024 | 1 | | | | | | | |
| CPU | 1024 | 1 | | | | | | — | — |

Device md5s (`logs/md5_device.log`, `md5_models.log`): skel ___,
`libnntrainer.so` ___, model bin (both names) ___. Unit ___ (SoC ___),
pool C ___.

Profile rows (`prof_Anew`, `prof_Qnew`, `M>1`): <paste>

## Accuracy: PPL (8 prompts, G 256) and text approval

Prefill PPL (`[PPL] prompt … ppl=`) per prompt, and decode PPL forced on
Aold's continuation (`[PPL] decode … source=file`; Aold's own run is
`source=self`, its p01 forced re-run is the null check). Fail threshold
for decode PPL: +2 % of Aold pooled over the 8 prompts (contract §1);
the prefill PPL cost of `attn_proj` / `dense_ffn` is the user's accepted
call and is read, not gated.

| prompt | Aold prefill PPL | Anew prefill PPL | Qnew prefill PPL | Aold decode PPL (self) | Anew decode PPL (forced) | Qnew decode PPL (forced) |
|---|---|---|---|---|---|---|
| p01 | | | | | | |
| p02 | | | | | | |
| p03 | | | | | | |
| p04 | | | | | | |
| p05 | | | | | | |
| p06 | | | | | | |
| p07 | | | | | | |
| p08 | | | | | | |
| pooled | | | | | | |

Null check (Aold p01 forced == self): ___

| variant | generated text (G = 64, run 1; old-config texts lack the first token) | text approved (user: y/n) |
|---|---|---|
| Aold | <paste> | (reference) |
| Anew | <paste> | |
| Qold | <paste> | |
| Qnew | <paste> | |
| CPU | <paste> | (control) |

## Notes from the run

<serial, uptime at start, thermal per block, LEAK / STOP / reboots, Q28 vs Q24, anything stale>
