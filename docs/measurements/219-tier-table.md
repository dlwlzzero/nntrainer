# Measurement 219 (close-out): the E2E one-PD row of the table of record with the tier on by default

Branch `htp/219-tier-default` — code @ `0235d9d79` (the staged set was built
from it; this file and the runner sit on top) — estimated device time:
**≈ 30 min** (reboot + 5 min idle, 14 runs ≈ 18 min, cool waits), plus
**≈ 5 min** of workstation steps. Device: an S25 Ultra (v79) on the farm,
run by the user.

## Why

#219's sitting (`219-arm-tier.md`) showed the tier removes the pool-miss
slow regime: Q28 one PD at P512 G64 decoded 48.47 tok/s with
`NNTR_MOE_TIER=2` (4 runs, sd 0.23) against 37.15 without it. The user
made `=2` the default on the E2E path (2026-10-06); this sitting
re-measures **the E2E row of #225's table of record** (P64 / P512 / P1024
× G64 / G512 / G1024) on the binary where nothing is set but
`NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28`, so the row of record carries
the default it ships with.

| gate | read from | pass |
|---|---|---|
| G1 tier on by default | every Q run | `token driver: on … tier=2`; the last `[HTP] tier:` line `experts=88` (≈ 466 MiB), all three `direct=1`; `tier_reads=0` on the pool line |
| G2 E2E hygiene | every Q run | `calls/token=1.00`; `[HTP] fc wh: … heap_kib=0 requant=0`; close clean (`timeouts=0 stale=0 id_mismatch=0`, `unmap_fail=0 detach_fail=0`); `mapped_mib + s1_arena_mib` ≤ 3840 (the tier is ARM memory: expected 192 + 3520 as #225) |
| G3 speed | `Q_P512_G64_r1` / `_r2` | decode ≥ #219's **48.4** (`=2` mean 48.47); Q0 of this sitting is the drift anchor against #225's 42.78 |
| G4 hybrid untouched | `B_P512_G64_r1` | `dspq: on`, no token driver, no `tier:` line; decode within the hybrid's spread of #225 (55.17). B's experts are resident, so it never reaches the tier hook whatever the knob says: B shows the hybrid of record unchanged, not the gate itself. The gate (a hybrid **with** a pool builds no tier when the variable is unset) is host-proven (`E2E tier unset … hybrid no tier ok`); that the same hybrid builds one with `=1` was run by hand on the host (PR body) |
| T2 texts | `logs/texts.txt`, `logs/loops.txt` | recorded, not gated; Q P512 G64 text == Q0's (the tier copies the file's bytes); r2 == r1; a loop where #225's A has none is noted; the user approves |

## What changed in the code

`NNTR_MOE_TIER` unset now means **2** when `NNTR_HTP_E2E=1` and **0**
otherwise (`tierKnob()` in `htp_compute_ops.cpp`). `=0` stays the explicit
off (#225's Q), `=1` the 8-slice copy. The tier is only ever built for a
layer whose experts are virtual (`NNTR_MOE_CACHE_EXPERTS` on an HTP engine);
the hybrid and a CPU-only run build none, with or without the pool (host
line `E2E tier unset lfm25 C=1: … hybrid no tier ok`). No DSP source, IDL
or kernel change.

## What to expect (arithmetic, not measured)

* **Q:** load prints three `[HTP] tier:` lines (layers 19–21 at C = 28),
  the last `experts=88 mib=466.5 … direct=1`; the first call's whole-file
  `DONTNEED` costs ≈ 0.45 s of load. Decode at P512 G64 ≈ 48.4 (#219 C1–C4),
  `miss_wait_us/token` ≈ 300, ≈ 0.44 ms/miss, `tier_waits` 3–4. P64 / P1024
  and G512 / G1024 have never run with the tier: #225's Q row (42.81 /
  49.34 / 48.49; 42.78 / 45.94 / 45.61; 37.83 / 43.93 / 43.52) is the
  untiered reference. Prefill ≈ #225's (the tier is decode-side; #219 read
  +1.7 %).
* **Q0:** #225's Q (no tier) — 42.78 at P512 G64 on #225's 52-s-old boot;
  #219's A ranged 29.6–42.6 on one boot, so Q0 is read as an anchor, not
  a gate.
* **B:** #225's hybrid, 55.17 / 727.3 at P512 G64.
* **Texts:** Q and Q0 run the same arithmetic (tier bytes = file bytes;
  host `bit_identical=1`): text equal. Q differs from B and from the CPU
  (rule 39).

## Variants (one binary set; env only; ≤ 4)

| variant | config | env | cells |
|---|---|---|---|
| **Q** | `models/q40-qs4cx-wh`, `cfg_new.json` = the config of record (md5 `3f6808e3…`, sidecar `71812a91…`, `init_seq_len 1024`) | `NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28` (tier unset = 2); model file pre-read + `page_cache_evict -1` | 9 + G64 r2 at each P = 12 |
| **Q0** | as Q | Q + `NNTR_MOE_TIER=0` | P512 × G64 (1) |
| **B** (control: the hybrid of record, experts resident) | as Q | nothing | P512 × G64 (1) |

There is no CPU A in this sitting: the row is read against #225's table
(same unit class, same config, same prompts) through Q0 and B, the
in-sitting anchors. Overlay per run (the runner applies it):
`num_to_generate` = G, `bad_word_ids [124900]`; `do_sample false`.
`NNTR_NUM_THREADS=8`. Prompts: **P64** = `p64.txt`, **P512** = `p01.txt`
(`77-prompt512.txt`), **P1024** = `p1024.txt` (`prompts/README.md`).

Order (`zone0` ≤ 35 °C at every block start): for P in 64, 512, 1024:
G64 `Q`, cool, `Q` (r2) [P512: `Q0`, `B`]; G512 `Q`; G1024 `Q`.
Run count: 12 + 1 + 1 = **14**.

## Same farm session as #236

`236-run.sh` (hybrid P1024, `htp/236-qkv-chunks`, ≈ 12 min) can run in the
same farm session. **Order: #236 first, then reboot + 5 min idle, then
this runner.** #236 is the hybrid only and needs no clean page cache;
this sitting's Q runs are the ones rule 61 says are boot-age sensitive, so
they get the fresh boot. The two use separate device dirs (`s236`,
`s219t`) and both leave the config of record pristine.

## Artifacts (workstation, SDK 6.4.0.1, HexKL 6.4.0.1 (`hexkl-1.0-beta.2`), NDK r30; built from `0235d9d79`)

Staged at `/local/mnt/workspace/htp_moe/219t/` by `219-tier-stage.sh`
(`md5.txt` = `058b97a5f67d30aea52852dbb6dc52c1`; the runner checks it on both ends).

| file | md5 | built with |
|---|---|---|
| `app/libnntr_hvx_skel.v79.so` (S25) | `1b35bff182c2e91b16dd641032ce39bf` | `HEX_ARCH=v79 test/htp/build.sh` (`ARCH OK (V79)`, `UNDEFINED SYMBOLS OK (62 runtime imports)`) |
| `app/libnntr_hvx_skel.v81.so` (S26, not used here) | `3a0677b63e487da0f2350afc146ae7af` | `HEX_ARCH=v81 test/htp/build.sh` (`ARCH OK (V81)`) |
| `app/nntrainer_causallm` | `7dcec2f6b607c96025575d97b2cf09c3` | `build_android.sh --htp`, then `--htp --cache` after `ninja -C builddir install` |
| `app/libcausallm_core.so` | `b2d51d1962c25b0f631df02a001bef16` | `NNTR_HTP_FORWARD_KINDS` strings: 2 |
| `app/libnntrainer.so` (`jni/obj/local`) | `c28fb0c0041b417d25c95377def88145` | NEEDED `libsdkl.so`, `libcdsprpc.so`; `graph: forward calls` strings: 1 |
| `app/libccapi-nntrainer.so` (`jni/obj/local`) | `b4463bce8f4ba2d84184653b17907bd4` | |
| `app/libc++_shared.so` (NDK r30) / `app/libsdkl.so` | `b1586b9b512712800fd36a24abac1c0a` / `0ad4e22a70e4f135bce38ad8fd1e001b` | as #225 |
| `app/page_cache_evict` | `42595651ef514e155887eb31f421b2dc` | NDK clang, `tools/htp/page_cache_evict.c` (= #225's) |
| `app/cfg_new.json` = `config/q40-qs4cx-wh.nntr_config.json` | `3f6808e30b6e16e1c8592fa39977814c` | the config of record |
| `app/p64.txt` / `app/p01.txt` / `app/p1024.txt` | `c0d3e9ff…` / `fc65c158…` / `2e47c5f4…` | `prompts/p64.txt`, `77-prompt512.txt`, `prompts/p1024.txt` |
| `model/nntr_lfm2_8b_a1b_q40_arm_fcwh.bin` | `71812a91d5acdbe9e026c479db8e275e` | #225 PR 1's sidecar |
| `run_219t.sh` (= `219-tier-run.sh`) / `loop_check.py` | `95d700df260505a905fdf4a2fcf4e66e` / `e607c5f7d4638a6eb199ac0d572867c1` | this branch / `tools/htp/loop_check.py` (host only) |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` (on the unit) | `7b7867fab51845664c0050c0a837073e` | NPU model (#78); the config names `nntr_lfm2.5_…` = a symlink the runner makes |

The skels are **not** #225's `58e3a85f…` / `ca25ec2d…`: the base
(`htp_first_version`) carries #232's DSP and IDL changes (Gemma 4 decode
binding, `adb9246f7`), so the stub in `libnntrainer.so` needs a skel from
this tree. This branch changes no DSP source.

## Steps (workstation → farm session, unit on USB)

1. **The set (workstation).** Already staged; to rebuild:
   `git fetch && git checkout htp/219-tier-default`, `source
   tools/htp/env.sh`, `HEX_ARCH=v79 ./test/htp/build.sh && cp
   test/htp/build/libnntr_hvx_skel.so test/htp/build/libnntr_hvx_skel.v79.so`
   (the same for v81), `(cd Applications/CausalLM && ./build_android.sh
   --htp)`, then `bash docs/measurements/219-tier-stage.sh <checkout>`.
   Either way:
   ```
   cd /local/mnt/workspace/htp_moe/219t && md5sum md5.txt   # 058b97a5f67d30aea52852dbb6dc52c1
   md5sum -c md5.txt | grep -vc ': OK$'                       # 0
   ```
   If the farm runs on another machine, copy the whole
   `/local/mnt/workspace/htp_moe/219t/` there (≈ 260 MB incl. the sidecar).
2. The unit holds `models/q40-qs4cx-wh/` under
   `/data/local/tmp/nntrainer/causallm/` (the #225 farm units do); the
   runner pushes the sidecar itself when its md5 differs. `adb devices`
   shows one unit; note its serial.
3. **Reboot the unit, wait 5 min idle** (rule 61), then
   ```
   bash /local/mnt/workspace/htp_moe/219t/run_219t.sh <serial>
   ```
   `MD5 OK (app, sidecar)` first; on `STOP` reboot and run the same
   command (resumes). A VOID is not a STOP.
4. Expected lines per Q run (the runner's `OK` / `BAD`; the last line
   counts the BADs): `prefill: <P> tokens`; one `moe m1 gemv: on`;
   `token driver: on … tier=2`; `[HTP] tier: experts=88 mib=466.5 …
   direct=1` (three tier lines, all `direct=1`); `fc wh: … heap_kib=0
   requant=0`; close clean; `e2e: close … unmap_fail=0 detach_fail=0`;
   `calls/token=1.00`; `tier_reads=0`; mapped + s1_arena ≤ 3840; then the
   `tier:`, `e2e: fc arena`, `e2e: close` and `token driver: pool …
   miss_wait_us/token=… tier_hits=… tier_waits=… tier_reads=…` lines.
   Q0: `tier=0`, no `tier:` line. B: `dspq: on`, no token driver, no
   `tier:` line. Summary: `logs/speed.txt` (per run: TPS, RSS, the tier
   line, misses / `miss_wait_us/token`, `tier_hits / waits / reads`,
   `ceiling=<mapped>+<s1_arena>`), `every r2 text == its r1`, `Q == Q0
   text`, `logs/loops.txt`, `logs/texts.txt`, VOID list, `logs/therm.log`.
5. Paste `logs/speed.txt`, the check lines of `Q_P512_G64_r1`, `Q0` and
   `B`, `logs/loops.txt` and `logs/texts.txt` below, fill the approval
   column, commit this file on the branch, push, and comment on #219
   (`state:measured`).

## Results (fill in)

Reference (#225 sitting 2026-10-06, farm `R3CY205ZMND`, decode / prefill
tok/s; Q = no tier): CPU A P64 54.05 / 52.54 / 51.48, P512 52.07 / 50.70 /
49.47, P1024 47.58 / 48.82 / 47.41; hybrid B P512 G64 55.17 / 727.3; Q
P64 42.81 / 49.34 / 48.49, P512 42.78 / 45.94 / 45.61, P1024 37.83 /
43.93 / 43.52 (G 64 / 512 / 1024). #219 `=2` P512 G64: 48.47 (4 runs).
Goal: decode ≥ 50 and above the CPU of the same cell.

| case | P | G 64 (r1 / r2) | G 512 | G 1024 |
|---|---|---|---|---|
| E2E one PD, tier default (Q) | 64 | | | |
| E2E one PD, tier default (Q) | 512 | | | |
| E2E one PD, tier default (Q) | 1024 | | | |
| Q0 (`NNTR_MOE_TIER=0`) | 512 | | – | – |
| hybrid B (control) | 512 | | – | – |

| run | prefill tok/s | decode (all) | decode (last 64) | peak RSS KB | tier: experts / mib | misses, miss_wait_us/token | tier hits / waits / reads | mapped + s1_arena MiB | text |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

## Text approval (T2)

| variant | generated text (G=64, run 1) | text approved (user: y/n) |
|---|---|---|
| Q P64 | | |
| Q P512 | | |
| Q P1024 | | |
| Q0 P512 | | |
| B P512 | | |

## Notes from the run

<unit serial, boot time / uptime at the first run, thermal, VOIDs, anything stale>

## Not verified here

* Every device number: no phone on the workstation (agents do not run adb).
* The skels' md5 equality with #225's set: not attainable (base DSP
  changes; and a rebuild of the same tree is not byte-reproducible on this
  workstation — two consecutive `build.sh` runs of one checkout gave
  different md5s).
