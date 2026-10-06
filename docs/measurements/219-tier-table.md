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

Read 2026-10-06 20:50–21:02 KST on `R3CY205ZMND` (SM-S938N, SM8750, v79),
attached to the workstation, run by the agent; `MD5 OK (app, sidecar)`,
`md5.txt` = `058b97a5…`; 14 runs, 0 BAD, 0 VOID (`expectation mismatches
(this invocation): 0`). Decode tok/s, all tokens (prefill tok/s in brackets):

| case | P | G 64 (r1 / r2) | G 512 | G 1024 |
|---|---|---|---|---|
| E2E one PD, tier default (Q) | 64 | **52.20 / 52.03** (281.9 / 283.2) | **54.60** (285.7) | **53.89** (287.0) |
| E2E one PD, tier default (Q) | 512 | **48.08 / 48.05** (793.8 / 801.3) | **51.79** (802.5) | **51.65** (800.0) |
| E2E one PD, tier default (Q) | 1024 | **44.02 / 44.14** (765.9 / 762.5) | **50.59** (773.4) | **50.78** (771.1) |
| Q0 (`NNTR_MOE_TIER=0`) | 512 | 37.58 (750.7) | – | – |
| hybrid B (control) | 512 | 56.44 (749.6) | – | – |

Against the references (decode, all tokens; G 64 r1 / 512 / 1024):

| P | vs #225 Q (no tier) | vs #225 CPU A |
|---|---|---|
| 64 | +21.9 % / +10.7 % / +11.1 % | −3.4 % / +3.9 % / +4.7 % |
| 512 | +12.4 % / +12.7 % / +13.2 % | −7.7 % / +2.1 % / +4.4 % |
| 1024 | +16.4 % / +15.2 % / +16.7 % | −7.5 % / +3.6 % / +7.1 % |

| run | prefill tok/s | decode (all) | decode (last 64) | peak RSS KB | tier: experts / mib | misses, miss_wait_us/token | tier hits / waits / reads | mapped + s1_arena MiB | text |
|---|---|---|---|---|---|---|---|---|---|
| Q_P64_G64_r1 | 281.938 | 52.2023 | 52.2023 | 1294324 | 88 / 466.5 | 74 (1.16/tok), 157.2 | 74 / 0 / 0 | 192 + 3520 | loop (= CPU, #225) |
| Q_P64_G64_r2 | 283.186 | 52.0325 | 52.0325 | 1275192 | 88 / 466.5 | 74 (1.16/tok), 157.5 | 74 / 0 / 0 | 192 + 3520 | == r1 |
| Q_P64_G512_r1 | 285.714 | 54.5959 | 53.9174 | 1272920 | 88 / 466.5 | 87 (0.17/tok), 21.1 | 87 / 0 / 0 | 192 + 3520 | loop |
| Q_P64_G1024_r1 | 286.996 | 53.8891 | 52.5452 | 1273528 | 88 / 466.5 | 88 (0.09/tok), 10.4 | 88 / 0 / 0 | 192 + 3520 | loop |
| Q_P512_G64_r1 | 793.798 | 48.0841 | 48.0841 | 1314212 | 88 / 466.5 | 103 (1.61/tok), 330.9 | 103 / 3 / 0 | 192 + 3520 | no loop; == Q0 |
| Q_P512_G64_r2 | 801.252 | 48.0480 | 48.0480 | 1313964 | 88 / 466.5 | 103 (1.61/tok), 317.4 | 103 / 3 / 0 | 192 + 3520 | == r1 |
| Q_P512_G512_r1 | 802.508 | 51.7852 | 51.4883 | 1312540 | 88 / 466.5 | 120 (0.23/tok), 42.6 | 120 / 4 / 0 | 192 + 3520 | loop (L1run 17) |
| Q_P512_G1024_r1 | 800.000 | 51.6467 | 50.7132 | 1311848 | 88 / 466.5 | 124 (0.12/tok), 21.7 | 124 / 4 / 0 | 192 + 3520 | loop (L1run 35) |
| Q_P1024_G64_r1 | 765.894 | 44.0165 | 44.0165 | 1337108 | 88 / 466.5 | 168 (2.62/tok), 475.2 | 168 / 3 / 0 | 192 + 3520 | no loop |
| Q_P1024_G64_r2 | 762.472 | 44.1379 | 44.1379 | 1338880 | 88 / 466.5 | 168 (2.62/tok), 472.1 | 168 / 3 / 0 | 192 + 3520 | == r1 |
| Q_P1024_G512_r1 | 773.414 | 50.5929 | 51.0774 | 1338968 | 88 / 466.5 | 193 (0.38/tok), 64.3 | 193 / 3 / 0 | 192 + 3520 | loop |
| Q_P1024_G1024_r1 | 771.084 | 50.7785 | 49.9220 | 1338904 | 88 / 466.5 | 199 (0.19/tok), 30.7 | 199 / 3 / 0 | 192 + 3520 | loop |
| Q0_P512_G64_r1 | 750.733 | 37.5807 | 37.5807 | 837512 | none (`tier=0`) | 103 (1.61/tok), 5904.8 | 0 / 0 / 0 | 192 + 3520 | no loop |
| B_P512_G64_r1 | 749.634 | 56.4374 | 56.4374 | 4977656 | none (`dspq: on`, no driver) | – | – | – | no loop |

Every number above is read from the per-run `logs/<run>.log` (`prefill:`,
`generation:`, `generation(last 64):`, the `[HTP] tier:` and `token driver:
pool` lines) and matches `logs/speed.txt`. Per-run `[HTP] tier:` lines: three
per Q run (24 / 56 / 88 experts, 127.2 / 296.8 / 466.5 MiB, read 56–79 ms
each, all `direct=1`); the first one's `drop_ms` is 320–363 ms at P64 / P512
G64 and 796–909 ms from P512 G512 on.

### Gates

| gate | verdict | evidence |
|---|---|---|
| G1 tier on by default | **PASS** | all 12 Q runs: `token driver: on … tier=2`, last tier line `experts=88 mib=466.5 … direct=1` (three `direct=1`), `tier_reads=0`, `tier_hits == misses` |
| G2 E2E hygiene | **PASS** | `calls/token=1.00`, `fc wh … heap_kib=0 requant=0`, close clean (`timeouts=0 stale=0 id_mismatch=0`, `unmap_fail=0 detach_fail=0`), `ceiling=192+3520` ≤ 3840 on every Q / Q0 run |
| G3 speed (Q P512 G64 ≥ #219's 48.4) | **met within drift, not a clean pass** | 48.08 / 48.05, −0.7 % below the line, inside drift (#219's `=2` sd 0.23 over 4 runs; same-boot r1/r2 spread 0.04). Not a regression of the tier: +27.9 % over this sitting's Q0 (37.58) and +12.4 % over #225's untiered Q (42.78) |
| G4 hybrid untouched | **PASS** | B `dspq: on`, no token driver, no `tier:` line; 56.44 = #236's B P512 G64 56.49 the same evening (−0.1 %), +2.3 % over #225's 55.17 |
| T2 texts | recorded; approval: user | Q P512 G64 text == Q0's; every G64 r2 == r1 |

### Readings

* At **G ≥ 512 the tiered E2E reads 50.6–54.6 tok/s, ≥ 50 at every P**, and
  is above #225's CPU A of the same cell at every P (+2.1 % … +7.1 %).
* At **G64 it reads 44–52 depending on P** (misses/token 1.16 / 1.61 / 2.62 at
  P64 / P512 / P1024, `miss_wait_us/token` 157 / 317–331 / 472–475), below
  the CPU at every P (−3.4 % / −7.7 % / −7.5 %). The residual cost at G64 is
  the first-token pool misses, which longer G amortises (10–64 us/token at
  G ≥ 512).
* E2E P512 is now above the CPU's 52.07 / 50.70 / 49.47 at G512 / G1024
  (51.79 / 51.65) and below at G64 (48.08).
* The hybrid (56.44) stays the fastest configuration at P512 G64.
* The tier is worth +10.7 % … +21.9 % over #225's untiered Q in every cell;
  the largest gain is P64 G64.
* Prefill at P512: Q 794–803 vs Q0 751 / B 750 this sitting (+6 %; #219 read
  +1.7 %). Q0 and B ran without a cool-wait before them (see notes), so the
  gap is not attributed to the tier here.

## Text approval (T2)

| variant | generated text (G=64, run 1) | text approved (user: y/n) |
|---|---|---|
| Q P64 | s in the air, and for most of its history it has lived by the tide, and for most of its history it has lived by the tide, and for most of its history it has lived by the tide, and for most of its history it has lived by the tide, and for most of its history it has | user |
| Q P512 | In the same style, add more detail about its history, its people, its weather and the seasons, and do not stop until you are told to. In the same style, add more detail about its people, its weather and the seasons, and do not stop until you are told to. In the same style, add | user |
| Q P1024 | . Thus, the final output is a JSON object with these keys: "customer_name", "email", "items", "delivery_date", "express", "total_eur". The values are to be filled from the order note. The order note does not provide explicit values for these keys, so we must infer | user |
| Q0 P512 | In the same style, add more detail about its history, its people, its weather and the seasons, and do not stop until you are told to. In the same style, add more detail about its people, its weather and the seasons, and do not stop until you are told to. In the same style, add | user |
| B P512 | In winter the wind comes straight off the water and the streets empty by four in the afternoon, but in summer the population nearly doubles as visitors arrive to walk the cliff paths, watch the seabirds, and eat fish and chips on the harbour wall while the gulls circle overhead hoping for scraps. The people of Ardley | user |

Q P64 loops (as the CPU on this prompt, #225); Q P512 == Q0 P512
(the tier copies the file's bytes); Q P1024 starts with `.` and a newline. Texts verbatim in `logs/texts.txt`.

## Notes from the run

* Unit `R3CY205ZMND` (SM-S938N, SM8750, v79), on the workstation's USB, run
  by the agent (not the farm, not the user); `run_219t.sh R3CY205ZMND`.
* **Deviation (rule 61):** reboot at ≈ 20:45:40, the run started 20:50:07:
  ≈ 4.5 min idle, not 5. The runner's own uptime line reads `317.11 s` at
  20:50:07 (kernel boot ≈ 20:44:50, 5.3 min), which does not agree with the
  20:45:40 reboot time; recorded as the deviation either way. The G64 r1 /
  r2 pairs agree to ≤ 0.3 % and Q0 sits in #219 A's range, so no stale-boot
  effect is visible.
* Thermal (`zone0`): 30.6 °C at t0; 58.9 / 59.3 / 57.4 °C at the P64 / P512
  / P1024 block boundaries, 57.0 °C at the end. The P-blocks ran back to
  back, but **the runner's cool-wait fired before every Q cell** (threshold
  35 °C, 30 s × 1–2 waits each; `block start zone0` 31.0–34.9 °C), so every
  Q run started ≤ 35 °C. Q0 and B ran directly after `Q_P512_G64_r2` with
  no cool-wait (as the runner's order has it); their starting temperature is
  not logged.
* 14 runs, 0 BAD, 0 VOID, 0 STOP. Each logcat carries the usual 8 fastrpc
  `E` lines (`open_shell … Permission denied`, `enable_kernel_optimizations`,
  `log_config` watcher, `libdspqueue_rpc_skel` method 3, notif thread exit),
  the same in every run including B; none affected a run.
* Config: `device configs: ../models/q40-qs4cx-wh = the config of record
  (pristine), generation config restored`; `do_sample false`.
* Loops (`loop_check.py`, T2, not gated): every P64 cell (the P64 prompt
  loops on the CPU too, #225), P512 G512 (L1run 17) / G1024 (L1run 35),
  P1024 G512 / G1024; none at P512 G64, P1024 G64, Q0, B.
* Logs: `/local/mnt/workspace/htp_moe/219t/logs/` (`speed.txt`, `loops.txt`,
  `texts.txt`, `therm.log`, `config.log`, `md5_*.log`, per-run `*.log` /
  `*.logcat`, `sitting.out`).

## Not verified here

* Every device number: no phone on the workstation (agents do not run adb).
* The skels' md5 equality with #225's set: not attainable (base DSP
  changes; and a rebuild of the same tree is not byte-reproducible on this
  workstation — two consecutive `build.sh` runs of one checkout gave
  different md5s).
