# Measurement 219: a pool miss as a memcpy from a cached ARM tier — `NNTR_MOE_TIER` A/B on a fresh and an old boot

Branch `htp/219-arm-tier` @ `8449de674` (code; this file and the runner
sit on top of it). Plan `docs/plans/219-arm-tier.md` §4 step 3. **Estimated
time: ≈ 45 min** — ≈ 30 min on the device (two boots: reboot + 1 min,
then reboot + 10 min idle; 15 runs a boot, ≈ 4–5 min of runs each), plus
≈ 15 min of workstation steps (staging the config, the report, filling
this file).

**Waits for #222.** The LFM2.5 config of record is changing under #222
(PR #223: `docs/measurements/config/q40-qs4cx-wh.nntr_config.json`,
engine keys `htp`, `init_seq_len 1024`, `model_file_name
nntr_lfm2.5_8b_a1b_q40_arm.bin`). This sitting runs **after #222's
sitting** and uses, for A and B alike, whatever config that sitting makes
the record **for Q28 one PD**: the new one if #222's `Qnew` loads, the
2026-09-21 one if `Qnew` is VOID (the address budget #222's handoff
names). The runner takes that file from the staging dir, pushes it, checks
it on the device (`CONFIG OK <md5>`, `num_to_generate` excluded), and
takes the model file name from it. Note which one in "Notes from the run".

## Why

Rule 61 / ㉜: a Q28 decode miss is 0.68–4.38 ms on one boot, the slow
mode a UFS read of an expert kswapd evicted from the page cache (the model
file is held twice: arena + cache). #218's fadvise lever made the miss a
constant ≈ 1 ms but cost −7 to −18 % prefill. This lever keeps the
arena's complement (88 experts, ≈ 467 MiB) in cached anon memory, so a miss
is a copy into the ION slot and the page cache leaves the memory sum.
Decision hanging on it: flip `NNTR_MOE_TIER=1` to the default (user's
call, as #115 / #151 / #218).

## What the code does

`NNTR_MOE_TIER=1` (unset / `0` = today: no tier, no thread, the same reads
and bytes):

* **Load.** `Lfm2MoELayer::preloadExperts` hands each layer's
  not-preloaded experts to `tier_qs4cx_wh_experts` (layers 19–21 at C =
  28): each is read `O_DIRECT` into a 4 KiB-aligned tier slot (the file
  image of its two weights), then **one** whole-file
  `POSIX_FADV_DONTNEED` per call (3 calls) on the loader. Log, per call:
  `[HTP] tier: experts=<n> mib=<m> read_ms=<r> drop_ms=<d> direct=<0|1>`
  — the last says `experts=88 mib≈467`. **`direct=0` on the device is a
  stop** (the refills would fill the page cache again; the runner stops).
* **Every load after that** — decode miss (pool server), prefill miss
  batch, prefetch readers — copies from the tier instead of `pread`ing the
  file (`use_pool`: the same 8-slice `parallel_for` as the file read; the
  readers one thread each). `=2` makes the copy one thread (diagnostic: the
  uncached ION store rate). The copied bytes are the file's, and the tail
  goes through the same `float → int32` cast, so the slot is byte for byte
  what the file read writes (host: `bit_identical=1`).
* **Eviction.** The victim is queued; one refill thread reads it back
  `O_DIRECT` into the tier slot the next load frees. A load of an expert
  whose refill is pending waits for it (`tier_waits`); one queued with no
  free slot is read from the file instead (`tier_reads`, expected 0).
* **Log.** Driver-on line `tier=<knob>`; the pool line appends
  `tier_hits= tier_waits= tier_wait_us= tier_reads= refill_ms=` (decode
  window: driver on → last token). `file read … ms/miss` keeps its name:
  it is the pool server's time per miss, a memcpy now.

No DSP change (IDL / stub / skel sources untouched; the skel is rebuilt
from this branch only because `htp_decode` moved DSP sources since #218's
`9d61aef4…`); DMA ring, worker pool and weight layout untouched; no new
DSP mapping (the tier is ARM anon memory: the 3840 MiB ceiling is
unaffected). The hybrid path (H / H0) never builds a tier (its experts are
not virtual: `preloadExperts` returns first).

## Artifacts (workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79, from `8449de674`)

Staged at `/local/mnt/workspace/htp_moe/219/app/` (`md5.txt` beside it;
the runner diffs it against the device's `md5sum` and stops on a
mismatch). One set for every variant; variants are env only.

| file (`app/`) | md5 | built with |
|---|---|---|
| `libnntrainer.so` (the change) | `d12e41f417b78d2dca1ff1c8552eb476` | `build_android.sh --htp` (`jni/obj/local/arm64-v8a/`) |
| `libcausallm_core.so` (the hook) | `e3b086f0c3cb8840a2529a5e94617cee` | same (`jni/libs/arm64-v8a/`) |
| `nntrainer_causallm` | `f21e46bddbdaaebbc4628d4ac2672ca6` | same |
| `libccapi-nntrainer.so` | `02e7ea8184a95c37bdaec8009c37c509` | same (`jni/obj/local/arm64-v8a/`) |
| `libnntr_hvx_skel.so` (v79; DSP sources = `htp_decode`'s, unchanged here) | `400ab688981570769d4e15a7ab3e6de9` | `test/htp/build.sh` (`UNDEFINED SYMBOLS OK (62 runtime imports)`, `ARCH OK (V79)`) |
| `unittest_hvx_two_sessions` (the ceiling check) | `42f0a8b2873bfa01de81cfeb86053b07` | `test/jni` `ndk-build` |
| `page_cache_evict` | `42595651ef514e155887eb31f421b2dc` (= #218's) | NDK clang `-O2 tools/htp/page_cache_evict.c` |
| `libc++_shared.so` / `libsdkl.so` | `b1586b9b…` / `0ad4e22a…` | NDK r30 / HexKL 6.4.0.1 |
| `prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt` |
| `nntr_config.json` (not in `app/`) | the runner prints `CONFIG OK <md5>` | #222's config of record, staged by the user (step 1) |
| `219-arm-tier-run.sh` / `219-arm-tier-report.py` / `216-fadvise-sampler.sh` | in this branch | — |

Rebuild recipe (fresh worktree): `git submodule update --init --depth 1`;
copy `Applications/CausalLM/lib/libtokenizers_android_c.a` from another
checkout; `export HEXKL_ROOT=…/hexkl_addon HEXKL_SDK_VER=6.4.0.1`;
`libc++_shared.so` from the NDK sysroot (`b1586b9b…`).

## Steps (workstation, device-farm session with one S25)

1. **After #222's sitting**, stage its config of record for Q28 (see
   above) as `/local/mnt/workspace/htp_moe/219/nntr_config.json`, e.g.
   `git show origin/htp_decode:docs/measurements/config/q40-qs4cx-wh.nntr_config.json > /local/mnt/workspace/htp_moe/219/nntr_config.json`
   once #223 is merged (or the 2026-09-21 file if `Qnew` was VOID). If its
   `model_file_name` is `nntr_lfm2.5_8b_a1b_q40_arm.bin`, the device needs
   #222's symlink in `models/q40-qs4cx-wh/` (its runner makes it; the
   runner here stops if the file is missing).
2. `git fetch && git checkout htp/219-arm-tier`; `(cd /local/mnt/workspace/htp_moe/219/app && md5sum -c ../md5.txt)` (exit 0).
   Sanity: `strings app/libnntrainer.so | grep -c 'tier: experts'` → 1;
   `strings app/libcausallm_core.so | grep -c NNTR_HTP_FORWARD_KINDS` → ≥ 1.
3. `adb devices` lists exactly one device; note its serial, battery %,
   warm or not under Notes. The model (`models/q40-qs4cx-wh/`, as #216 /
   #222 left it) must be installed under `/data/local/tmp/nntrainer/causallm/`.
4. Fresh boot: `bash docs/measurements/219-arm-tier-run.sh <serial> fresh fresh`
   (pushes the set into `causallm/s219`, reboots, runs from uptime 60 s).
5. Old boot: `bash docs/measurements/219-arm-tier-run.sh <serial> old old`
   (reboots, idles to uptime 600 s, runs).
6. `python3 docs/measurements/219-arm-tier-report.py /local/mnt/workspace/htp_moe/219/logs/{fresh,old} > /local/mnt/workspace/htp_moe/219/logs/report.md`;
   paste its two tables and the `tier:` / `CONFIG OK` / `MD5 OK` lines
   below, commit this file on the branch, push, set #219 to `state:measured`.

Per boot the runner does, one binary set, env only (every Q run with
`NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28`, prompt 512, `NNTR_NUM_THREADS=8`,
the model file pre-read `cat` + `page_cache_evict -1` before A and B
alike; text against the boot's first A of the same G, `strip` + `cmp`;
S1's ceiling after every run, STOP < 3840):

| runs | G | variant | profile |
|---|---|---|---|
| A1 B1 A2 B2 A3 B3 A4 B4 | 64 | **A** = knob unset (reference, first) / **B** = `NNTR_MOE_TIER=1` | 4th pair `NNTR_HTP_PROFILE=2` |
| B2one | 64 | **B2** = `NNTR_MOE_TIER=2` (one-thread copy) | 2 |
| A512 B512 | 512 | A / B | 2 |
| A1024 B1024 | 1024 | A / B | – |
| H0 H | 64 | hybrid A, nothing set / `NNTR_MOE_TIER=1` (control pair) | – |

Expected lines per Q run: `prefill: 512 tokens, … TPS`, `generation: <G>
tokens, … TPS`, `[HTP] token driver: on … tier=<k>`, `[HTP] token
driver: pool misses=… pgpgin_mib=… tier_hits=… tier_waits=…
tier_wait_us=… tier_reads=… refill_ms=…`, `calls/token=1.00`, `close
tokens=… timeouts=0 stale=0`; B / B2 also three `[HTP] tier:` lines at
load, the last `experts=88 … direct=1`.

## Gate (plan §1)

| # | check | pass |
|---|---|---|
| G1 | ms/miss (profiled `file read … ms/miss`, else `arm_ms/round × rounds / misses`) | **≤ 0.5 on every B run**, both boots. 0.5–0.7 on every B run with `tier_hits = misses` and refaults ≈ 0 reads "the slow regime is gone, the copy is store-bound" (user's call). Any B run > 1.1 is a fail |
| G2 | no eviction-driven read | per B run: `tier_hits = misses`, `tier_waits ≤ 2`, `tier_reads = 0`; window `workingset_refault_file` ≤ 2 000, PSI io ≤ 70 ms, `pswpin = pswpout = 0`; `pgpgin_mib` within ± 11 MiB of `misses × 5.29` (the refills, nothing else) |
| G3 | prefill | B's prefill tok/s ≥ −5 % of the same block's A (4-run means at G = 64; single pairs at 512 / 1024 read with that caveat) |
| G4 | decode | B's G = 64 block mean ≥ A's fast regime on the unit (≥ 43.7 tok/s on `R3CY10WM83Y`, #216 old A3; on another unit: A's best G = 64 run of the block); B ≥ A at G = 512 and 1024 |
| G5 | text | `same` on every run (the tier holds the file's bytes: a `DIFF` is a fail, not an approval question) |
| G6 | hygiene | `calls/token=1.00`, close clean, ceiling 3840 after every run; H vs H0 inside the hybrid's spread |
| G7 | memory | B's `resident after` ≈ 0 MiB (A's 1–4 GiB); last `tier:` line `experts=88 mib≈467 direct=1`; no `pswpout` in B's windows |

## Results (fill in; `219-arm-tier-report.py` prints these columns)

`*` = `arm_ms/round × rounds / misses` (run not profiled). `misses × 5.29`
= the refills' expected `pgpgin`.

**Fresh boot** (reboot `<time>`, first run at uptime ≈ 60 s):

| run | cell | up s | prefill tok/s | decode tok/s | ms/miss | arm_ms/round | miss_wait us/tok | misses | tier hits / waits / reads | refill ms | drop ms | app pgpgin MiB | misses x 5.29 | win pgpgin MiB | win refault | win PSI io ms | win kswapd scan/s | win pswpin / pswpout | resident MiB in win (min-max) | resident after | text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | A G=64 prof=0 | | | | | | | | | | | | | | | | | | | | |
| B1 | B G=64 prof=0 | | | | | | | | | | | | | | | | | | | | |
| A2 | A G=64 prof=0 | | | | | | | | | | | | | | | | | | | | |
| B2 | B G=64 prof=0 | | | | | | | | | | | | | | | | | | | | |
| A3 | A G=64 prof=0 | | | | | | | | | | | | | | | | | | | | |
| B3 | B G=64 prof=0 | | | | | | | | | | | | | | | | | | | | |
| A4 | A G=64 prof=2 | | | | | | | | | | | | | | | | | | | | |
| B4 | B G=64 prof=2 | | | | | | | | | | | | | | | | | | | | |
| B2one | B2 G=64 prof=2 | | | | | | | | | | | | | | | | | | | | |
| A512 | A G=512 prof=2 | | | | | | | | | | | | | | | | | | | | |
| B512 | B G=512 prof=2 | | | | | | | | | | | | | | | | | | | | |
| A1024 | A G=1024 prof=0 | | | | | | | | | | | | | | | | | | | | |
| B1024 | B G=1024 prof=0 | | | | | | | | | | | | | | | | | | | | |
| H0 | H0 G=64 prof=0 | | | | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | | |
| H | H G=64 prof=0 | | | | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | | |

**Old boot** (reboot `<time>`, idle to uptime 600 s): the same table.

Block means (G = 64, four runs each; tok/s read only inside a block):

| boot | prefill A → B | decode A → B | ms/miss A (range) | ms/miss B (range) | ms/miss B2 |
|---|---|---|---|---|---|
| fresh | | | | | |
| old | | | | | |
| G = 512 / 1024 (one each) | | | | | – |

Reference (#216, `R3CY10WM83Y`, Q28, the 2026-09-21 config): A G = 64
33.4–43.7 tok/s at 0.68–4.38 ms/miss, prefill 462–558 tok/s; #218's B
(fadvise) 41.3–43.3 tok/s at 0.86–1.13 ms/miss, prefill −7 / −10 %; hybrid
50.7–52.2 tok/s. Goal ≥ 50 decode, prefill ≥ 497 (contract §1.1). With a
different config of record (#222) or unit, read only this sitting's A.

## Text

Every run `same` (G5). For the record, A1's and B1's generated text (G =
64, fresh boot):

| variant | generated text (G = 64, fresh, run 1) |
|---|---|
| A | <paste> |
| B | <paste> |

## Host (this branch, the workstation)

Run on `8449de674` after `source tools/htp/env.sh` (no device; no number
here is a device number):

* `tools/htp_syntax_check.sh` type-checks; `clang-format-14` clean on
  every changed hunk; `git diff --check` clean.
* `*qs4cx*` 2 / 2, `*Lfm2Moe*` 6 / 6 (none skipped);
  `run_host_checks.sh` `ALL CHECKS PASS` + `WORKER POOL LANES OK`.
* `run_inproc_e2e.sh` `INPROC E2E PASS`; the untiered pool lines as before
  (`misses=` 5 / 56 / 14), and the new tier lines — the pool preloaded as
  the app loads it (`htp_e2e_test --repack`), against the same pool
  untiered:

  ```
  E2E e3 pool tier=1 C=2 hd64 == tier=0 bit_identical=1 misses=5 tier_hits=5 tier_reads=0 calls/token=1.00 [tier: experts=4 mib=0.1 read_ms=5.0 drop_ms=0.0 direct=1]
  E2E e3 pool tier=1 C=1 lfm25 == tier=0 bit_identical=1 misses=56 tier_hits=56 tier_reads=0 calls/token=1.00 [tier: experts=12 mib=9.3 read_ms=2.3 drop_ms=0.0 direct=1]
  E2E e3 pool tier=2 C=1 lfm25 == tier=0 bit_identical=1 misses=56 tier_hits=56 tier_reads=0 calls/token=1.00 [tier: experts=12 mib=9.3 read_ms=2.1 drop_ms=0.0 direct=1]
  E2E e3 pool tier=1 C=2 lfm25 == tier=0 bit_identical=1 misses=15 tier_hits=15 tier_reads=0 calls/token=1.00 [tier: experts=8 mib=6.2 read_ms=2.3 drop_ms=0.0 direct=1]
  ```

  `tier_waits` was 0 on every line except the last on this run: 2 waits,
  22.8 ms in total, at the decode start behind the prefill's refill
  backlog (`refill_ms=42.5` in the window; the same line in the run
  before had 0 waits). The device's `tier_waits` / `tier_wait_us` (G2)
  is where that shows at scale.

  On the fixtures the refills are the whole of `pgpgin_mib=`: 43.5 MiB =
  56 misses × 0.78 MiB (lfm25 C = 1), 11.7 = 15 × 0.78 (C = 2) — G2's
  arithmetic, the device's 5.29 MiB an expert in place of 0.78.
* Rung 2: `test/htp/build.sh` `UNDEFINED SYMBOLS OK (62 runtime imports)`,
  `ARCH OK (V79)`. Rung 3: `build_android.sh --htp` (`libsdkl.so` /
  `libcdsprpc.so` NEEDED, `NNTR_HTP_FORWARD_KINDS` count 2, `tier:
  experts` in `libnntrainer.so`); device gtests `unittest_hvx_{mm_u8i4,
  softmax,attn,fc,two_sessions}` built (not run: no phone).

## Notes from the run

<serial, battery, warm; which config (`CONFIG OK` md5, new or 2026-09-21);
the three `tier:` lines of one B run (drop_ms!); STOP lines; anything stale>

## Not verified here

* Every device number (G1–G4, G7): no phone on the workstation.
* The real model file through the tier on the workstation (plan step 2's
  optional registration-only run): not run — the in-process arena is
  capped at 512 MiB and the 8B load on x86 is not a host check. The
  `O_DIRECT` path ran on the fixtures (`direct=1`, ext4 `/tmp`); f2fs
  `/data` is read on the device's `tier:` line.
* The cost of the whole-file `DONTNEED` on the S25 kernel (`drop_ms`), and
  whether `pgpgin` in B's window is only the refills (G2).
* The process RSS (plan §3's budget, anon ≈ 766 + 467 MiB): not a column
  of the runner; `MemAvailable` / `SwapFree` in `core.samples` and the
  window's `pswpout` are the proxy.
