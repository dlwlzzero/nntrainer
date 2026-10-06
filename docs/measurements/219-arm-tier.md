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

**Fresh boot** (the user's reboot seen 18:27:07 KST 2026-10-06, `boot_completed` at uptime 19 s, first run at uptime 62 s):

| run | cell | up s | prefill tok/s | decode tok/s | ms/miss | arm_ms/round | miss_wait us/tok | misses | tier hits / waits / reads | refill ms | drop ms | app pgpgin MiB | misses x 5.29 | win pgpgin MiB | win refault | win PSI io ms | win kswapd scan/s | win pswpin / pswpout | resident MiB in win (min-max) | resident after | text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | A G=64 prof=0 | 62 | 766.467 | 37.405 | 4.13* | 6.262 | 6289.1 | 103 | 0 / 0 / 0 | 0.0 | - | 554.0 | 545 | 554 | 143213 | 155 | 112746 | 47 / 10359 | 957-957 | 1015 | same |
| B1 | B G=64 prof=0 | 74 | 811.41 | 47.7256 | 0.68* | 1.030 | 689.5 | 103 | 103 / 3 / 0 | 248.1 | 0.0 | 549.8 | 545 | 543 | 494 | 23 | 0 | 48 / 0 | 0-0 | 0 | same |
| A2 | A G=64 prof=0 | 82 | 782.875 | 38.835 | 3.43* | 5.189 | 5151.3 | 103 | 0 / 0 / 0 | 0.0 | - | 544.3 | 545 | 528 | 135190 | 145 | 126459 | 62 / 0 | 2654-2654 | 2645 | same |
| B2 | B G=64 prof=0 | 91 | 812.698 | 47.7969 | 0.64* | 0.967 | 625.2 | 103 | 103 / 4 / 0 | 248.1 | 0.0 | 549.0 | 545 | 533 | 571 | 29 | 0 | 33 / 0 | 0-0 | 0 | DIFF |
| A3 | A G=64 prof=0 | 100 | 786.482 | 42.4403 | 2.05* | 3.099 | 2927.0 | 103 | 0 / 0 / 0 | 0.0 | - | 305.3 | 545 | 299 | 75860 | 81 | 87062 | 701 / 0 | 2906-2906 | 2893 | same |
| B3 | B G=64 prof=0 | 110 | 787.692 | 47.3373 | 0.73* | 1.101 | 768.3 | 103 | 103 / 4 / 0 | 242.9 | 0.0 | 546.7 | 545 | 541 | 29 | 33 | 0 | 163 / 0 | 0-0 | 0 | same |
| A4 | A G=64 prof=2 | 118 | 774.584 | 41.2637 | 2.54 | 3.850 | 3720.9 | 103 | 0 / 0 / 0 | 0.0 | - | 295.4 | 545 | 290 | 73974 | 83 | 65379 | 44 / 0 | 2857-2857 | 2862 | same |
| B4 | B G=64 prof=2 | 127 | 763.04 | 47.5483 | 0.67 | 1.019 | 678.7 | 103 | 103 / 4 / 0 | 252.6 | 0.0 | 548.5 | 545 | 530 | 0 | 23 | 0 | 0 / 0 | 0-0 | 0 | same |
| B2one | B2 G=64 prof=2 | 135 | 764.179 | 48.3749 | 0.45 | 0.677 | 315.9 | 103 | 103 / 3 / 0 | 236.5 | 0.0 | 546.0 | 545 | 541 | 0 | 37 | 0 | 0 / 0 | - | 0 | same |
| A512 | A G=512 prof=2 | 145 | 767.616 | 51.1335 | 2.86 | 4.040 | 611.8 | 120 | 0 / 0 / 0 | 0.0 | - | 441.9 | 635 | 442 | 112691 | 124 | 13194 | 569 / 0 | 2838-2870 | 2845 | same |
| B512 | B G=512 prof=2 | 163 | 758.519 | 52.7454 | 0.53 | 0.752 | 60.1 | 120 | 120 / 4 / 0 | 310.3 | 0.0 | 638.7 | 635 | 639 | 194 | 30 | 0 | 188 / 0 | 0-0 | 0 | same |
| A1024 | A G=1024 prof=0 | 180 | 801.252 | 51.6937 | 2.62* | 3.648 | 285.7 | 124 | 0 / 0 / 0 | 0.0 | - | 420.9 | 656 | 421 | 97854 | 111 | 7266 | 5938 / 51 | 2667-2883 | 2667 | same |
| B1024 | B G=1024 prof=0 | 208 | 757.396 | 52.2662 | 0.69* | 0.961 | 49.3 | 124 | 124 / 4 / 0 | 330.8 | 0.0 | 658.0 | 656 | 658 | 146 | 27 | 0 | 54 / 0 | 0-0 | 0 | same |
| H0 | H0 G=64 prof=0 | 235 | 744.186 | 50.3541 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 2068 | DIFF |
| H | H G=64 prof=0 | 253 | 748.538 | 51.0774 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 2083 | DIFF |

**Old boot** (the user's reboot seen 18:48:05, `boot_completed` at uptime 17 s, idle to uptime 600 s, first run at 601 s):

| run | cell | up s | prefill tok/s | decode tok/s | ms/miss | arm_ms/round | miss_wait us/tok | misses | tier hits / waits / reads | refill ms | drop ms | app pgpgin MiB | misses x 5.29 | win pgpgin MiB | win refault | win PSI io ms | win kswapd scan/s | win pswpin / pswpout | resident MiB in win (min-max) | resident after | text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | A G=64 prof=0 | 601 | 821.83 | 39.072 | 3.52* | 5.331 | 5306.0 | 103 | 0 / 0 / 0 | 0.0 | - | 549.0 | 545 | 533 | 136392 | 139 | 123979 | 14 / 0 | 713-713 | 714 | same |
| B1 | B G=64 prof=0 | 610 | 782.875 | 47.1281 | 0.75* | 1.130 | 796.4 | 103 | 103 / 4 / 0 | 241.6 | 0.0 | 551.6 | 545 | 644 | 130 | 26 | 0 | 1261 / 0 | 0-0 | 0 | same |
| A2 | A G=64 prof=0 | 619 | 754.05 | 37.4927 | 4.02* | 6.096 | 6113.7 | 103 | 0 / 0 / 0 | 0.0 | - | 537.8 | 545 | 533 | 136304 | 146 | 89976 | 45 / 0 | 3233-3233 | 3219 | same |
| B2 | B G=64 prof=0 | 628 | 796.267 | 48.012 | 0.56* | 0.855 | 505.7 | 103 | 103 / 4 / 0 | 246.6 | 0.0 | 546.0 | 545 | 530 | 0 | 23 | 0 | 3 / 0 | 0-0 | 0 | same |
| A3 | A G=64 prof=0 | 636 | 788.906 | 39.8258 | 3.04* | 4.610 | 4538.7 | 103 | 0 / 0 / 0 | 0.0 | - | 424.0 | 545 | 413 | 105592 | 112 | 92966 | 2 / 0 | 3140-3140 | 3131 | same |
| B3 | B G=64 prof=0 | 645 | 813.99 | 47.976 | 0.60* | 0.906 | 556.6 | 103 | 103 / 4 / 0 | 241.5 | 0.0 | 546.0 | 545 | 546 | 40 | 26 | 0 | 10 / 0 | 0-0 | 0 | same |
| A4 | A G=64 prof=2 | 653 | 769.925 | 38.3923 | 3.77 | 5.715 | 5713.5 | 103 | 0 / 0 / 0 | 0.0 | - | 543.4 | 545 | 533 | 136163 | 140 | 97099 | 121 / 0 | 3120-3120 | 3112 | same |
| B4 | B G=64 prof=2 | 662 | 768.769 | 48.1565 | 0.51 | 0.772 | 419.8 | 103 | 103 / 4 / 0 | 248.9 | 0.0 | 546.0 | 545 | 535 | 0 | 21 | 0 | 0 / 0 | 0-0 | 0 | same |
| B2one | B2 G=64 prof=2 | 669 | 771.084 | 48.4481 | 0.44 | 0.667 | 316.9 | 103 | 103 / 4 / 0 | 239.2 | 0.0 | 546.4 | 545 | 546 | 0 | 33 | 0 | 4 / 0 | 0-0 | 0 | same |
| A512 | A G=512 prof=2 | 679 | 766.467 | 50.5429 | 3.64 | 5.141 | 795.1 | 120 | 0 / 0 / 0 | 0.0 | - | 600.4 | 635 | 600 | 153489 | 173 | 18604 | 37 / 0 | 3023-3071 | 3023 | same |
| B512 | B G=512 prof=2 | 697 | 758.519 | 52.7291 | 0.58 | 0.813 | 67.9 | 120 | 120 / 4 / 0 | 313.7 | 0.0 | 636.1 | 635 | 636 | 14 | 24 | 0 | 11 / 0 | 0-0 | 0 | same |
| A1024 | A G=1024 prof=0 | 714 | 779.3 | 51.2179 | 3.15* | 4.385 | 350.1 | 124 | 0 / 0 / 0 | 0.0 | - | 523.0 | 656 | 523 | 133772 | 148 | 7840 | 222 / 0 | 3009-3058 | 3020 | same |
| B1024 | B G=1024 prof=0 | 742 | 788.906 | 52.4161 | 0.60* | 0.841 | 38.9 | 124 | 124 / 4 / 0 | 330.7 | 0.0 | 657.8 | 656 | 658 | 172 | 25 | 0 | 55 / 0 | 0-0 | 0 | same |
| H0 | H0 G=64 prof=0 | 769 | 754.05 | 53.5117 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 2150 | DIFF |
| H | H G=64 prof=0 | 787 | 764.179 | 53.9174 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 2198 | DIFF |

Text column: B2 (fresh) reads `DIFF` because a load banner's tail
(` wh_handles=112`) interleaved onto the line after the prompt echo and
the runner's `strip` dropped the prompt with it; with that fragment
removed, B2's generated text is byte-identical to A1's. H0 / H read
`DIFF` against A1 by construction on this config: the hybrid's decode
FCs are the CPU's Q4_0 GEMV, Q28's the DSP WH GEMV (#225 PR 2). H == H0
(both boots), fresh H0 == old H0, fresh A1 == old A1 (generated text,
`[HTP]` banners removed). `app pgpgin` is within 7 MiB of
`misses × 5.29` on every B run; `win pgpgin` (the sampler's window)
is within ±11 MiB on 10 of 14 B runs (fresh B2 −12, B4 −15; old B1 +99,
B2 −15).

Block means (G = 64, four runs each; tok/s read only inside a block):

| boot | prefill A → B | decode A → B | ms/miss A (range) | ms/miss B (range) | ms/miss B2 |
|---|---|---|---|---|---|
| fresh | 777.6 → 793.7 (+2.1 %) | 39.99 (37.41–42.44) → 47.60 (47.34–47.80), +19.0 % | 2.05–4.13 (A4 profiled 2.54) | 0.64–0.73 (B4 profiled 0.67) | 0.45 |
| old | 783.7 → 790.5 (+0.9 %) | 38.70 (37.49–39.83) → 47.82 (47.13–48.16), +23.6 % | 3.04–4.02 (A4 profiled 3.77) | 0.51–0.75 (B4 profiled 0.51) | 0.44 |
| G = 512 / 1024 (one each) | fresh 767.6 → 758.5 (−1.2 %) / 801.3 → 757.4 (−5.5 %); old 766.5 → 758.5 (−1.0 %) / 779.3 → 788.9 (+1.2 %) | fresh 51.13 → 52.75 / 51.69 → 52.27; old 50.54 → 52.73 / 51.22 → 52.42 | 2.62–3.64 | 0.53–0.69 | – |

Gate reading (plan §1):

| # | reading | verdict |
|---|---|---|
| G1 | B 0.51–0.75 ms/miss (profiled 0.67 / 0.51; G 512 0.53 / 0.58), every B run with `tier_hits = misses` and refaults ≤ 571: the 0.5–0.7 band, "the slow regime is gone, the copy is store-bound" (user's call). B2 (one-thread copy) 0.45 / 0.44 | **not ≤ 0.5** for B; B2 passes |
| G2 | `tier_hits = misses`, `tier_reads = 0` on all 14 B runs; refault 0–571 (≤ 2000); PSI io 21–37 ms (≤ 70); pswpout 0; **`tier_waits` 3–4 on every B run (gate ≤ 2)**; **pswpin 0–1261, not 0**; app `pgpgin` = refills ± 7 MiB | **fails** on `tier_waits` and `pswpin` |
| G3 | G 64 block means +2.1 % / +0.9 %; single pairs −1.2 / −5.5 % (fresh), −1.0 / +1.2 % (old) | pass on the block means; fresh G 1024's single pair −5.5 % |
| G4 | B's block means 47.60 / 47.82 > A's best run of the block (42.44 / 39.83); B ≥ A at G 512 and 1024 on both boots | pass |
| G5 | every Q run same (B2 fresh: the banner fragment, see above) | pass |
| G6 | `calls/token=1.00`, close clean, ceiling 3840 after all 30 runs; H vs H0 +0.7 / +0.4 tok/s, text identical | pass |
| G7 | B's `resident after` 0 MiB on all 14 (A's 0.7–3.2 GiB); last `tier:` line `experts=88 mib=466.5 direct=1`; pswpout 0 in every B window | pass |

Reference (#216, `R3CY10WM83Y`, Q28, the 2026-09-21 config): A G = 64
33.4–43.7 tok/s at 0.68–4.38 ms/miss, prefill 462–558 tok/s; #218's B
(fadvise) 41.3–43.3 tok/s at 0.86–1.13 ms/miss, prefill −7 / −10 %; hybrid
50.7–52.2 tok/s. Goal ≥ 50 decode, prefill ≥ 497 (contract §1.1). With a
different config of record (#222) or unit, read only this sitting's A.

## Text

Every Q run `same` (G5; B2 fresh after removing the banner fragment, see Results). For the record, A1's and B1's generated text (G =
64, fresh boot):

| variant | generated text (G = 64, fresh, run 1) |
|---|---|
| A (Q28 one PD, no tier) |  In the same style, add more detail about its history, its people, its weather and the seasons, and do not stop until you are told to. In the same style, add more detail about its people, its weather and the seasons, and do not stop until you are told to. In the same style, add |
| B (Q28 one PD, `NNTR_MOE_TIER=1`) |  In the same style, add more detail about its history, its people, its weather and the seasons, and do not stop until you are told to. In the same style, add more detail about its people, its weather and the seasons, and do not stop until you are told to. In the same style, add |

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

* Unit `R3CY205ZMND` (S25 Ultra SM-S938N, SM8750, v79) on the ADF farm
  through an SSH `adb` shim; the runner's `adb reboot` was answered by
  the user rebooting by hand (the shim waits for the uptime to drop). Battery
  not logged by this runner. Runs 18:28–18:31 (fresh) and 18:58–19:01 (old)
  KST 2026-10-06, after #225's sitting on the same unit the same afternoon.
  No STOP; ceiling 3840 after every run.
* **Config:** the config of record after #225 PR 2,
  `docs/measurements/config/q40-qs4cx-wh.nntr_config.json` (md5
  `3f6808e3…`, `fc_wh_file_name` the FC WH sidecar `71812a91…`),
  `CONFIG OK 3848ab71…` (without `num_to_generate`) on both boots. Not
  the 2026-09-21 one: on #225's sitting Q28 loads on the config of record.
* **Binaries (deviation):** the code under test is #224 (merged into
  `htp_first_version`) **plus #225 PR 2** (#233, open), which the config
  of record needs for Q28 to load. Built from a **local merge**
  `8629b9392` = `htp/225-fcwh-e2e` @ `f08cccfeb` + `htp_first_version` @
  `9f33d7d43` (conflict only in `test/htp/host/run_inproc_e2e.sh`, both
  sides kept; not pushed). Staged `md5.txt` (device `MD5 OK` on both
  boots): `libnntrainer.so 48545aa4…`, `libcausallm_core.so 7f2dbb0d…`,
  `nntrainer_causallm 513f138c…`, `libccapi-nntrainer.so 618c6bc2…`,
  skel v79 `44339c7a…` (`UNDEFINED SYMBOLS OK (62)`, `ARCH OK (V79)`),
  `unittest_hvx_two_sessions f2e45bfb…`, `page_cache_evict 42595651…`,
  `libc++_shared.so b1586b9b…`, `libsdkl.so 0ad4e22a…`, `prompt512.txt
  fc65c158…`. Not the Artifacts table's set.
* **Host gates on the merge:** `*qs4cx*` 2/2, `*Lfm2Moe*` 7/7 (none
  skipped), `run_host_checks.sh` ALL CHECKS PASS + WORKER POOL LANES OK,
  syntax check 0; `run_inproc_e2e.sh` prints the four tier lines
  `bit_identical=1` (`tier_hits` 5 / 56 / 56 / 15, `tier_reads=0`,
  `direct=1`) and #225 PR 2's `fcwh … ok`, but ends `INPROC E2E FAIL` on
  `fwd-lfm25 min_snr_db=29.82 < floor 30` with `golden … bit_identical=0`:
  the fixture weights were regenerated on this workstation (the lfm25
  generator also rewrote `reference_logits.json`), and the unmerged
  `htp/225-fcwh-e2e` with the same fixtures fails identically (29.82,
  138.05 dB), so it is the fixtures, not the merge.
* `tier:` lines of fresh B1 (one B run; the same on every B run within
  a few ms):
  ```
  [HTP] tier: experts=24 mib=127.2 read_ms=58.5 drop_ms=451.6 direct=1
  [HTP] tier: experts=56 mib=296.8 read_ms=74.1 drop_ms=0.0 direct=1
  [HTP] tier: experts=88 mib=466.5 read_ms=78.5 drop_ms=0.0 direct=1
  ```
  The first call's whole-file `DONTNEED` costs 0.45 s at load.
* B2 (`NNTR_MOE_TIER=2`, one-thread copy) is faster per miss than B
  (8-slice) on both boots: 0.45 / 0.44 vs 0.67 / 0.51 ms, decode 48.37 /
  48.45 vs B's block means 47.60 / 47.82.

## Follow-up: `NNTR_MOE_TIER=2` four times (user, 2026-10-06)

B2 ran once a boot above; the user asked for four. `219-b2x4-run.sh` is
`219-arm-tier-run.sh` with the run list replaced by A / B / C interleaved
four times at G = 64 (C = `NNTR_MOE_TIER=2`, the 4th of each profiled),
everything else unchanged. Same unit, set, config and `MD5 OK` / `CONFIG
OK 3848ab71…`; a fresh boot (the user's reboot seen 19:09:18, first run at
uptime 61 s, last at 177 s). No STOP, ceiling 3840 after all 12, every
text `same`.

| run | cell | up s | prefill tok/s | decode tok/s | ms/miss | arm_ms/round | miss_wait us/tok | misses | tier hits / waits / reads | refill ms | drop ms | app pgpgin MiB | misses x 5.29 | win pgpgin MiB | win refault | win PSI io ms | win kswapd scan/s | win pswpin / pswpout | resident MiB in win (min-max) | resident after | text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | A G=64 prof=0 | 61 | 763.04 | 37.7804 | 3.98* | 6.036 | 6049.6 | 103 | 0 / 0 / 0 | 0.0 | - | 553.2 | 545 | 537 | 138475 | 169 | 139273 | 39 / 0 | 1144-1144 | 1160 | same |
| B1 | B G=64 prof=0 | 73 | 810.127 | 47.3373 | 0.73* | 1.101 | 770.4 | 103 | 103 / 4 / 0 | 255.2 | 0.0 | 546.0 | 545 | 541 | 7 | 26 | 0 | 2 / 0 | 0-0 | 0 | same |
| C1 | B2 G=64 prof=0 | 85 | 791.345 | 48.4115 | 0.45* | 0.677 | 321.8 | 103 | 103 / 4 / 0 | 245.5 | 0.0 | 546.0 | 545 | 530 | 81 | 28 | 0 | 5 / 0 | 0-0 | 0 | same |
| A2 | A G=64 prof=0 | 97 | 779.3 | 29.6434 | 8.11* | 12.284 | 12669.0 | 103 | 0 / 0 / 0 | 0.0 | - | 607.2 | 545 | 602 | 140141 | 176 | 33371 | 14058 / 129790 | 2736-2736 | 2803 | same |
| B2 | B G=64 prof=0 | 106 | 784.074 | 46.6813 | 0.80* | 1.218 | 900.3 | 103 | 103 / 4 / 0 | 261.6 | 0.0 | 558.1 | 545 | 553 | 2294 | 36 | 0 | 1169 / 0 | 0-0 | 0 | same |
| C2 | B2 G=64 prof=0 | 115 | 784.074 | 48.1928 | 0.44* | 0.666 | 310.6 | 103 | 103 / 3 / 0 | 237.6 | 0.0 | 546.0 | 545 | 535 | 11 | 22 | 0 | 1 / 12 | 0-0 | 0 | same |
| A3 | A G=64 prof=0 | 124 | 776.935 | 42.6099 | 2.06* | 3.125 | 2946.1 | 103 | 0 / 0 / 0 | 0.0 | - | 247.3 | 545 | 247 | 63148 | 75 | 41057 | 43 / 0 | 3136-3136 | 3106 | same |
| B3 | B G=64 prof=0 | 133 | 797.508 | 48.3019 | 0.52* | 0.795 | 451.5 | 103 | 103 / 4 / 0 | 243.7 | 0.0 | 601.9 | 545 | 591 | 6236 | 31 | 0 | 1467 / 0 | 0-0 | 0 | same |
| C3 | B2 G=64 prof=0 | 141 | 802.508 | 48.7433 | 0.44* | 0.659 | 307.4 | 103 | 103 / 3 / 0 | 235.2 | 0.0 | 546.9 | 545 | 536 | 57 | 26 | 0 | 168 / 0 | - | 0 | same |
| A4 | A G=64 prof=2 | 151 | 756.278 | 38.5775 | 3.69 | 5.587 | 5574.9 | 103 | 0 / 0 / 0 | 0.0 | - | 498.8 | 545 | 494 | 126401 | 151 | 82690 | 3 / 0 | 3051-3051 | 3045 | same |
| B4 | B G=64 prof=2 | 160 | 706.207 | 46.9208 | 0.83 | 1.256 | 934.4 | 103 | 103 / 4 / 0 | 247.4 | 0.0 | 548.3 | 545 | 554 | 0 | 21 | 0 | 2105 / 3621 | 0-0 | 0 | same |
| C4 | B2 G=64 prof=2 | 168 | 750.733 | 48.5216 | 0.43 | 0.655 | 302.4 | 103 | 103 / 3 / 0 | 236.3 | 0.0 | 546.0 | 545 | 546 | 0 | 19 | 0 | 3 / 0 | 0-0 | 0 | same |

| case (G = 64, 4 runs) | decode tok/s mean (runs) | sd | prefill tok/s mean | ms/miss (profiled; others `*`) | miss_wait us/tok | tier_waits | win refault | win pswpin / pswpout |
|---|---|---|---|---|---|---|---|---|
| A, no tier | 37.15 (37.78, 29.64, 42.61, 38.58) | 5.43 | 768.9 | 3.69 (2.06–8.11) | 2946–12669 | – | 63k–140k | up to 14058 / 129790 (A2) |
| B, `=1` (8-slice copy) | 47.31 (47.34, 46.68, 48.30, 46.92) | 0.71 | 774.5 | 0.83 (0.52–0.80) | 452–934 | 4, 4, 4, 4 | 0–6236 (B2 2294, B3 6236 > 2000) | up to 2105 / 3621 (B4) |
| C, `=2` (one-thread copy) | **48.47** (48.41, 48.19, 48.74, 48.52) | **0.23** | 782.2 | **0.43** (0.44–0.45) | **302–322** | 4, 3, 3, 3 | 0–81 | up to 168 / 12 (C2) |

Reading: on four runs `=2` is faster than `=1` by 1.16 tok/s (+2.5 %),
steadier (sd 0.23 vs 0.71), and every C run is under G1's 0.5 ms/miss
(0.43 profiled; `=1` 0.83 profiled this boot). `=1` also went over G2's
refault bound on two runs and swapped out on one; `=2` did not. `=2`
still has `tier_waits` 3–4 (G2's ≤ 2) and a non-zero pswpin. Prefill:
`=2` +1.7 % on A's block mean, `=1` +0.7 %.

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
