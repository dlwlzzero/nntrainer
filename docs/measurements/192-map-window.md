# Measurement 192: the single-session map window (cost of mapping the next layer's FC weights per token)

Branch `htp/192-map-window` @ `72086ab3` (stacked on `htp/178-probe` @
`bf78a8f9`; issue #192 is the spec, probe only, no app change) —
estimated device time: **20 min**

## Why

#178 keeps the FC set + lm_head (383 MiB) on the NPU with a second cDSP
session because the loaded app's PD has ≈ 113 MiB left beside its
3696 MiB MoE arena. The single-session alternative maps layer L+1's FC
weights (≈ 11 MiB) into a small rotating window while layer L runs:
22 map/unmap pairs per token. Stop rule (issue): > 1.0 ms/token projected
= not viable; ≤ 0.3 ms = viable, plan the window design.

## What the SDK offers (6.4.0.1, `incs/remote.h`, `incs/HAP_mem.h`, `docs/software/ipc/rpc.html`)

* ARM: `fastrpc_mmap` / `fastrpc_munmap` with `enum fastrpc_map_flags`:
  `FASTRPC_MAP_STATIC` (0, not tagged with the fd: the DSP gets no VA via
  `HAP_mmap_get`), `FASTRPC_MAP_FD` (2, the app's arena flag),
  `FASTRPC_MAP_FD_DELAYED` (3, the DSP maps with `HAP_mmap`),
  `FASTRPC_MAP_FD_NOMAP` (16, DELAYED without the CPU mapping),
  `FASTRPC_MAP_FD_EXTENDED` / `FASTRPC_MAP_FD_DELAYED_EXTENDED` (17 / 18,
  "the extended address space on DSP"). `fastrpc_mem_request`
  (`FASTRPC_MEM_MAP` / `UNMAP`) adds `FASTRPC_MAP_ATTR_RETAIN_IOVA` (keeps
  the SMMU IOVA reserved between maps of the same buffer); resolved with
  `dlsym`, the probe prints `mem_request=missing` if the device runtime
  lacks it. (`remote_mem_map` is the older STATIC-only API; not probed.)
* DSP: `HAP_mmap` / `HAP_munmap` (and the `2` variants), `HAP_mmap_get` /
  `HAP_mmap_put`, `HAP_mem_request` (`HAP_MEM_MAP` / `UNMAP` /
  `HAP_RESERVE_VA`, not probed), `HAP_mem_get_stats`. rpc.html lists "Map
  HLOS memory allocated by the corresponding HLOS application" among the
  unsigned-PD services, so `HAP_mmap` should be available here; the probe
  answers it (`way=fd_delayed`, `way=dsp_only_hap_mmap`).
  `HAP_MEM_FLAGS_EXTENDED_MAP` is documented "Unsupported currently".

## Probe (`unittest_hvx_two_sessions --gtest_filter='MapWindow.*'`)

One session opened as the app opens it, the app's arena held as a ladder
(14 × 256 + 112 = 3696 MiB, attached) for the whole suite.

| test | cell | what |
|---|---|---|
| W1 | (a)+(b) | per way (the six flags, mem_request + RETAIN_IOVA, DSP-only `HAP_mmap`) × 1/4/11/12/24 MiB uncached ION: 1 cold + 50 repeated map → DSP step → unmap; ARM µs (median/p90), DSP µs, first-touch µs (one word per 4 KiB) |
| W2 | (c) + headroom | 24 MiB window mapped + attached once; 20× memcpy 11 MiB into one half, `q4m1_release` + `q4m1_register_arena` (DSP cache invalidate) → FC bit-exact; uncached and cached ION; `HAP_mem_get_stats` before/with the window; VA left (8 MiB mappings, cap 512 MiB) |
| W3 | (d) | exact FC hvx_intrin, L2 feed, 3 lanes, K=7168 N=2048: DSP heap vs long-mapped vs just-mapped (first and second call, 10 cycles) |
| W4 | projection | per way: 22 × (pair + fresh-FC penalty) ms/token, the re-attach copy alternative, best way and the stop rule; lm_head: one real 140 MiB FD map beside the arena + the pair fit and 12 × 12 MiB slices; last: the heap beside a mapped 24 MiB window, capped at 64 MiB |

Leak rules: every mapping is removed before close (a failed munmap prints
`W_STOP rule=leak` and ends the suite); no heap is grown to the end.

## Artifacts (built on the workstation, SDK 6.4.0.1, HexKL 6.4.0.1, NDK r30, v79)

Staged in `/local/mnt/workspace/htp_moe/192/set/` with `md5.txt`, all
from `72086ab3` (the IDL grew by two entries: skel, stub and app from one
tree, rule 3).

| file | md5 | built with |
|---|---|---|
| `libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| `libcausallm_core.so` | `fa7d08abc2860d5ead1ca65d3ae3575d` | `build_android.sh --htp` (2 × `NNTR_HTP_FORWARD_KINDS`, rule 36) |
| `libccapi-nntrainer.so` | `dec699b3a6114549d9804986808b869f` | `jni/obj/local/arm64-v8a/` |
| `libnntr_hvx_skel.so` | `d97793d14c37d446d159b53d8d68bf7d` | `test/htp/build.sh` (v79; not byte-reproducible, the staged md5 is the one) |
| `libnntrainer.so` | `e621bc29fd8b331777605c14cdcde861` | `jni/obj/local/arm64-v8a/`; NEEDED `libsdkl.so`, `libcdsprpc.so` |
| `libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL beta.2 `lib/6.4.0.1/armv8_android26` |
| `nntrainer_causallm` | `255e43c87dd1c916434bb5519deb2d27` | `build_android.sh --htp` (A, nothing set) |
| `prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | the 512-token prompt (as #178) |
| `unittest_hvx_softmax` | `1095280455c70a82dcbf64104e6796a5` | `ndk-build unittest_hvx_softmax` (the stale-skel canary) |
| `unittest_hvx_two_sessions` | `f93d7e0d69c8f6d6da116cace5062e1d` | `ndk-build unittest_hvx_two_sessions` (the probe, `MapWindow.*`) |

## Steps

Reboot the phone first, then
`bash /local/mnt/workspace/htp_moe/192/run_192.sh [serial]` (copy in
`docs/measurements/192-run.sh`; default serial R3CY10WM83Y). It checks the
md5s, installs to `/data/local/tmp/nntrainer/causallm/s192`, runs the
stale-skel canary (`HvxFcQ4.MatchesSpecBitExact`, 10 hvx_intrin cells
bad=0), A at G = 8 (prompt 512, nothing set; banner `0x703e1`), the probe,
A at G = 8 again (the app must still map its arena and generate: a failure
names a leak), zone0 ≤ 35 °C before each block. Logs in
`/local/mnt/workspace/htp_moe/192/logs/`, summary `logs/sitting.out`.

Expected lines: `W_FIELD ladder_mib=3696`, four `MapWindow.W[1-4]` results,
40 `W_FIELD map way=…` rows, two `W_FIELD reattach ion=…` rows with
`bad=0`, one `W_FIELD fc K=7168 …` row with `bad=0`, `W_FIELD project …`
rows, one of `W_STOP rule=not_viable` / `W_VERDICT viable` /
`W_VERDICT between`, `W_FIELD lm_head_140 …`, `W_FIELD heap_with_window_mib=…`,
no `W_STOP rule=leak`.

## Results (fill in)

| cell | value |
|---|---|
| sanity_0 / sanity_1 prefill, decode tok/s | |
| mem_request | |
| best way, pair µs at 11 MiB, ms/token (22 pairs) | |
| fd: arm map / unmap µs, dsp get µs, touch µs (11 MiB) | |
| fd_delayed / dsp_only: HAP_mmap rc, µs | |
| re-attach: copy 11 MiB µs (GB/s), re-point µs, bad (uncached / cached) | |
| FC µs: heap / long-mapped / fresh first / fresh second | |
| heap stats with window; VA headroom; heap_with_window_mib | |
| lm_head 140: map rc, µs; fit | |
| verdict (stop rule) | |

Reference: #178 set3 — S1 ladder 3840 MiB (256 steps), loaded app heap
≈ 113 MiB (#132 Part A G4), exact FC hvx_intrin L2 feed on S2 (#178 Q4).

## Notes from the run

<serial, uptime (reboot first), battery, zone0, FARF/AEE lines>
