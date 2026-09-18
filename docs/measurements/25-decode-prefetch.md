# Measurement 25 (H1): cross-op weight prefetch A/B, the W8 weight-stream ceiling, and the 64-row chunk

Branch `hvx/25-decode-prefetch` @ `7d318718` (the last commit that changes DSP sources; later commits on
the branch are tools/docs only) — estimated device time: **30 min** (8 runs at 512 ≈ 10 min, four
1024 / 4096 runs ≈ 10 min, the 4096 image push ≈ 3 min if absent, md5 checks; the H2 kernel sweep is a
separate, later handoff).

## Why
Decode is the 596 MB weight stream (#24 closed the host path: the logits return is < 0.1 ms inside a
34 ms step). Today every tiled matmul kicks its own first DMA chunk only after the op starts — with 6
workers the decode W8A8 ops are 1–2 chunks per worker, so the first chunk is fully exposed and the DDR
idles through ROPE / ATTN / ADD / RMSNORM / SILU_MUL. This branch keeps one DMA queue per worker for the
session and lets each matmul kick the *next* matmul's chunk 0 into its idle half-slab (plan
`docs/plans/25-decode-prefetch.md`, ledger ①). The simulator does not charge DDR, so only silicon can
say what that buys; the same run measures the **weight-stream ceiling** (a skel that streams every
weight byte but multiplies nothing) that replaces the provisional `≥ 60 tok/s` goal and gates #51, and
the 64-row chunk of ledger ⑨. Decisions that hang on it: the prefetch default (on / compiled-in-but-off),
the chunk default, the stage-1 W8A8 goal cell, and whether H2 (`MM_TB` / `MM16_R`) runs on B or D.

## Artifacts (built in the container, SDK 6.4.0.2 / hexagon-clang 19.0.04 `toolv19`, `HEX_ARCH=v79`, @ `7d318718`)

| file | md5 (size) | built with (`HEX_EXTRA_CFLAGS=`) | answers |
|---|---|---|---|
| build_hexagon/skel/libnntr_htp_skel.A.so | `e25597fe0dc36adfcb0247c63f384b7c` (51,112 B) | `-DHTP_PROF_FARF -DHTP_MM_NO_PREFETCH` | **A control**: environment check, the A/B baseline, the device non-matmul share |
| build_hexagon/skel/libnntr_htp_skel.B.so | `208b25a23c4b308d5d8eb0eec8a4d103` (55,208 B) | `-DHTP_PROF_FARF` | **B prefetch**: G1 |
| build_hexagon/skel/libnntr_htp_skel.C.so | `853af711a5772852b440ef45a22cfaf3` (51,112 B) | `-DHTP_PROF_FARF -DHTP_MM_STREAM_ONLY` | **C ceiling**: G1' (GB/s, ceiling tok/s at 512 / 1024 / 4096). Outputs are garbage by construction |
| build_hexagon/skel/libnntr_htp_skel.D.so | `2da2a4f9217ae904bf11c9bd99825f9b` (55,208 B) | `-DHTP_PROF_FARF -DHTP_MM_CHUNK_ROWS=64u` | **D chunk**: ledger ⑨ on top of B |
| build_hexagon/skel/libnntr_htp_skel.so (default, not run) | `7dc42f5981cf000393d2d23c3c181314` (55,208 B) | (none) | the shipping build = B without the FARF line |
| build_hexagon/host/hexagon_e2e_test | `88996fee9a57113cf47793d877e4ae19` | `./tools/hexagon/build_host_test.sh` (host sources unchanged since #24) | |
| build_hexagon/host/hexagon_rpc_test | `dd05aaab4cbe916c2efabfb9b6757678` | same | |
| /tmp/qwen3_full.hexw / .hexcfg | `5abf61bef086368559423daa3bea9a99` / `5926be5703f85531c9c46c1978c25287` | packer unchanged since P4 (#23 / #35 tables) | |
| /tmp/qwen3_full4k.hexw / .hexcfg | `e5c92fad696fb35405918941d1062be3` / `b0d8aca9a6c7626b0af8dc81e1de59eb` | `--max-seq 4224` | |
| /tmp/t512_23.i32 | `10bb428f92c792aa7733339f8e6b0da1` | `make_tokens.py --limit 512` on `eval_23.txt` (#35 §1) | accuracy anchor: #35 B1 on this unit **41.4947 / 162** |
| /tmp/t1024.i32, /tmp/t4096.i32 | `86e44c0633ebab556aafe3aa68d80c63` / `29684fbd372495912951cb3cc0d25e20` | #24 table | goal rows |

All four skels carry `-DHTP_PROF_FARF` so its cost (one FARF per call) cancels in every ratio. **If the
Mac binaries did not reach the workstation, rebuild them there** (SDK 6.4.0.1 gives the same `toolv19`;
#35 measured a uniform +192 B and a different md5 per skel from the runtime/header delta — write the
md5 you get, keep the size):

```bash
git fetch && git checkout hvx/25-decode-prefetch      # do not rebase; commit the result on this branch
source <SDK>/setup_sdk_env.source; export ANDROID_NDK=<ndk path>
for v in "A:-DHTP_PROF_FARF -DHTP_MM_NO_PREFETCH" "B:-DHTP_PROF_FARF" "C:-DHTP_PROF_FARF -DHTP_MM_STREAM_ONLY" "D:-DHTP_PROF_FARF -DHTP_MM_CHUNK_ROWS=64u"; do
  HEX_EXTRA_CFLAGS="${v#*:}" ./tools/hexagon/build_skel.sh && cp build_hexagon/skel/libnntr_htp_skel.so build_hexagon/skel/libnntr_htp_skel.${v%%:*}.so
done
./tools/hexagon/build_host_test.sh
md5sum build_hexagon/skel/libnntr_htp_skel.{A,B,C,D}.so build_hexagon/host/hexagon_e2e_test
```

## Steps (workstation, S25 Ultra on USB; ≈ 30 min)

Name the unit: `adb devices` → paste the serial in the Results header. Pass rules are DSP Mcyc ratios
within this session, so the unit only has to be the same for all rows (HEXAGON.md §7 rule 9). Insert
the serial after the image path only if more than one device is attached.

```bash
adb devices
run() { cp build_hexagon/skel/libnntr_htp_skel.$1.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so; shift; ./tools/hexagon/run_e2e_test.sh "$@"; }
# 0. transport sanity on A (RPC_TEST PASS), also warms the DSP
cp build_hexagon/skel/libnntr_htp_skel.A.so build_hexagon/skel/libnntr_htp_skel.so && ./tools/hexagon/run_device_test.sh
# 1. pass 1 at 512: A, B, C, D — speed, then --eval (C has no --eval: garbage outputs by construction)
for v in A B C D; do run $v /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --chunk 128 --steps 64; [ $v != C ] && run $v /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --eval; done
# 2. pass 2, reversed: D, C, B, A — the pass the tables read (rule 9(d): pass 1's first run is cold)
for v in D C B A; do run $v /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --chunk 128 --steps 64; [ $v != C ] && run $v /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --eval; done
# 3. goal rows: B at 1024 / 4096 (speed + --eval), C at 1024 / 4096 (speed)
run B /tmp/qwen3_full   -- --tokens /tmp/t1024.i32 --chunk 128 --steps 64
run B /tmp/qwen3_full   -- --tokens /tmp/t1024.i32 --eval
run B /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64
run B /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --eval
run C /tmp/qwen3_full   -- --tokens /tmp/t1024.i32 --chunk 128 --steps 64
run C /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64
adb shell md5sum /data/local/tmp/nntr_htp/libnntr_htp_skel.so /data/local/tmp/nntr_htp/hexagon_e2e_test   # the C skel, last pushed
# 4. the per-kind DSP split of every run (FARF lines are in the device_farf log of each run)
python3 tools/hexagon/summ_farf_prof.py logs/hexagon/device_farf_*.log
```

Expected lines per run: `E2E init ok weights=598623744 kv=234881024 act=3932160 n_ops=451` (the 4096
image: `weights=599180800`), 4 × `E2E step … n=128 …`, 63 × `E2E step … n=1 pcycles=<c> us=<t> top1=<id>`,
`E2E gen <64 ids>`, `E2E decode steps=63 median_us=<u> median_pcycles=<c> pcycles_per_us=<r>`, `E2E wall_ms`;
`--eval`: `E2E ppl <x> steps 511 top1 <n> wall_ms <w>` then `E2E decode steps=511 …`. In
`logs/hexagon/device_farf_<stamp>.log` (every variant): one `nntr_htp: prof n=<n> pos=<p> ops=451 kcyc=<k>
mm8=<k> mm16=<k> lg=<k> attn=<k> rest=<k>` per call (kilo-pcycles); for **C** additionally
`nntr_htp: stream bytes/step=595984384 (STREAM_ONLY, outputs are garbage)` after `init ok`. C's `top1`
ids and `E2E gen` are meaningless and may repeat one id — that is not a failure. A hang or DSP restart:
note the last `E2E step` line and the FARF tail in the Notes, `adb reboot`, continue with the next command
(risk 3 of the plan: a descriptor pending across the worker barrier is the one new runtime pattern; a stall
shows as a > 2× step in the `pcycles` column of the first prefetched op, i.e. right after step 0 of a
B / D run).

Reading the numbers (HEXAGON.md §8.2, #24): prefill tok/s = 512 ÷ (sum of `us` of the 4 `n=128` steps
/ 1e6); decode median host ms = `median_us` / 1000; decode tok/s = 1000 ÷ that; **DSP Mcyc/step =
`median_pcycles` / 1e6** (the column every rule reads); `pcycles_per_us` = the run's own clock. For C:
GB/s = 595,984,384 ÷ (`median_pcycles` ÷ `pcycles_per_us`) ÷ 1000; ceiling tok/s = 1000 ÷ (`median_us` / 1000).

## Results (fill in) — unit serial: `__________`, pass 2 rows are the record

Reference cells for this unit (`R3CY10WM83Y`, v79 shipping skel, #35 B1 = `hvx_impl` today): 512 →
198.7 prefill tok/s / 32.391 ms / 30.87 tok/s / **61.200 Mcyc** at 1905–1915 `pcycles_per_us`; 1024 →
131.1 / 40.227 / 24.86 / **77.781**; 4096 → 32.71 / 89.103 / 11.22 / **186.753** (cooled re-run; the 4096 cell
compares only within this session, rule 9(e)). On `R3CY205ZMND` the v79 references are 60.1 / 76.5 /
177.6 Mcyc. **Environment check:** A (pass 2) must land within ±5 % of the unit's 512 Mcyc reference
(61.2 or 60.1); outside that band the session is void (stale artifact or wrong unit), not a result.

### Speed, 512 tokens (`--chunk 128 --steps 64`, `t512_23.i32`, `qwen3_full`)

| variant | pass | skel md5 (from `run()`) | prefill tok/s | decode median host ms | decode tok/s | **DSP Mcyc/step** | `pcycles_per_us` | FARF median kcyc: mm8 / mm16 / lg / attn / rest | log stamp |
|---|---|---|---|---|---|---|---|---|---|
| A control | 1 | | | | | | | | |
| B prefetch | 1 | | | | | | | | |
| C ceiling | 1 | | | | | | | | |
| D chunk 64 | 1 | | | | | | | | |
| **D chunk 64** | 2 | | | | | | | | |
| **C ceiling** | 2 | | | | | | | | |
| **B prefetch** | 2 | | | | | | | | |
| **A control** | 2 | | | | | | | | |

### Goal rows (B and C at 1024 / 4096; references are #35 B1 on `R3CY10WM83Y`)

| variant | ctx | image | prefill tok/s (ref) | decode median host ms (ref) | decode tok/s (ref) | DSP Mcyc/step (ref) | `pcycles_per_us` | C only: GB/s, ceiling tok/s |
|---|---|---|---|---|---|---|---|---|
| B prefetch | 1024 | qwen3_full | (131.1) | (40.227) | (24.86) | (77.781) | | — |
| B prefetch | 4096 | qwen3_full4k | (32.71) | (89.103) | (11.22) | (186.753) | | — |
| C ceiling | 512 | qwen3_full | | | | | | |
| C ceiling | 1024 | qwen3_full | | | | | | |
| C ceiling | 4096 | qwen3_full4k | | | | | | |

### Accuracy (`--eval`, both passes; C is `n/a (stream only)`)

| variant | pass | ctx | token file | `--eval` PPL (ref) | top-1 (ref) | `E2E gen` identical to A pass 2? (`diff <(grep -o 'top1=[0-9]*' a.log) <(… b.log)`) |
|---|---|---|---|---|---|---|
| A control | 1 | 512 | t512_23.i32 | (41.4947) | (162) | |
| B prefetch | 1 | 512 | t512_23.i32 | (41.4947) | (162) | |
| D chunk 64 | 1 | 512 | t512_23.i32 | (41.4947) | (162) | |
| D chunk 64 | 2 | 512 | t512_23.i32 | (41.4947) | (162) | |
| B prefetch | 2 | 512 | t512_23.i32 | (41.4947) | (162) | |
| A control | 2 | 512 | t512_23.i32 | (41.4947) | (162) | — (the reference) |
| B prefetch | — | 1024 | t1024.i32 | (5.8595) | (692) | vs #35 B1 log |
| B prefetch | — | 4096 | t4096.i32 | (1.5623) | (3759) | vs #35 B1 log |
| C ceiling | — | — | — | n/a (stream only) | n/a | n/a |

`adb shell md5sum` after the last run: `__________` (the C skel) / `__________` (hexagon_e2e_test).

### Pass rules (plan §1; fill the verdict column)

| rule | reads | verdict |
|---|---|---|
| env | A pass 2 within ±5 % of 61.2 Mcyc (`R3CY10WM83Y`) or 60.1 (`R3CY205ZMND`) at 512 | |
| G1 | `Mcyc(B) ≤ 0.90 × Mcyc(A)` at 512, pass 2 → prefetch **on** by default; 0.90–0.97 partial (adopt, record); > 0.97 no lever (keep the graph-lifetime queue, prefetch compiled in but off) | |
| G1' | C: GB/s and ceiling tok/s at 512 / 1024 / 4096 → the stage-1 W8A8 goal cell in HEXAGON_BENCHMARK.md and the #51 input | |
| D rule | D ≥ 2 % below B at 512 decode (Mcyc, pass 2) **and** D prefill tok/s ≥ B → `HTP_MM_CHUNK_ROWS 64` becomes the default; else keep max-fit | |
| G3 | A, B, D: PPL / top-1 equal to the digit across variants and passes, and `E2E gen` byte-identical to A pass 2 | |
| no hang / SSR / FARF fatal | all 14 runs | |

## Notes from the run
- phone model / serial / SDK version on the workstation:
- rebuilt on the workstation? (then the md5s in the artifact table do not apply; the sizes should match up to the +192 B of #35)
- warm-up: does pass 1 A differ from pass 2 A by the usual 3–5 %?
- anything odd (thermal, FARF errors, stale-file pushes, a > 2× step right after the prefill chunks):
- `summ_farf_prof.py` output for A pass 2 and C pass 2 (the device non-matmul share and the matmul-only share of C):

Hand back: `git add docs/measurements/25-decode-prefetch.md && git commit -s -m "[docs] Fill the #25 H1 measurement (<unit>, SDK <ver>)" && git push`,
then `gh issue edit 25 -R dlwlzzero/nntrainer --remove-label state:needs-measurement --add-label state:measured`.
