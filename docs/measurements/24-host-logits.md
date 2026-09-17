# Measurement 24: what the 607,744 B logits return costs, and does rpcmem remove it

Branch `hvx/24-host-logits` @ `83039562` (source; this file's own commit sits on top and changes
no source) — estimated device time: **25 min**

**Deviation from the handoff template, stated up front.** The artifacts are *not* prebuilt: this
branch was implemented without SDK, container or device, so the user builds the two host
harnesses on the Linux workstation first (native SDK 6.4.0.1 + NDK r26d, no `tools/docker/run.sh`)
and fills the md5 column from their own `md5sum`. The v75 skel is **reused** from the #33 gate-4
build already in `build_hexagon/skel/libnntr_htp_skel.so` — this plan changes no file under
`nntrainer/tensor/hexagon/htp/`, so no `build_skel.sh` run is needed. By the user's decision of
2026-09-17 the device measurement comes first and the simulator gate (bit-identical STATs) is
deferred to a later log push, listed last below.

## Why
`forward()` returns 151,936 fp32 logits per step as a `rout sequence<float>`; until #24 both the
app and the harness handed a `malloc` buffer, which the FastRPC library serves through a staging
copy, while a pointer inside an `rpcmem_alloc` buffer is passed as its dma-buf fd. The #23 logs,
re-read the way the harness measures, already put the host-vs-DSP gap at 0.5–3 ms per decode step
(plan §1), so this run answers (i) what the return costs in isolation per host memory kind, (ii)
what survives in the real step, and (iii) whether the generated ids and `--eval` PPL are
byte-identical across the paths. The decision that hangs on it: ship `rpcmem` or `static` as the
harness default, and whether a DSP top-k method (ABI v5, plan §3 contingency (a)) is ever needed
(only if the real-step host-vs-DSP gap with `static` is still ≥ 2 ms).

## Artifacts (Linux workstation, Hexagon SDK 6.4.0.1, hexagon-clang 19.0.04, NDK r26d, HEX_ARCH=v75)
| file | md5 (from your `md5sum`) | built with |
|---|---|---|
| build_hexagon/skel/libnntr_htp_skel.so (v75, reused from #33 gate 4) | `<md5sum>` (46,824 B was #23's size) | `HEX_ARCH=v75 ./tools/hexagon/build_skel.sh` @ the #33 branch; DSP sources identical to this branch |
| build_hexagon/host/hexagon_rpc_test | `<md5sum>` | `./tools/hexagon/build_host_test.sh` @ 83039562 (step 1) |
| build_hexagon/host/hexagon_e2e_test | `<md5sum>` (must differ from #23's `abb4fb72…`) | same |
| /tmp/qwen3_full.hexw / .hexcfg | `5abf61bef086368559423daa3bea9a99` / `5926be5703f85531c9c46c1978c25287` | the P4 image (#23 table) |
| /tmp/qwen3_full4k.hexw / .hexcfg | `e5c92fad696fb35405918941d1062be3` / `b0d8aca9a6c7626b0af8dc81e1de59eb` | the `--max-seq 4224` image (#23 table) |
| /tmp/t512.i32 | `9da5d7b4…` (#23: the P4 prompt, accuracy anchor 33.0884 / 189) | existing |
| /tmp/t1024.i32, /tmp/t4096.i32 | `<md5sum>` (not recorded in #23) | existing, P4 run |

## Steps (workstation, phone on USB, single device — no serial argument)
```
git fetch && git checkout hvx/24-host-logits
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/setup_sdk_env.source; export ANDROID_NDK=/opt/android-ndk-r26d
# 0. plan step 0: the flag names the static variant relies on (paste the output into "Notes")
grep -n "FASTRPC_MAP_STATIC\|FASTRPC_MAP_FD_DELAYED\|remote_register_buf" $HEXAGON_SDK_ROOT/incs/remote.h
grep -n "RPCMEM_DEFAULT_FLAGS\|RPCMEM_FLAG" $HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc/rpcmem.h
# 1. host harnesses only; the script prints "remote.h: FASTRPC_MAP_STATIC probe -> -DNNTR_HAVE_FASTRPC_MAP_STATIC=1"
./tools/hexagon/build_host_test.sh
md5sum build_hexagon/skel/libnntr_htp_skel.so build_hexagon/host/hexagon_rpc_test build_hexagon/host/hexagon_e2e_test /tmp/t1024.i32 /tmp/t4096.i32
# 2. transport in isolation (G1)
./tools/hexagon/run_device_test.sh            # prints "logs: logs/hexagon/device_test_<stamp>.log, ..."
python3 tools/hexagon/check_rpc_log.py logs/hexagon/device_test_<stamp>.log
# 3. the real step, three host memories, same image and prompt (G2/G3)
for m in malloc rpcmem static; do ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512.i32 --chunk 128 --steps 64 --logits-mem $m; done
# 4. accuracy, A vs B (G3; also the idle-gap clock)
for m in malloc rpcmem; do ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512.i32 --eval --logits-mem $m; done
# 5. goal rows with the winner of step 2 (rpcmem, or static if it beat rpcmem by >= 0.1 ms)
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t1024.i32 --chunk 128 --steps 64 --logits-mem <winner>
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64 --logits-mem <winner>
adb shell md5sum /data/local/tmp/nntr_htp/libnntr_htp_skel.so /data/local/tmp/nntr_htp/hexagon_e2e_test
```
Then paste the lines into the tables below, commit this file on the branch, push, and
`gh issue edit 24 --remove-label state:needs-measurement --add-label state:measured`.

Expected lines. Step 2: `RPC_TEST open ok`, `rpcmem ok`, `bad-version rejected ok`, `init ok`,
32 × `RPC_TEST forward_us <us>`, `RPC_TEST pattern ok`, 3 × 32 × `RPC_TEST forward_full_us mem={malloc,rpcmem,static} <us>`
(if the build printed the probe as absent, `mem=static` is skipped with one message — then the C
rows below are void and steps 3/5 run without `static`), `RPC_TEST full-logits pattern ok`,
`RPC_TEST PASS`; `check_rpc_log.py` prints `forward_full mem=<m> x32: … median <us> …, -forward_us(8) <delta>` per
memory and `VERDICT: PASS`. Steps 3–5: `E2E init ok weights=598623744 kv=234881024 act=3932160 n_ops=451`
(the 4096 image: `weights=599180800`), `E2E logits_mem <m>`, 4 × `E2E step … n=128 …`, 63 × `E2E step … n=1 pcycles=<c> us=<t> top1=<id>`,
`E2E gen <64 ids>`, `E2E decode steps=63 median_us=<u> median_pcycles=<c> pcycles_per_us=<r>`, `E2E wall_ms <w>`.
Step 4: `E2E ppl <x> steps 511 top1 <n> wall_ms <w>` then `E2E decode steps=511 …`. `--logits-mem static`
failing with `register_static failed (0x…)` is itself a result: note it and continue.

**Deferred (run before the PR merges, not needed for `state:measured`):** the simulator gate proving
no DSP byte moved — `HEX_ARCH=v75 ./tools/hexagon/build_sim_test.sh; HEX_ARCH=v75 ./tools/hexagon/run_sim_test.sh profile acc`
→ `SIM_TEST profile PASS`, `profile_prefill_acc STAT max_abs=0.077216 max_rel=98.2637`; then
`for t in smoke pool exp quant matmul matmul_dma rmsnorm rope eltwise embed attn logits graph; do HEX_ARCH=v75 ./tools/hexagon/run_sim_test.sh $t || break; done`
→ 13 × `SIM_TEST <t> PASS`, `graph_prefill 0.0245416/9.3795`, `graph_decode 0.0202219/23.975`, `quant_generic 0/65536`,
`quant16_generic 167/25600`. Any STAT movement means a DSP file changed by mistake (this branch touches none).

## Results (fill in)
Pass rules (plan §1): G1 — medians recorded, `VERDICT: PASS`; G2 — on the shipped variant
`median_us − median_pcycles/2090` ≤ 5 ms (expected ≈ 0.5 ms) and `pcycles_per_us` ≥ 2.05;
G3 — `E2E gen` byte-identical across A/B/C and to #23's A 512 run (`logs/hexagon/e2e_20260917_111035.log`,
the first A command of #23), `--eval` PPL/top-1 identical A vs B and to 33.0884 / 189;
G4 — tok/s within #23's noise (512 ±5 %, 1024/4096 ±3 %).

(i) transport, `hexagon_rpc_test` (n_ops == 0). The DSP fill of the 151,936 floats is a scalar
loop with a software modulo per element (`executor.c`, untouched by this plan) and the dummy path
reports no pcycles, so "− 8 floats" is an *upper bound* that includes that fill — read A − B and
B − C as the transport, and the plan's rule (2) from table (ii)'s gap, not from this column.
| host memory | median `forward_full_us` | min / max | − median `forward_us` (8 floats) | A − this row | reading |
|---|---|---|---|---|---|
| malloc (A) | | | | 0 | staging copy of 607,744 B |
| rpcmem (B) | | | | | fd path, mapped per call |
| rpcmem + FASTRPC_MAP_STATIC (C) | | | | | mapped once |
| 8-float `forward_us` median | | | 0 | — | reference (#23: ~250 µs class) |

(ii) real step, `--chunk 128 --steps 64`; #23 A (malloc) for reference: 512 → 31.3 ms / 64.3 Mcyc / 2.054 / 31.9 tok/s, 1024 → 43.2 / 84.2 / 1.949 / 23.2, 4096 → 104.1 / 213.0 / 2.046 / 9.6
| variant | ctx | image | skel md5 (adb read-back) | prefill tok/s | median_us | median_pcycles | pcycles_per_us | gap ms = us/1000 − Mcyc/2090 | decode tok/s = 1000/(us/1000) | `E2E gen` identical to #23 A? |
|---|---|---|---|---|---|---|---|---|---|---|
| A malloc | 512 | qwen3_full | | | | | | | | |
| B rpcmem | 512 | qwen3_full | | | | | | | | |
| C static | 512 | qwen3_full | | | | | | | | |
| winner = | 1024 | qwen3_full | | | | | | | | — (no #23 gen reference kept) |
| winner = | 4096 | qwen3_full4k | | | | | | | | — |

(iii) accuracy, `--eval` on the P4 prompt (reference #23 A: 33.0884 / 189; x86 33.0195 / 184)
| variant | ctx | --eval PPL | top-1 | median_us | median_pcycles | pcycles_per_us (idle-gap clock; #23-era estimate 1.15 G) | wall_ms |
|---|---|---|---|---|---|---|---|
| A malloc | 512 | | | | | | |
| B rpcmem | 512 | | | | | | |

Decision after filling (plan §3): (1) ship `rpcmem`, or `static` if it beats rpcmem by ≥ 0.1 ms in (i);
(2) if the C row's real-step gap in (ii) is still ≥ 2 ms (the plan wrote this rule against (i)'s
"− 8 floats" column, which the DSP fill confounds) → follow-on plan for the DSP top-k method;
(3) if the e2e gap exceeds 5 ms while (i) shows < 1 ms, the residual is not the logits → file
"DSP clock during host idle gaps" with (iii)'s `pcycles_per_us`; (4) if rpcmem == malloc in (i),
libadsprpc did not recognise the buffer → add `remote_register_buf_attr(data, size, fd, 0)` to
`RpcmemBuffer` and re-issue step 2 only.

## Notes from the run
- step 0 grep output:
- build_host_test.sh probe line:
- anything odd (thermal, warm-up, FARF errors, stale-file pushes, a `static` failure code):
