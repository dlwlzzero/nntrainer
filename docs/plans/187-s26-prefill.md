# 187 — S26 prefill above the S25: HMX clock vote, CPU prefill loops, noise-proof protocol

Issue: dlwlzzero/nntrainer#187 (p0). Read against `htp/177-m1-dma-queues` @
`926819a2`. `htp/185-m1-schedule-overlap` exists locally at the same commit
and has no commits of its own yet. Work branch **`htp/187-s26-prefill`**. Branch
it from `htp/185-m1-schedule-overlap`'s head if #185 has commits when work
starts, otherwise from `htp/177-m1-dma-queues`. PR into **`htp_moe_v81`**
(stacked on #185's or #181's PR while those are open). Device:
`adb -s R5KL20NFRCK` only (contract §12, 2026-09-29). The S25 units are off
limits. The shared checkout `/home/j2z0-lee/nntrainer` and its `build/` and
`builddir/` belong to the #185 implementer, so all work happens in a
`git worktree` under `/local/mnt/workspace/htp_moe/187/`, with that
worktree's own build directories.

## 0. What the planner found

**Where the S26 prefill goes.** From the orchestrator's `--profile` logs
(`/local/mnt/workspace/htp_moe/hmx/breakdown/npu_g1_r{1,2}.log`), each type
summed over the run, in ms:

| type | r1 | r2 | note |
|---|---|---|---|
| `lfm2_moe` (DSP, 23 M>1 calls) | 344 | 355 | M>1 `dsp=` 14.09 ms/call; with the HMX vote 12.95 (`prof/pH*.log`); `acc` 3.00 → 1.69 ms |
| CPU Q4_0 FCs (in_proj, out_proj, qkv, attn_out, dense FFN) | 505 | 543 | 232 G MAC in total, ≈ 0.46 TMAC/s. in_proj alone is 6.44 G MAC in 14.4 ms per layer |
| `mha_core` (CPU) | 195 | 227 | 6 layers, ≈ 33–38 ms each |
| **`causal_conv1d` (CPU)** | **45** | **44** | **2.5 ms per layer for 3.1 M MAC: single-threaded, column-strided (below)** |
| `split` (conv_chunk) | 13 | 13 | a copy |

**The CPU Q4_0 prefill GEMM path at M = 512 (the S26 app, FC engine `cpu`).**
`fully_connected` → `FloatTensor::dot` → `dotQnK`
(`nntrainer/tensor/float_tensor.cpp:1027-1036`; the CPU ops report no accel
support) → `CpuOps::gemm_q4_0_fp32` (`cpu_ops_table.h:171`) →
`arm_compute_backend.cpp:383-387` → `__ggml_q4_0_4x8_q8_0_GEMM<float>`,
`ggml_interface/ggml_interface_omp.cpp:28-138`. The Android build uses
`thread-backend=omp` (the meson default; `builddir` reads `thread-backend omp`,
`-march=armv8.2-a+fp16+dotprod+i8mm`), so `_bs_threadpool`/`_mixed` are not
built. The driver does the following:

1. Quantize each group of 4 rows with `nntr_quantize_mat_q8_0_4x8`, in
   parallel when M/4 ≥ 16 (`:67-81`).
2. Run `parallel_for` over 32 row chunks of 16 rows × column chunks of
   `clamp(N·32/(64·threads)/4·4, 16, 64)` (`:97-122`).
3. Call `nntr_gemm_q4_0_4x8_q8_0` for each chunk. This is llama.cpp's
   hand-written i8mm asm (`nntr_ggml_impl/nntr_ggml_impl_neon.cpp:277-690`).
   The SVE file carries identical asm and is built only for
   `arm-arch=armv9.2-a`.

The weights are the `--isa ARM` q4_0x4 repack: 4 columns interleaved in 8-byte
chunks, with the nibbles XORed by 0x88. `qkv_layer` goes through the vector
`dot` (`float_tensor.cpp:803-819`), and the CPU ops have no batch entry, so it
loops per weight (`:816`) and quantizes the same activation three times.

**The kernel's numerics, per output element (r, c).** For each block
b = 0 … K/32−1, in order:

* `isum_b` = Σ₃₂ q_w·q_a, an exact int32. It is computed as 4 SMMLA on
  `w<<4` and then `SCVTF #4`. That gives exactly `(float)isum_b`, because
  |16·isum| ≤ 520 192 < 2²⁴.
* `s_b` = fl(d_w·d_a): FCVTL fp16→fp32 is exact, followed by one FMUL.
* `acc = fma((float)isum_b, s_b, acc)`: FMLA, fused. `acc` starts at +0 and
  is stored without a bias.

The 16-row loop (`:374-383`) and the 4-row tail do the same thing. So **the
result does not depend on how rows and columns are cut into kernel calls or
which thread runs them.** Any re-tiling that feeds the kernel the same QA and B
bytes is bit-identical by construction. The x86 build runs the same driver
file with the AVX 4x8 kernel (`nntr_ggml_impl_avx.cpp:78`), so the host can
check the driver's index arithmetic. The scalar fallback
(`nntr_ggml_impl_fallback.cpp:261-312`, `sumf += sumi·dw·da`, not fused) is
**not** a bit spec.

**SME verdict.**

* *Bit-identical: feasible.* Integer block dots are exact, `(float)isum` is
  exact, and streaming SVE `FMLA/FMAD` is fused. A scratch compile with NDK r30
  clang 21 (`-march=armv9.2-a+sme`, `svmopa_za32_s8_m` + `svread_hor_za32` +
  `svcvt` + `svmla`) emits `smopa` ×8, `mov z←za0h`, `scvtf`, `fmad`.
* *Faster than 8-core i8mm: not expected.* Bit-identity forbids deferring or
  re-associating the per-32-K float epilogue. Each 16×16 ZA tile therefore
  pays 8 SMOPA and then 16 × (ZA→Z move, SCVTF, FMUL, FMLA) per block, with
  float accumulators that do not fit in Z registers beyond one or two tiles.
  All of this runs on a streaming unit that known implementations share per
  cluster. i8mm runs on all 8 cores.
* Arm's own answer points the same way. llama.cpp's KleidiAI integration
  (`~/llama.cpp` @ 2026-08-03, `ggml/src/ggml-cpu/kleidiai/kernels.cpp:24,
  319-329, 365-367`) runs Q4_0 GEMM on SME2 as
  `f16p1vlx2_qsi4c32p4vlx2_…_sme2_mopa`: the LHS is converted to fp16 and fed
  to FMOPA, which is not the q8_0 arithmetic. The int8 path is used only for
  GEMV (`…_sme2_sdot`).
* What only the phone can answer: SVL, the number of SME units, streaming
  op throughput, and whether SME2 is present. Step 6 is a bounded probe with a
  kill criterion. The SME kernel is not in this PR.

**Cheaper bit-identical lever that the issue did not list.**
`neon::causal_depthwise_conv1d_k3` (`arm/neon_impl.cpp:2171-2275`) walks
channels outer and time inner. Every `vld1q` is 8 KB away from the previous
one, and the loop runs on one thread. That costs 2.5 ms per layer, 18 layers
× 2.5 = 44 ms, ≈ 4 % of the prefill. Each output lane is
`vfmaq(vfmaq(cur·w0, prev1, w1), prev2, w2)` (`VFMAQ_F32` = `vfmaq_f32` on
aarch64, `:36-40`). A time-outer loop split over channel blocks with
`ThreadManager::parallel_for` (already used in this file at `:2122`) keeps
every lane's operation sequence. It is included as step 3, and the
orchestrator may drop it.

## 1. Goal and gate

Goal (issue): on `R5KL20NFRCK`, NPU prefill above the S25's #158 B
(**509.4 / 423.5 / 405.5 tok/s** at G 64 / 512 / 1024), with no bit changed,
proved under a protocol that survives the S26's 333–574 tok/s run-to-run
noise.

| # | check | pass |
|---|---|---|
| G1 | prefill vs S25 (the goal) | For the shipped variant, at every G: the **median** of its measured runs is > the S25 value, and **≥ 3 of 4** runs are > the S25 value (§4 protocol). Reported in BENCHMARK.md's S26 v81 block with a "vs S25" column |
| G2 | prefill vs A (the lever is real) | shipped median > A's median at G = 64 and at G = 512. The mechanism is visible in a low-noise column: H's level-2 `K=2048 N=2048 M>1` row reads `dsp` ≤ A's − 0.8 ms/call and `acc` ≤ 2.0 ms. HC's conv gtest time is ≤ 0.5 ms (A: ≈ 2.5). HCF's FC microbench gain holds (step 4) |
| G3 | bits | the MoE dumps (`NNTR_HTP_DUMP`: the **input** of every MoE call, prefill M>1 included, so every upstream CPU FC and conv output) of H, HC and HCF are `bit_identical=1` against this sitting's A (`tools/htp/htp_dump_eval.py`). A is `bit_identical=1` against the base's reference dump (185's shipped variant if based on #185, else `/local/mnt/workspace/htp_moe/177/dump/Q4`) |
| G4 | nll + text | the `[PPL] decode step=` lines of every variant, forced on A's `cont.ids` (G = 512), are byte-identical to A's. Text ≡ A on the 8-prompt set (G = 64) |
| G5 | the vote took | H's logcat shows `nntr_hvx_open: hmx vote rc=0x0 … hmx_hz before=… after=…` with after > before. If cDSP FARF does not reach logcat on this build, G2's `acc`/`dsp` columns are the proof and the doc says so |
| G6 | decode guard | every variant's decode median is ≥ A's − 2 % at each G. The vote keeps HMX at TUR for the session, and decode must not pay for it thermally |
| G7 | hygiene | `md5sum -c` passes before every run. Every log has `dspq: close calls=N served=N bad=0`. One banner kind (`… dma_bypass=1 dma_q=4 …`). No `MD5 FAIL`, `EXIT≠0`, `dspq: off`, `gemv: off` |
| standing | prefill ≥ −5 % of A; accuracy | G1/G2 cover the first. On `htp_moe_v81` the accuracy gate is bit-identity to the reference (contract §12, 2026-09-29), covered by G3 and G4. Text vs CPU `q40` is recorded as information only |

Ship rule: the PR ships every commit whose variant passes G3–G6. G1 is read on
the richest variant that passes. If HCF adds < 2 % over HC's median at G = 512,
the FC commit is dropped (it is the only one that changes a shared driver).

## 2. Where it lives

| file:line | change |
|---|---|
| `test/htp/hvx_add_f32.c:126-178` (`nntr_hvx_open`, the `NNTR_HVX_HAVE_HAP_POWER` block), after the bus vote at `:167-176` | **C1, ≈ 25 lines.** Under `#ifdef HAP_POWER_SET_HMX_V2_DEFINED` (`HAP_power.h:60`): `req.type = HAP_power_set_HMX_v2`, `hmx_v2 = {set_power=TRUE, power_up=TRUE, set_clock=TRUE, pick_default=FALSE, target_corner=HAP_DCVS_EXP_VCORNER_TUR, min_corner=TUR, max_corner=HAP_DCVS_EXP_VCORNER_MAX, perf_mode=HAP_CLK_PERF_HIGH, freq_mhz=0, floor_freq_mhz=0}` (`HAP_power.h:434-472, 258, 268, 277`). Around it, `HAP_power_get(s, &r)` with `HAP_power_get_hmx_core_clk_Freq` (`:593`) before and after, then **one `FARF(ALWAYS, "nntr_hvx_open: hmx vote rc=0x%x get_rc=0x%x hmx_hz before=%u after=%u")` line, never a return**. The same best-effort policy as the three votes above: HAP documents `AEE_EBADPARM` on chips without a separate HMX clock, and v79 must run on as before. Match the probe's field set with the orchestrator before committing, because the probe diff is not archived (only `hmx/set_H{1,2}` binaries) |
| `nntrainer/tensor/cpu_backend/arm/neon_impl.cpp:2171-2275` (`causal_depthwise_conv1d_k3`) | **C2, ≈ 30 lines.** For W % 4 == 0: `parallel_for` over channel blocks of 64. Inside a block the loop runs time outer and 4-lane vectors inner. `prev1`/`prev2` are loaded from rows t−1/t−2 (zero for t < 1/2) rather than carried, with the **same** `vmulq`, `VFMAQ_F32`, `VFMAQ_F32`, (+`vaddq` bias) sequence per lane. The scalar tail (`W % 4`) and the decode function stay untouched. The caller is `Applications/CausalLM/layers/causal_conv1d_layer.cpp:129-144`, unchanged |
| `test/unittest/unittest_nntrainer_cpu_backend.cpp:196-270` | new `causal_depthwise_conv1d_k3_prefill_bit_spec` test. The spec is scalar `fmaf(prev2, w2, fmaf(prev1, w1, cur*w0))` (+ b), compared with `memcmp` at (B=1, H=512, W=2048) and (1, 7, 64), with and without bias. It prints the function's µs at 512×2048. Existing tolerance tests stay. It runs on the device (the x86 build dispatches to `avx2::`) |
| `nntrainer/tensor/cpu_backend/ggml_interface/ggml_interface_omp.cpp:28-138` (single-N GEMM), `ggml_interface.h:159-190` | **C3, only if step 4's gate passes.** Move the `(row_chunk, col_chunk policy, task order)` of `:95-122` into an internal `__ggml_q4_0_4x8_q8_0_GEMM_sched(…, const q4_0_sched &)`. The public entry calls it with the chosen constants, so there is no runtime knob. The kernel (`neon.cpp:277-690`) and the QA quantization are untouched. Optional C3b, only if step 4 shows the three qkv quantizations matter: a CPU `gemm_q4_0_batch_fp32` override on the CPU ops (`compute_ops.h:206-213`, `cpu_ops_table.h`) that quantizes QA once and runs the single-N schedule per weight |
| `test/unittest/unittest_nntrainer_ggml_arm.cpp:633-684` (`DISABLED_gemm_q4_0_4x8_benchmark`), `test/jni/Android.mk:862-876` | **T1.** Extend the benchmark to the LFM2 shapes (M = 512; (K, N) = (2048, 6144), (2048, 2048), (2048, 512), (2048, 7168), (7168, 2048)) and a schedule grid. For every grid cell and shape it checks `memcmp == 0` against the unchanged entry, then prints GMAC/s. The DISABLED `quantize_mat_q8_0_4x8` bench (`:392`) gives the per-call quantization cost |
| `test/unittest/unittest_nntrainer_cpu_backend.cpp` (x86 host) | **T2.** The same grid-vs-default `memcmp` on `__ggml_q4_0_4x8_q8_0_GEMM<float>` with 4x4-bl-repacked weights (`nntr_repack_q4_0_to_q4_0_4_bl`), at M ∈ {16, 36, 512}. This proves the driver refactor on the workstation |

**Nothing else moves.** No IDL change (`test/htp/nntr_hvx.idl`, `generate_stub.sh`
untouched). No `HtpComputeOps` change. No quantizer format tag
(`nntr_quantize_stream`) or loader check: the weight layout is unchanged. No
`NNTR_HTP_PROFILE` stage and no `tools/htp_fc_report.py` column. The DSP arena
and address-space budget are untouched: C1 allocates nothing, and C2 and C3 are
ARM-side with the same buffers.

## 3. Design

* **C1: the vote, in `nntr_hvx_open`, for the session's lifetime.** HexKL
  powers HMX with `HMX_v2 set_clock=0` (the lowest HMX clock on chips with a
  separate HMX clock). A second client's `target_corner` is aggregated by max
  (`HAP_power.h:449-452`), so our context `s` can raise it without touching
  HexKL.
  * *Why it preserves bits:* the HMX computes int8×int4 → int32 tiles
    whatever the clock. The pool's dequant/requant and every reduction order
    are fixed per tile, not per timing (doc 45 §3; the M>1 path has no
    timing-dependent assignment that changes arithmetic). G3 checks this.
  * *Rejected alternative:* `pick_default=TRUE` (the HMX clock follows the
    Q6 vote). It was measured: no change.
  * The probe returned the `HAP_power_set` error from `open`. Production must
    not do that: v79 or a future HexKL may reject the request, and the
    session must open anyway.
* **C2: conv loop order + threads.** This is a pure loop transform (§0). It is
  bit-identical by construction, and a device `memcmp` against an `fmaf` spec
  checks it.
  * *Rejected alternative:* move the conv into the HTP conv block. That
    changes the arithmetic and needs the round trips doc 50 measured.
* **C3: re-schedule, not re-kernel.** Only the row/column cut and the task
  order change (§0 shows the per-element sum does not depend on them).
  * Candidates:
    * **S1**, column-major task order: a thread keeps one B slice (64 cols ×
      1152 B) hot while A (1.1 MB) streams from L2.
    * **S2**, row chunk 32/64.
    * **S3**, column chunk 32/128.
  * Adopt the best only if it pays (step 4 gate).
  * *Rejected alternatives:* an SME kernel (§0 verdict; probe in step 6), and
    KleidiAI's SME2 Q4_0 GEMM (fp16 LHS, not bit-identical).
* **Contract checks.**
  * No wall is touched.
  * No CPU fallback is created for `QS4CX_WH`.
  * No new quantizer input, so no `_det` question arises: the conv output
    feeds `conv_out_proj`'s Q8_0 quantizer, but its bits are unchanged by
    construction and G3 checks it.
  * No DMA path changes.

## 4. Steps (ordered by expected gain per effort)

| step | effort | expected, prefill median (A ≈ #177 Q4: 514 / 485 / 513) |
|---|---|---|
| 1 protocol | 0 code | turns #177's +0.9 % at G = 64 (n = 2, means) into a verdict |
| 2 HMX vote (C1) | ≈ 25 lines, skel only | −25 ms of ≈ 1 000 ms, **+2.5 %: ≈ 527 / 497 / 526** |
| 3 conv loop (C2) | ≈ 30 lines + test | −40 ms, **+4 %: ≈ 548 / 517 / 547** cumulative |
| 4 FC schedule measure → C3 | ½ day + device bench | 0 to +10 % FC throughput ⇒ 0 to +4.6 % prefill (FC ≈ 44 % of prefill). Adopted only if ≥ +10 % on the model-weighted FC time |
| 5 E2E sitting | ≈ 2.5 h device | the verdict |
| 6 SME probe | ≤ ½ day, optional | expected: no gain. It closes the question with the phone's numbers |
| not here | — | `mha_core` (17–18 %, ≈ 200 ms): already row-parallel; a bit-identical re-tiling is the next lever, as its own issue. Also out of scope: overlapping the CPU FCs with the DSP's 340 ms of M>1 MoE (the CPU idles then), which needs chunked prefill |

Prelude (every shell):
```
W=/local/mnt/workspace/htp_moe/187; mkdir -p $W
G=/home/j2z0-lee/nntrainer; BASE=htp/185-m1-schedule-overlap
[ "$(git -C $G rev-parse $BASE)" = "$(git -C $G rev-parse htp/177-m1-dma-queues)" ] && BASE=htp/177-m1-dma-queues   # #185 has no commits yet
git -C $G worktree add -b htp/187-s26-prefill $W/wt $BASE   # once
cd $W/wt && git submodule update --init --depth 1 && export HEX_ARCH=v81 && source tools/htp/env.sh && export HEXKL_ROOT HEXKL_SDK_VER
S=R5KL20NFRCK; D=/data/local/tmp/nntrainer/causallm/s187; MR=/data/local/tmp/nntrainer/causallm/models; DD=/data/local/tmp/s187dump
P=<base reference dir: /local/mnt/workspace/htp_moe/185 if based on #185, else /local/mnt/workspace/htp_moe/177>
therm() { adb -s $S shell "echo \$(date +%T) bat=\$(dumpsys battery | sed -n 's/^  temperature: //p') zone0=\$(cat /sys/class/thermal/thermal_zone0/temp) cpumax=\$(cat /sys/devices/system/cpu/cpufreq/policy*/scaling_max_freq | tr '\n' ,) majflt=\$(sed -n 's/^pgmajfault //p' /proc/vmstat)"; }
```
Host builds go to `$W/wt/build` (host `meson setup` flags from the gates skill).
Android builds go to the worktree's own `builddir` (gates skill "fresh
checkout" notes: tokenizer `.a`, `libc++_shared.so`, `prefix`).

**Step 1: the prefill protocol (applies to step 5; no code).**
* **Warm-up.** One discarded G = 8 run per variant at sitting start and after
  any pause > 10 min. It doubles as the sanity run (banner,
  `calls=176 served=176 bad=0`).
* **Cooling.** `cool` to zone0 ≤ 45 °C before every run, logging the wait.
  `therm` runs before and after every run. Its `cpumax` shows a thermal cap,
  and its `majflt` delta shows a cold page cache. Both explain an outlier;
  neither removes it.
* **n.** Per (variant, G): 4 runs from 2 mirrored rounds
  (A H HC HCF HCF HC H A). When a cell has a flagged outlier, run one more
  round for that G.
* **Statistic.** Median, with mean, min..max and n beside it. The S25 column
  is 509.4 / 423.5 / 405.5 (#158 B means, `R3CY10WM83Y`).
* **Verdict.** G1 and G2.
* **Binaries.** tok/s never comes from a `--profile` build.

Medians:
`for f in $W/logs/tps_*; do …; done | python3 -c 'import sys,statistics as s,collections as c; d=c.defaultdict(list); [d[(l.split()[0],l.split()[1])].append(float(l.split()[2])) for l in sys.stdin]; [print(k, s.median(v), min(v), max(v), len(v)) for k,v in sorted(d.items())]'`
(input lines: `variant G prefill`).

**Step 2: C1.**
* Gate, rung 1 subset: `bash tools/htp_syntax_check.sh` exits 0, and
  `bash test/htp/host/run_host_checks.sh` prints `ALL CHECKS PASS` (no host
  code changed, so this proves nothing broke).
* Gate, rung 2: `HEX_ARCH=v79 ./test/htp/build.sh` (compile-only, then
  `Flags: 0x79`, not run). Then `HEX_ARCH=v81 ./test/htp/build.sh` →
  `UNDEFINED SYMBOLS OK` (`HAP_power_get` falls under the `HAP_` allowance,
  `build.sh:125`), `hexagon-readelf -h … | grep 'Flags:.*0x81'`.
* Copy to `$W/set/H/libnntr_hvx_skel.so` and record the md5 (skel md5s do not
  reproduce across builds, measurement 177).
* Commit `[HTP] nntr_hvx_open: vote the HMX clock to TURBO (#187)`.

**Step 3: C2 + its test.**
* Host gate, rung 1: `ninja -C $W/wt/build`, then
  `unittest_nntrainer_cpu_backend` passes. This proves only that the code
  compiles and the x86 path is untouched: the NEON function does not run on
  x86, and the plan says so.
* Rung 3 (app, `build_android.sh --htp` in the worktree), then
  `ndk-build unittest_nntrainer_cpu_backend` in `test/jni`.
* **Device (agent, S26, when the phone is free):**
  `NNTR_NUM_THREADS=8 ./unittest_nntrainer_cpu_backend --gtest_filter='*conv1d*'`.
  Pass: the bit-spec test passes, and the old and new timings are printed.
  Expected ≈ 2.5 ms → ≤ 0.5 ms.
* **Negative, once, pasted in the PR:** swap the two `VFMAQ_F32` lines → the
  bit-spec test fails. Revert.
* Commit `[CausalLM/cpu] causal conv1d prefill: time-outer, channel-parallel (#187)`.

**Step 4: FC schedule, measure first (device unavoidable).**
* T2 on the host must print `memcmp` = 0 on every grid cell (rung 1).
* Build T1 with `ndk-build unittest_nntrainer_ggml_arm` against the step 3
  app build.
* On the S26: `NNTR_NUM_THREADS=8 ./unittest_nntrainer_ggml_arm --bench --gtest_filter='*gemm_q4_0_4x8*:*quantize_mat*'`,
  cooled, 3 repetitions, one line per (shape, schedule): GMAC/s and memcmp.
  Also run at 4 and 6 threads for the default schedule only. That separates
  compute-bound from sync-bound, against the issue's 354/421/482–530.
* **Gate F:** the best schedule is bit-equal on every shape, and it cuts the
  layer-weighted time Σ(in_proj×18, out_proj×18, q×6, k×6, v×6, attn_out×6,
  up×2, gate×2, down×2) by ≥ 10 %, with no single shape > 2 % slower.
  * Pass: C3 with those constants, then T1/T2 again, rung 1 and rung 3.
  * Fail: no C3. Record the table in the measurement doc and LEDGER.
  * C3b only if the quantization bench puts 2 extra quantizations per
    attention layer at ≥ 1 ms.

**Step 5: the E2E sitting (device unavoidable; ≈ 2.5 h with cool-downs).**
* Confirm the phone is free (`adb -s $S shell pidof nntrainer_causallm
  unittest_hvx_dma_probe` prints nothing, and the orchestrator has released
  it).
* Variants: ≤ 4, full model, prompt 512, G 64 / 512 / 1024, directories
  `$D/{A,H,HC,HCF}`. Each is a full binary set plus `md5.txt` and
  `variant.env` (`export NNTR_MOE_DMA_QUEUES=4` in all four, harmless if the
  default already is 4). The model config is rewritten only in `$D/model`,
  with symlinks to `$MR/q40-qs4cx-wh`, as 185 §7 does.

| variant | skel | app (`nntrainer_causallm`, `libcausallm_core.so`, `libnntrainer.so`, …) |
|---|---|---|
| A (reference, unchanged base) | base head's skel | base head's app set |
| H | `$W/set/H` | A's app set |
| HC | `$W/set/H` | C2 commit's app set |
| HCF (only if C3 exists) | `$W/set/H` | C3 commit's app set |

* **Run steps.** Reuse 185 §7's `run()`/`cool()` unchanged, with the `therm`
  above. It already saves `adb logcat -d` per run, which is G5's source.
  * **5.1** Warm-up/sanity, G = 8, per variant.
  * **5.2** tok/s: `for g in 64 64 512 512 1024 1024` (each a mirrored round
    A H HC HCF HCF HC H A), then one extra round for any G with a flagged
    outlier.
  * **5.3** Level-2 profile, G = 64, one per variant (`NNTR_HTP_PROFILE=2`):
    the M>1 row's `dsp`, `acc`, `mm` (G2).
  * **5.4** Dumps and nll, as in 185 §7.4, with A against `$P`'s reference
    (G3, G4).
  * **5.5** Text on 8 prompts, G = 64 (G4).
  * **5.6** 185 §7.6's greps (G7), plus `grep -h 'hmx vote' $W/logs/*.logcat | sort | uniq -c` (G5).
* **Clean-up.** Remove `$D` and `$DD`.
* Record the sitting in `docs/measurements/187-s26-prefill.md` (177's shape:
  commits, md5s, serial, per-run table with therm, cpumax and majflt, the
  medians table with "vs S25" and "vs A"). Commit it and open the PR.

**Step 6: SME probe (optional, bounded, not in the PR).**
* Build a standalone probe in `$W/sme/` (NDK r30,
  `-march=armv9.2-a+sme[+sme2]`, `__arm_locally_streaming`), linked against
  the step 3 `libnntrainer.so` for `nntr_quantize_mat_q8_0_4x8` and the
  repack. It prints:
  1. The `getauxval(AT_HWCAP2)` bits SME / SME2 / SME_I8I32 and SVL (`svcntsb`).
  2. SMOPA-only throughput at 1/2/4/6/8 threads, to count the shared units.
  3. The bit-identical per-block kernel of §0 (int8 SMOPA on `w>>4`-exact
     nibbles, then ZA→Z, SCVTF, FMUL, FMLA in block order) on
     (512, 2048, 6144): `memcmp` against `nntr_gemm_q4_0_4x8_q8_0`, and
     GMAC/s against step 4's i8mm number at 8 threads.
* Kill criterion: the best aggregate is < 1.3× the 8-core i8mm figure, or
  `memcmp` ≠ 0. Then close with a LEDGER verdict. Only a pass files a
  follow-up issue.

## 5. Risks

* **Thermal drift and DVFS between runs and sittings.** Controls: the
  median-of-4 rule with "≥ 3 of 4 above S25" (G1), the mirrored order, and
  `therm`'s `cpumax` and zone0 per run. The S25 column is a cross-device,
  cross-sitting mean by definition of the goal, so G1 is read only on the
  median rule. G2's profile/gtest columns are the low-noise evidence (DSP
  columns move ≤ 4 % between sittings, LEDGER rules 9 and 20).
* **First-run-after-idle.** The warm-up is discarded, and `majflt` per run
  names page-cache cold starts. Such runs are flagged, not dropped.
* **The HMX vote adds heat** during prefill that later decode may pay for at
  G = 1024. G6 is the guard, and zone0 post per run shows it.
* **Stale or mixed skel.** All banners are identical by design. Only
  `md5sum -c` before each run tells H from A on the phone. The profile `acc`
  column and the logcat line tell them apart afterwards. `libcdsprpc*` is
  never staged.
* **Microbench vs in-app.** In the app the CPU FCs alternate with DSP waits,
  and the cores may clock down in between. A step 4 win can therefore shrink
  in HCF. The E2E medians decide, and the 2 % drop rule in §1 applies.
* **Host coverage.** The NEON conv and the i8mm asm do not run on the x86
  workstation (no qemu installed). Host checks cover only the driver's
  schedule arithmetic (T2) and compilation. Bits of C2 and C3 on ARM are
  proven on the device: the gtest `memcmp`, then the E2E dumps.
* **DMA rate / address space.** Untouched. The M>1 DMA–compute overlap may
  shift with a faster HMX, which the level-2 row shows (`mm`, `stage`).
* **SME OS support** (streaming-mode context switch on Android) is unknown.
  The probe fails fast.

## 6. Docs to update

* `docs/htp_moe/BENCHMARK.md`: the S26 v81 prefill rows (median, min..max, n,
  "vs S25", "vs A") for A, H, HC, HCF at G 64 / 512 / 1024. Add a Method line
  for the step 1 protocol.
* `docs/htp_moe/LEDGER.md`, new items:
  1. HexKL leaves HMX at the lowest clock; the `HMX_v2` TUR vote gives M>1
     `acc` −44 % and `dsp` −8 % on v81, bit-identical.
  2. The prefill protocol rule (median of ≥ 3 cooled runs, warm-up discarded,
     ≥ 3/4 above the reference).
  3. The CPU Q4_0 GEMM is tiling-invariant bit-for-bit (the §0 numerics), and
     the step 4 schedule result.
  4. The SME verdict (bit-feasible, not throughput-favourable for block-32
     Q4_0; KleidiAI uses fp16 MOPA), plus step 6's numbers if run.
  5. `causal_conv1d` prefill was 44 ms from a strided single-thread loop.
* The contract's "now" rows stay the supervisor's call.
