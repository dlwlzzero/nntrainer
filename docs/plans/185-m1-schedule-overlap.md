# 185 — Hide the decode MoE call's schedule cost on the S26 (v81, 4 DMA queues)

Issue: dlwlzzero/nntrainer#185 (p0). Read against `htp/177-m1-dma-queues` @
`926819a2` (PR #181, open, base `htp_moe_v81`). Work branch
**`htp/185-m1-schedule-overlap`** from `htp/177-m1-dma-queues`, PR into
**`htp_moe_v81`**, stacked on #181. Device: `adb -s R5KL20NFRCK` only
(contract §12, 2026-09-29). Never touch `R3CY10WM83Y` or `R3CN80CW3FY`.

## 0. What the planner found (it changes the issue's reading)

**The call is DMA-bound, so GU(0) is transfer time, not schedule time.**
Q4's M==1 call moves 21 504 KB. At the in-app 4-queue rate (62.1 GB/s,
measurement 177) that is ≈ 346 µs of the 456 µs `dsp`. The measured stages
fit a simple model: DMA ≈ 59 µs per gate_up and ≈ 29.5 µs per down, per-expert
compute ≈ 34 µs (A) and ≈ 17 µs (C) on 6 lanes (`swiglu(hidden) 1228.3`
lane-µs split by tile reads, 2 : 1), and ≈ 2 µs per fork/join:
`4·(59+2) + 4·(17+2) = 320` = the measured `mm 320.4`, and B = D(2)+D(3) ≈ 59
(measured `requant 57.5`). GU(0) has to land before any A compute, and the
only work in front of it is the scan and pack (`quant 6.9`). So at most
≈ 7 µs of the "62 µs exposed GU(0)" can be hidden. The rest is transfer
time that every schedule pays.

**What can be recovered is the time the DMA engines sit idle.** Today that is:

| DMA idle | µs/call | recoverable by |
|---|---|---|
| scan + slot order + pack, between the GU(0) run and A(0) | ≈ 7 | issue GU(0) as a workers-only job and do the scan/pack on the caller meanwhile (§3 Q) |
| C(0)..C(3): no transfer rides them | ≈ 76 | D(2) → C(0), D(3) → C(1) (§3 D); C(2), C(3) remain idle |
| B's requant hidden under D(2)/D(3); once they leave, B is DMA-idle | ≈ 12 | fold requant(i) into A(i+1)'s run and requant(n−1) into C(0)'s (§3 R) |

**"The previous call's tail" is not available.** The expert set of call L is
an input of call L (the router runs on the ARM before it). The DSP cannot
start GU(0) during call L−1. This plan does not pursue it.

Expected, not promised (the same model): `dsp` 456 → ≈ 419 µs/call, which is
≈ −0.81 ms/token over 22 calls. Decode would move from 50.61 / 49.77 / 47.41 to
≈ 52.8 / 51.9 / 49.3 tok/s, against the S25's 51.82 / 50.61 / 47.36. The
lower bound of this slab scheme is ≈ 346 + C(3) + ≈ 13 of scatter, stage and
copy, ≈ 380 µs.

## 1. Goal and gate

Goal (issue): on `R5KL20NFRCK` with `NNTR_MOE_DMA_QUEUES=4`, S26 decode above
the S25's #158 B (51.82 / 50.61 / 47.36 tok/s at G 64 / 512 / 1024), with no
bit changed.

| # | check | pass |
|---|---|---|
| G1 | bits | the MoE dumps of DQ and DQR (G=64) are `bit_identical=1` against this sitting's A (`tools/htp/htp_dump_eval.py`). A's dump is `bit_identical=1` against 177's Q4 dump (`/local/mnt/workspace/htp_moe/177/dump/Q4`) |
| G2 | nll | the `[PPL] decode step=` lines of DQ and DQR, forced on A's `cont.ids` (G=512), are byte-identical (`cmp`) to A's. A's are byte-identical to `177/logs/ppl_Q4.log` |
| G3 | text | DQ ≡ A and DQR ≡ A on the 8-prompt set (G=64, 177's `strip` + model-path filter). A ≡ `177/logs/text_Q4_p0{1..8}.log` |
| G4 | speed vs S25 (the goal) | the **shipped** variant's mean decode tok/s from the mirrored runs is above 51.82 at G=64, above 50.61 at G=512 and above 47.36 at G=1024 (BENCHMARK.md, S26 v81 block, "vs S25" column). In the same sitting its G=512 mean is above A's max |
| G5 | prefill (standing) | the shipped variant's prefill mean is ≥ −5 % of A's at every G. The M>1 path runs no new code (§2), so a miss is thermal and gets the re-run round (step 7.2) |
| G6 | the profile says why | level-2 `K=2048 N=2048 M==1` row: DQR's `dsp` is below A's by ≥ 25 µs/call, `requant` ≈ 0 (no B run at n=4), and `dmaq=4.00`. DQ's `rest` is below A's (GU(0) remainder after QUANT) and its `requant` ≈ 12 |
| G7 | run hygiene | every log has `dspq: close calls=N served=N bad=0`. `md5sum -c` passes before every run. Every banner reads `applied=0x1f03e1 … dma_bypass=1 dma_q=4`. zone0 ≤ 45 °C at each run's start. No `MD5 FAIL`, `dspq: off`, `gemv: off`, `arena: cannot`, `EXIT≠0` |
| G8 | text vs CPU (standing, amended) | on `htp_moe_v81` the accuracy gate is bit-identity to the reference (contract §12, 2026-09-29), covered by G1–G3. Text vs CPU `q40` is recorded as information only |

Ship rule: ship DQR if it passes G4 and its G=512 mean ≥ DQ's. Ship DQ (drop
commit 3) if DQR is not better than DQ in the sitting. If neither passes G4,
the PR still ships the better variant when it beats A outside A's spread,
and the ≈ 40 µs to the lower bound goes to a new issue (§5).

## 2. Where it lives

| file:line | change |
|---|---|
| `nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c:588-600` (the N-queue section comment and its ponytail), `:1636-1638` (the #177 ponytail at run B) | rewrite them to describe the new run table (§3). Delete the ponytail, since this issue is its upgrade |
| `…moe.c:891-911` (`moe_m1_qrun`) | two fields: `uint32_t rq` (the expert whose requant rides this run, `UINT32_MAX` = none) and `const moe_m1_ctx *rq_c`. `moe_m1_q_run` (`:966-974`) resets `rq` after the join, as it does `n_push` |
| `…moe.c:737-754` (`moe_m1_requant_worker`) | move the loop body into `static void moe_m1_requant_one(const moe_m1_ctx *c, uint32_t u)`. The worker calls it per unit (same instructions, same order) |
| `…moe.c:934-961` (`moe_m1_q_worker`) | after `inner`, before the poll: `if (q->rq != UINT32_MAX && i == n - 1u) moe_m1_requant_one(q->rq_c, q->rq);`. Lane n−1 is a non-DMA lane whenever n > nq |
| `…moe.c` new, next to `moe_m1_q_run` | `moe_m1_q_submit(pool, q)`: `hvx_worker_pool_submit(pool, moe_m1_q_worker, q, q->nq)`. `moe_m1_q_join(pool, q)`: `hvx_worker_pool_wait(pool); q->n_push = 0;`. About 10 lines |
| `…moe.c:1463-1475` (GU(0) under N > 1) | when `hvx_worker_pool_workers(pool) >= m1q.nq`: post GU(0), `moe_m1_q_submit`, set `gu0_async = 1`. Otherwise keep today's `moe_m1_q_run` (the fallback) |
| `…moe.c:1489` (the scan) | pass `gu0_async ? NULL : pool`. Only the M ≤ 4 feed path can have `gu0_async`. `hvx_quant_rows_u8_params` computes each row with the same `quant_row_params_one` whichever thread runs it (`hvx_quant_u8.c:98-119`), and at M = 1 it already runs inline (`hvx_worker_pool.c:422`) |
| `…moe.c:1549-1566` | after `HEXKL_PROBE_ADD(QUANT)`, and before `HEXKL_PROBE_COUNT(M1_FEED, nq_used)` (which must read the job's `nq_used`): `if (gu0_async) { T0; moe_m1_q_join; ADD(DMA_FIRST); COUNT(DMA_FIRST_KB) }`. `DMA_FIRST` then times the remainder exposed after QUANT, as the N = 1 path's does |
| `…moe.c:1582-1602` (stage A, N > 1) | before run A(i), for i ≥ 1: `m1q.rq = i − 1` |
| `…moe.c:1601-1602` (the last A's post) and `:1634-1661` (B and stage C, N > 1) | **the downs of the last A's pair** {j0, j0+1} = {2(i&1), 2(i&1)+1}, i = n−1: D(j) rides run C(j − 2) when j ≥ 2, and run B when j < 2. **B** runs only when it carries a down, or when n = 1. It carries requant(n−1) plus its downs; under C3 it no longer does the other requants. Otherwise requant(n−1) rides C(0). #177's `D(j+3)` rule for n > 4 stays. For n = 4 the table is in §3. `REQUANT` times B when it runs and is 0 otherwise |
| `nntrainer/tensor/htp_backend/hvx/hvx_worker_pool.h` (after `:40`), `hvx_worker_pool.c` (`n_workers` at `:94`) | `uint32_t hvx_worker_pool_workers(const hvx_worker_pool *pool)`: returns `pool ? pool->n_workers : 0`. Three lines |
| `test/htp/host/moe_layer_host_check.c:405-440` (pool stand-ins) | **submit** emulates the workers-only lanes: `n = min(n_units, g_workers)` (default `PF_LANES − 1` = 5), `++g_run`, lane ids 0..n−1, then `score_run_end()`. While the job is pending (until `wait`), a `run` or `submit` on a feed cell (`g_score_on`) fails with `FEED pool used while a submitted job is in flight`. **run** fails with `FEED nested pool run` when it is called inside a run. New stand-in `hvx_worker_pool_workers` returns `g_workers` |
| `moe_layer_host_check.c:52-160` + `test/htp/host/standin/hvx_scalar.h:53-64`, `standin/hvx_scalar.c` (`:87-100` gemv, `:106` quant rows, `:136` pack, `:168` dequant tile, `:272` dequant swiglu) | **dataflow scoreboard (new, ≈ 40 lines)**. A third hook, `buf(const void *p, size_t bytes, int write)`, is called by those stand-ins for their reads (act_ah / mid_ah, gate_f32, rq_scale / rq_zp) and writes (gate_f32, rq_*, mid_ah, res). The check keeps the current run's accesses (lane, range, r/w) and fails with `FEED cross-lane RAW/WAR in run %u` when a read and a write from different lanes of one run overlap. Only on feed cells. Initializers `{NULL, NULL}` elsewhere stay valid in C |
| `moe_layer_host_check.c:945-949` (case table) | add cases `{1}` and `{1, 1}` (M=1, n = 1 and 2) so both the B and no-B branches and n < 4 run. Today's cases give n = 4, 6, 10, 3, 5. Add one cell with `g_workers = 3` and N = 4 so the fallback GU(0) run is exercised. New pass line: `M1 FEED OVERLAP OK (n=1..6,10; B only for odd n; downs ride C(j−2); requant(i) rides A(i+1); GU(0) job beside QUANT; dataflow clean)` |
| `test/htp/host/hvx_scalar_stubs.c:69-89` | a `hvx_worker_pool_workers` stand-in (returns 0, which keeps the fallback). The conv-block check links the MoE file (`run_host_checks.sh:52-60`) |

Consumers checked that do **not** move: the IDL `test/htp/nntr_hvx.idl` (no
new argument or flag, so nothing for `generate_stub.sh` beyond `build.sh`'s
usual regeneration). `HtpComputeOps` (`htp_compute_ops.cpp`): no new knob or
slot. The M==1 row's `requant` / `rest` / first-chunk fields change meaning
only (see §6). The quantizer's `QS4CX_WH` tag (`nntr_quantize_stream`) and the
loader check: no byte-layout change. `NNTR_HTP_PROFILE` stage slots: none
added. `DMA_FIRST` under N > 1 goes back to "exposed remainder", as for
N = 1. `tools/htp_fc_report.py` parses neither the M==1 row nor the banner.
The M>1 HMX path is not touched (the scan keeps `pool` there). The N = 1
ring branch is unchanged. The flags word does not change: the new schedule
is the N > 1 schedule. There is no knob; A is #177's skel.

## 3. Design

**Chosen: keep #177's invariant and move work between runs.** The invariant:
a slice is issued and polled by the same lane inside one pool job, and it
is read only by a later job. Three commits, each bit-preserving:

* **C1 (D):** the last A's two downs ride the C runs two ahead of their reader.
* **C2 (Q):** GU(0) goes out as a workers-only `submit` job, and the caller
  does the scan, slot order and pack before it joins.
* **C3 (R):** requant(i) rides run A(i+1) on lane n−1, and requant(n−1)
  rides C(0). B is left only for odd n.

Decode shape (M = 1, n = 4, 6 lanes, nq = 4), before → after (DQR):

| job | #177 (A) carries → dest | DQR carries → dest | compute (DQR) | est. wall A → DQR |
|---|---|---|---|---|
| GU0 | GU(0) → slab 0 (`run`, caller = lane 0) | GU(0) → slab 0 (`submit`, workers 0..3) | caller: scan + slot + pack (QUANT) | 61 + 7 → 61 |
| A(0) | GU(1) → slab 1 | same | A(0) | 61 |
| A(1) | GU(2) → slab 0 | same | A(1) + **requant(0)** | 61 |
| A(2) | GU(3) → slab 1 | same | A(2) + **requant(1)** | 61 |
| A(3) | D(0), D(1) → slab 0 | same | A(3) + **requant(2)** | 61 |
| B | D(2), D(3) → slab 1 | *(no run)* | — | 59 → 0 |
| C(0) | — | **D(2) → slab 1 half 0** | C(0) + **requant(3)** | 19 → 31.5 |
| C(1) | — | **D(3) → slab 1 half 1** | C(1) | 19 → 31.5 |
| C(2), C(3) | — | — | C(2), C(3) | 38 → 38 |

**How the consumer still waits before it reads.** D(2)'s slices are polled
by their issuing lanes inside C(0), before C(0)'s join. C(2) reads them two
joins later. D(3) is polled inside C(1) and read by C(3). GU(0)'s slices are
polled by worker lanes 0..3 inside the job. `moe_m1_q_join`
(`hvx_worker_pool_wait`) returns only after those units return, and A(0) is
the next job (`hvx_worker_pool_run` also waits for an outstanding submit
first, `hvx_worker_pool.c:421`). requant(i) reads gate_f32(i), which is
complete at A(i)'s join, one job earlier. It writes mid_ah(i) / rq(i), which
are read by C(i), a later job. No lane waits on another lane's descriptor,
and no issuer blocks before its poll. The rejected "blocked issuer" question
of plan 177 §3 stays out of the design.

**Why the slabs stay legal.** Slab 1 holds GU(3) until A(3)'s join, and
C(0) starts after it. D(2)/D(3) are the same destinations as in #177, one or
two jobs later. For odd n the pair is {0, 1}: D(0) must land before C(0),
so B stays (it carries D(0), D(1) and requant(n−1)). For n > 4, C(1) carries
D(3) plus #177's D(4). That is ≤ 2 pushes per job, which `g_m1q_desc[4][2]`
holds. The host scoreboard already fails on `FEED push onto a live slab`.

**Why arithmetic and order are unchanged.** The GEMV units, their lane
slicing (`moe_m1_slice` over the same `[u_lo, u_hi)` and the same n; lane
n−1 only appends work after its units), the dequant/SwiGLU, the requant
(`moe_m1_requant_one` = today's loop body, per expert, the same inputs), the
row scan (per-row function, `quant_row_params_one`) and the caller's
scatter order (`:1680-1690`) do not change. Each VTCM byte has one writer
per push, the same bytes as #177, landed earlier or later. The int32 sums
and the f32 output are therefore the same bytes. The host check requires
byte-equality against N = 1 on every feed cell, and G1–G3 check it on
silicon. No quantizer input changes, so no `_det` spec is due (doc 45
§3.3). The new host dataflow check is the "one runnable check" for C3.

Contract checks (§2 and doc 45 §3): the arena is unchanged (7 MiB of slabs,
the same 1 KiB of static descriptors), with no heap and no address-space
growth. `QS4CX_WH` keeps no CPU fallback. The DMA stays behind compute
everywhere except GU(0), and that is inherent (§0). The activation-handle
path is not touched.

**Rejected: the issue's literal "D(3) into C(0)", with D(2) left in B.**
Under the one-job-per-push rule a job lasts max(compute, DMA). B's compute
(≈ 12) covers less of a down's 29.5 µs than a C run (≈ 17), and B then
still cannot shrink below a down. Estimated B + C: D(2) in B and D(3) in C(0)
gives 31.5 + 31.5 + 3·19 = 120. D(2) in C(0) and D(3) in C(1) gives
12.4 + 31.5 + 31.5 + 2·19 = 113 (C1 alone). With C3 it gives 101.

**Rejected: running the scan and pack as lane 0's inner work in the GU(0)
`run`**, which keeps the caller as a DMA lane. That needs lines 1488-1548
(the scan and slot loop shared with the prefill path, plus the m1 pack and
carve) moved into a context struct of ≈ 25 fields, for the same ≈ 7 µs.
`submit` keeps those lines where they are. Its one exposure, a pool with
fewer than nq workers, falls back to today's run through
`hvx_worker_pool_workers`.

`ponytail:` C(2) and C(3) still move no bytes (≈ 38 µs/call DMA-idle). The
next rung is a third down slot in the arena's spare ≈ 0.9 MiB (half a down)
or row-splitting a down over two jobs. Both need their own host proof, and
neither is in this PR.

## 4. Steps

Every shell starts with:
```
cd <your checkout> && git fetch origin && git switch -c htp/185-m1-schedule-overlap origin/htp/177-m1-dma-queues
export HEX_ARCH=v81 && source tools/htp/env.sh && export HEXKL_ROOT HEXKL_SDK_VER
S=R5KL20NFRCK; W=/local/mnt/workspace/htp_moe/185; P=/local/mnt/workspace/htp_moe/177
D=/data/local/tmp/nntrainer/causallm/s185; MR=/data/local/tmp/nntrainer/causallm/models; DD=/data/local/tmp/s185dump
therm() { adb -s $S shell "echo \$(date +%T) bat=\$(dumpsys battery | sed -n 's/^  temperature: //p') zone0=\$(cat /sys/class/thermal/thermal_zone0/temp)"; }
```
(The shared checkout at `/home/j2z0-lee/nntrainer` may be in use. If so,
work in a `git worktree add $W/wt htp/177-m1-dma-queues` and branch there.)

**Step 1 — the harness first (host only).** Add the submit-lane emulation,
the pending/nested flags, the `hvx_worker_pool_workers` stand-ins
(`moe_layer_host_check.c`, `hvx_scalar_stubs.c`), the `buf` hook and the
dataflow scoreboard, and the n = 1 / 2 cases, all against **#177's
unchanged kernel**. Gate (rung 1 subset): `bash test/htp/host/run_host_checks.sh`
→ `ALL CHECKS PASS`, `WORKER POOL LANES OK`, `M1 FEED QUEUES OK`,
`M1 GEMV VTCM FEED SCHEDULE OK`, and a clean dataflow line on #177's
schedule (#177's B requant is a separate job, so it must be clean). Commit
(test-only). This proves the new checks do not fire on a known-good schedule.

**Step 2 — C1 (D).** The pool accessor is not needed yet. Change the last-A
post and the B/C loop (§2 rows 8 and 9). Gate: `run_host_checks.sh` →
`ALL CHECKS PASS` with `M1 FEED QUEUES OK` (output byte-equal to N = 1 on
every feed cell, n = 1..6, 10). **Negative, once, pasted in the PR:** post
D(2) on C(2)'s run instead of C(0)'s → the check prints `FEED read not
covered: … slice read in the run that issued it` and fails. Revert.

**Step 3 — C2 (Q).** Add the accessor, `moe_m1_q_submit/join`, the GU(0)
branch, the scan's `NULL`, and the join before `M1_FEED`. Gate: as step 2,
plus the fallback cell (`g_workers = 3`) byte-equal with `fed == 4`.
**Negative:** delete the `moe_m1_q_join` → `FEED pool used while a
submitted job is in flight` (A(0)'s run), and the check fails. Revert.

**Step 4 — C3 (R).** `moe_m1_requant_one`, the `rq` fields, the A / C(0)
assignments, and B only for odd n. Gate: as step 2, plus the new
`M1 FEED OVERLAP OK (…)` line. **Negative:** set `m1q.rq = i` (A(i)'s own
run) → `FEED cross-lane RAW/WAR in run …` on the real shape (56 units over
6 lanes: lane 5 reads gate_f32 that lanes 0–4 write). Revert.

**Step 5 — rung 1 in full, then rung 2.** `ninja -C build`, the two gtests
of the gates skill, `bash tools/htp_syntax_check.sh` (exit 0),
`bash test/htp/host/run_inproc_e2e.sh` twice: unset, then with
`NNTR_MOE_DMA_QUEUES=4` exported (all of its pass lines, including `E2E eval
golden … bit_identical=1`, `E2E tokens htp==cpu 8/8`, `INPROC E2E PASS`). It
compiles the real pool, so it proves the accessor links. Skel: at C2's
commit (in a worktree, `git worktree add $W/wt-dq <C2 sha>`), run
`HEX_ARCH=v81 ./test/htp/build.sh` → `UNDEFINED SYMBOLS OK`, then
`hexagon-readelf -h … | grep 'Flags:.*0x81'`, and copy it to
`$W/set/DQ/libnntr_hvx_skel.so`. Repeat at C3 for `$W/set/DQR/`. Build v79
once (`HEX_ARCH=v79`, `Flags: 0x79`, not run), then v81 last. Record the md5s
(they do not reproduce across builds, measurement 177. The staged file is
the reference).

**Step 6 — rung 3.** `(cd Applications/CausalLM && ./build_android.sh --htp --cache)`
with the skill's `NEEDED` / `strings` / md5 checks, for the PR gate. This
issue changes no ARM code. **All three variants stage 177's app set**
(`$P/A/`: `nntrainer_causallm 0c0e0f78…`, `libcausallm_core.so de1323e8…`,
`libnntrainer.so 2657fdae…`, `libccapi-nntrainer.so 65c9034c…`,
`libc++_shared.so b1586b9b…`, `libsdkl.so 0ad4e22a…`), so only the skel
differs. Record rung 3's md5s next to them.

**Step 7 — the device A/B (unavoidable silicon step; agent-run on
`R5KL20NFRCK` under the v81 exception; ≈ 70 min with cool-downs).** Before
starting, confirm that the phone is free (`adb -s $S shell pidof
nntrainer_causallm unittest_hvx_dma_probe` prints nothing) and that the
orchestrator has released it. Record the run as
`docs/measurements/185-m1-schedule-overlap.md` in 177's shape.

| variant | skel | `variant.env` | expected banner |
|---|---|---|---|
| A (reference: #177 Q4, unchanged) | `$P/Q4t/libnntr_hvx_skel.so` (`0200414e…`, 177 head `a1a43a28` kernel) | `export NNTR_MOE_DMA_QUEUES=4` | `applied=0x1f03e1 … dma_bypass=1 dma_q=4 source=default` |
| DQ (C1 + C2) | `$W/set/DQ/…` | same | same |
| DQR (C1 + C2 + C3) | `$W/set/DQR/…` | same | same |

All three banners are identical by design. Only the `md5sum -c` before every
run tells the skels apart on the phone, and 7.3's profile tells them apart
afterwards (G6).
```
for v in A DQ DQR; do mkdir -p $W/$v && cp $P/A/{nntrainer_causallm,libcausallm_core.so,libnntrainer.so,libccapi-nntrainer.so,libc++_shared.so,libsdkl.so,prompt512.txt,bitset-0*.txt} $W/$v/
  echo 'export NNTR_MOE_DMA_QUEUES=4' > $W/$v/variant.env; done
cp $P/Q4t/libnntr_hvx_skel.so $W/A/; cp $W/set/DQ/libnntr_hvx_skel.so $W/DQ/; cp $W/set/DQR/libnntr_hvx_skel.so $W/DQR/
for v in A DQ DQR; do (cd $W/$v && md5sum $(ls | grep -v md5) > md5.txt); done   # never stage libcdsprpc*
adb -s $S devices -l; adb -s $S shell df -h /data; therm
adb -s $S shell "md5sum $MR/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin"    # 7b7867fa…
adb -s $S shell "rm -rf $D && mkdir -p $D/model && cp $MR/q40-qs4cx-wh/*.json $D/model/ && \
  ln -s $MR/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin $MR/q40-qs4cx-wh/tokenizer.json $D/model/"   # as 177: only s185/model's config is rewritten
for v in A DQ DQR; do adb -s $S shell "mkdir -p $D/$v" && adb -s $S push $W/$v/. $D/$v/ >/dev/null
  adb -s $S shell "chmod 755 $D/$v/nntrainer_causallm; cd $D/$v && md5sum -c md5.txt" | tee $W/$v/md5_device.log; done
cool() { for k in $(seq 30); do t=$(adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'); [ "$t" -le 45000 ] && break; sleep 10; done; echo "cool zone0=$t waited=$((k*10))s" | tee -a $W/logs/cool.log; }
run() { # run <variant> <G> <log> <prompt> [extra env]
  L=$W/logs; mkdir -p $L; echo "pre $(therm)" > $L/$3.therm; adb -s $S logcat -c
  adb -s $S shell "cd $D/$1 && md5sum -c md5.txt >/dev/null || { echo MD5 FAIL; exit 9; } && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' $D/model/nntr_config.json && \
    . ./variant.env && NNTR_NUM_THREADS=8 $5 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $D/model \"\$(cat $4)\"; echo EXIT=\$?" 2>&1 | tee $L/$3.log \
    | grep -E '^(prefill|generation)|moe m1 gemv|dspq: close|MD5 FAIL|EXIT='
  echo "post $(therm)" >> $L/$3.therm; adb -s $S logcat -d -v time > $L/$3.logcat
}
```
7.1 **Sanity and warm-up, G=8:** `for v in A DQ DQR; do cool; run $v 8 sanity_$v prompt512.txt; done`
→ the banner and `calls=176 served=176 bad=0`. These runs also absorb the
first-run-after-idle outlier (177: 34.88*) and are not read as tok/s.

7.2 **tok/s, prompt 512, mirrored** (A DQ DQR DQR DQ A per round): G=512
twice, then G=64 and G=1024 once each. If G5 misses on the G=512 means, run
one more G=512 round (177's s5 precedent).
```
k=0; for g in 512 512 64 1024; do k=$((k+1)); r=0; for v in A DQ DQR DQR DQ A; do r=$((r+1)); cool; run $v $g tps_${v}_G${g}_s${k}r$r prompt512.txt; done; done
```
7.3 **Level-2 profile, G=64, one per variant (A DQ DQR):**
`cool; run $v 64 prof_$v prompt512.txt NNTR_HTP_PROFILE=2`. Record the
`K=2048 N=2048 M==1` row (`dsp=`, `quant`, `mm`, `requant`, `rest`, `feed=…
dmaq=`) and `weight DMA: … first 3584 KB took X us`. The expected signature:
DQ has `requant` ≈ 12, `rest` < A's, and first-chunk µs < A's (a remainder).
DQR has `requant` ≈ 0. A `rest` ≈ A's on DQ means the fallback ran (fewer
than 4 workers), and it is recorded as such (§5).

7.4 **Bits:**
```
for v in A DQ DQR; do adb -s $S shell "rm -rf $DD/$v && mkdir -p $DD/$v"; cool; run $v 64 dump_$v prompt512.txt NNTR_HTP_DUMP=$DD/$v
  adb -s $S pull $DD/$v $W/dump/$v >/dev/null && adb -s $S shell "rm -rf $DD/$v"; done
python3 tools/htp/htp_dump_eval.py --label A_vs_177 $P/dump/Q4 $W/dump/A | tail -1          # bit_identical=1
for v in DQ DQR; do python3 tools/htp/htp_dump_eval.py --label $v $W/dump/A $W/dump/$v | tail -1; done   # G1
adb -s $S shell "rm -f $DD/cont.ids"; cool; run A 512 ppl_A prompt512.txt NNTR_PPL_DECODE=$DD/cont.ids
for v in DQ DQR; do cool; run $v 512 ppl_$v prompt512.txt NNTR_PPL_DECODE=$DD/cont.ids; done
nll() { grep -o '\[PPL\] decode step=.*' "$1"; }
cmp <(nll $W/logs/ppl_A.log) <(nll $P/logs/ppl_Q4.log) && echo "nll A == 177 Q4"
for v in DQ DQR; do cmp <(nll $W/logs/ppl_$v.log) <(nll $W/logs/ppl_A.log) && echo "G2 nll $v == A"; done
```
7.5 **Text, 8 prompts, G=64:**
`P8="prompt512.txt bitset-02-code.txt … bitset-08-short.txt"; i=0; for p in $P8; do i=$((i+1)); for v in A DQ DQR; do cool; run $v 64 text_${v}_p0$i $p; done; done`.
Compare with 177's `strip` plus the model-path filter: DQ ≡ A, DQR ≡ A,
A ≡ `$P/logs/text_Q4_p0$i.log`.

7.6 **Checks (workstation):**
```
grep -L 'dspq: close calls=\([0-9]*\) served=\1 bad=0' $W/logs/*.log      # nothing
grep -h 'moe m1 gemv' $W/logs/*.log | sort | uniq -c                         # one banner kind (dma_q=4)
grep -l 'MD5 FAIL\|dspq: off\|moe m1 gemv: off\|HTP arena: cannot\|EXIT=[1-9]' $W/logs/*.log   # nothing
grep -h -E '^(prefill|generation)' $W/logs/tps_*.log                         # into the G4/G5 tables with the .therm pairs
```
Fill the measurement doc: commit shas, md5s, serial, therm before/after
every run, cool-down waits, the per-run table with a "vs S25 prefill /
decode" column, the means table (min..max), the profile table, and the bit
and text tables. Commit it on the work branch, open the PR into
`htp_moe_v81` (stacked on #181, body naming the shipped variant per §1's
ship rule), and set `state:review`.

**Clean-up:** remove `$D` and `$DD`. `models/` is never written.

## 5. Risks (host vs device) and how the table shows them

| risk | how it shows up | handling |
|---|---|---|
| The in-app DMA rate differs from 62 GB/s (DVFS, fabric sharing with HVX VTCM reads). If the downs are faster than a C run, D/R gain less | 7.3 first-chunk µs, `mm`, `dsp` per variant | the verdict is `dsp` and tok/s. The model's numbers are stated as estimates (§0) |
| The v81 pool has fewer than 4 workers, so C2 falls back to today's GU(0) run | DQ `rest` ≈ A's in 7.3, and `quant` still outside it | correct by construction (the fallback is today's code). Recorded. If it happens, the ≈ 7 µs is lost and the lane count goes into LEDGER via a FARF diagnostic skel (plan 168 §5 recipe), not in this sitting |
| Lane n−1 of an A run is slower by one requant (≈ 10 µs). If A compute + requant exceeds a gate_up's DMA, the A runs lengthen | DQR `mm` > DQ `mm` + DQ `requant` | DQ is the in-sitting control. The ship rule (§1) drops C3 |
| Thermal drift and DVFS within the sitting (177: zone0 45 → 62–72 °C per run; prefill outliers of −20 %) | `.therm` pairs, `cool.log` | mirrored order per G, cool to ≤ 45 °C (max 5 min, logged), G=512 twice (plus a re-run round for G5), warm-up at G=8 |
| Cross-sitting / cross-device comparison with the S25 | the "vs S25" column | reported as the contract asks (2026-09-29): no scaling. G4 also requires DQR > A's max in-sitting, so a "win" that is only S26 drift is visible |
| Stale or wrong skel. All three banners are identical, so the echo cannot catch a swap | `md5sum -c` before every run; the 7.3 signature (`requant` ≈ 0 only on DQR) | a run with `MD5 FAIL` is void (`run()` exits 9) |
| Host lanes run one after another, so a same-run cross-lane race computes the right bytes on the host | only the dataflow and DMA scoreboards can catch it | steps 2–4 negatives prove each check fires. G1–G3 check on silicon |
| Address-space / arena budget | none | no new memory. The same 7 MiB of slabs and 1 KiB of descriptors |
| The phone is shared (another measurement ran during planning) | `pidof` before step 7 | no step touches the phone before the orchestrator releases it |

## 6. Docs to update (supervisor, from the filled measurement doc)

* `docs/htp_moe/BENCHMARK.md`, the S26 Ultra (v81) block: rows A / DQ / DQR
  × G 64 / 512 / 1024 (prefill, decode all, last 64, therm pre/post) tagged
  `R5KL20NFRCK`, with the "vs S25 (#158 B)" column and the M==1 `dsp` column.
  If the shipped variant passes G4, it becomes the S26 "now" row with
  `NNTR_MOE_DMA_QUEUES=4` named. The default flip of #177 is still the
  user's call.
* `docs/htp_moe/LEDGER.md`:
  - A rule: at 4 queues the decode MoE call is DMA-bound (21.5 MB ÷ in-app
    rate ≈ 76 % of `dsp`). GU(0) is transfer time, not schedule. The
    recoverable µs are the DMA-idle pieces (§0), with the measured split
    from 7.3.
  - The M==1 row's field meanings under N > 1 after this PR: `requant` = the
    B run only (0 at even n), `rest` / first chunk = GU(0)'s remainder after
    QUANT.
  - Open items: C(2)/C(3) DMA-idle (≈ 38 µs, third down slot or a row-split
    down); more than 4 queues (the probe tested 1 / 2 / 4. The flags
    field, a 2-bit mask at `hexkl_mm_u8i4_moe.h:366-381`, and `g_m1q_desc[4]` stop at 4); the S26 pool's worker count if the C2 fallback
    fired; the "blocked issuer" question (plan 177 §3) stays open.
