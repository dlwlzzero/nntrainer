# 177 — Split the decode MoE weight DMA over 2–4 queues on the S26 Ultra (v81)

Issue: dlwlzzero/nntrainer#177 (p0). Read against `htp_moe_v81` @ `82acfb16`.
PR #173 (#168's arch guard) is still open. If it has merged by the time work
starts, rebase onto it; if not, check the arch by hand with `hexagon-readelf -h`
(step 5). Work branch **`htp/177-m1-dma-queues`**, PR into **`htp_moe_v81`**.
Device: `adb -s R5KL20NFRCK` only (contract §12, 2026-09-29 row). Never touch
`R3CY10WM83Y` or `R3CN80CW3FY`.

## 0. What the planner found in the code (answer to the issue's first question)

**Multi-queue is not reachable from the M==1 feed today. No env var or knob
turns it on.**

* Every DMA queue belongs to the thread that issued the `dmstart`
  (`hexkl_dma_ring.h:51-55`, "Starts this thread's DMA engine").
  `hexkl_dma_ring.c` is one static chain (`g_ring[256]`, `:29-32`) driven by
  one thread: `push2d` either `dmstart`s or `dmlink`s onto `g_tail` (`:82-88`).
* The M==1 feed issues every weight push from the caller on that ring. Each
  push is one whole-matrix descriptor, `moe_m1_push`
  (`hexkl_mm_u8i4_moe.c:855-872`). The gate_up descriptor is 57 344 B × 64 rows
  = 3.5 MiB, and the down descriptor is 65 536 B × 56 rows. The feed waits on
  the ring with `moe_m1_wait` (`:874-882`). Schedule: GU(0), GU(1) before the
  scan (`:1333-1340`); stage A per expert, with wait → pool run → push GU(i+2)
  or D(2s), D(2s+1) (`:1440-1464`); stage C per expert, with wait → pool run
  (`:1478-1490`).
* #158's multi-worker support (`NNTR_MOE_DMA_SPLIT_ROWS`,
  `nntr_moe_dma_plan.h:261-266`) exists only in the skel's **replay** entry.
  `replay_worker` (`nntr_hvx_dma_probe.c:314-398`) gives every pool thread its
  own chain, `REPLAY_DESC(k, i)`, and a row slice
  (`replay_rows`, `:229-238`). The kernel never includes it.
* The flags word has no queue field (`hexkl_mm_u8i4_moe.h:236-360`,
  `HEXKL_MOE_FLAGS_KNOWN` ends at bit 18).

**One fact changes how the issue's evidence reads.** The 94.2 / 73.8 GB/s
figures come from `DmaProbeShapes`, which reads with **`src_bypass = 0`**
(`nntr_hvx_dma_probe.c:78`), re-reads the same source every pass, and uses
8 KiB columns. The production feed reads with `src_bypass = 1` (the default
since #158), always from fresh sources, as whole matrices. On the S25 the
production-shaped probe (`DmaSettings`: f2, fresh, tag-checked, bypass 0/1 ×
1/2/4 queues) found that "more UDMA queues … add nothing" once bypass was on
(LEDGER rule 43). Nobody has run `DmaSettings` on the S26. It is the free
go/no-go below (step 1).

## 1. Goal and gate

Goal (issue): on `R5KL20NFRCK`, issue the M==1 feed's weight DMA on N = 2–4
queues without changing any bit, and measure it as an A/B full-model run. With
the knob unset, N = 1, and that path is today's code unchanged.

| # | check | pass |
|---|---|---|
| G0 | go/no-go (step 1) | `DMA_SETTINGS_MEAN src_bypass=1 queues=2` or `queues=4` is ≥ +10 % (`NNTR_DMA_SETTINGS_MIN_GAIN`) above `src_bypass=1 queues=1` in the same run, with `checksum_ok=y`. Otherwise **stop before any code**: comment the table on #177 and report |
| G1 | bits, default unchanged | A's MoE dumps are `bit_identical=1` against #168's v81 dump (`/local/mnt/workspace/htp_moe/168/dump/v81`). A's decode `[PPL] decode step=` lines are byte-identical to `168/logs/ppl_npu.log`. A's text ≡ `168/logs/text_npu_p0{1..8}.log` |
| G2 | bits, N > 1 | Q2 and Q4 MoE dumps are `bit_identical=1` against A (`tools/htp/htp_dump_eval.py`). Their nll lines, forced on A's `cont.ids`, are byte-identical to A's. Their text ≡ A on the 8-prompt set (G=64) |
| G3 | the knob really ran | the banner reads `dma_bypass=1 dma_q=N` with `applied=0x703e1` (A), `0xf03e1` (Q2), `0x1f03e1` (Q4). The level-2 `M==1` row reads `feed=C/C dmaq=N.00`. Every run's `dspq: close calls=served=22·G bad=0`. The device `md5sum -c` of the variant directory passes before every run |
| G4 | speed (the A/B) | decode tok/s at G=512 from mirrored runs (A Q2 Q4 Q4 Q2 A): report Q/A in %. G=64 and G=1024 are read the same way. The level-2 profile per variant gives the M==1 `dsp=` µs/call, `mm`, and `weight DMA: first 3584 KB took … GB/s` |
| G5 | prefill (standing) | prefill tok/s of Q2 and Q4 ≥ −5 % of A, on the means of the same mirrored runs. The M>1 path does not read the new bits, so a miss here is thermal and must be re-run |
| G6 | text vs CPU (standing, amended) | on `htp_moe_v81` the accuracy gate is bit-identity to the reference (contract §12, 2026-09-29), which G1 and G2 cover. Text vs CPU `q40` is recorded as information only (#168 G5) |

The default flip is **not** part of this PR. The PR ships N = 1. If G4 shows
that one Q beats A outside A's own spread at G=512 (Q's mean > A's max) and is
no worse at 64 or 1024, then the PR proposes flipping the default in a
separate one-line commit on `HTP_MOE_DMA_QUEUES_DEFAULT` and the user decides.
That is the #158 precedent: bit-identical, and the flip changes only the unset
value.

## 2. Where it lives

| file:line | change |
|---|---|
| `nntrainer/tensor/htp_backend/hmx/hexkl_dma_ring.h` (after `:120`), `hexkl_dma_ring.c` | **new, ~25 lines**: `hexkl_dma_lane_push2d(hexkl_dma_desc2d *d, hexkl_dma_desc2d *prev, dst, src, dst_stride, src_stride, row_size, nrows, src_bypass, dst_vtcm)` fills the descriptor as `push2d` does (`:59-81`), runs `dccleaninva`, and then `dmstart`s if `prev == NULL`, else `dmlink`s onto `prev`, **on the calling thread's own queue**. `hexkl_dma_lane_wait(hexkl_dma_desc2d *d)` spins on `d->done` with `dmpoll` (the ring's guard). The ring itself does not change |
| `hexkl_mm_u8i4_moe.h:329-360` | new field: bits [20:19] of the flags word = N − 1 (`HEXKL_MOE_DMA_Q_SHIFT 19`, `HEXKL_MOE_DMA_Q_BITS 3`), added to `HEXKL_MOE_FLAGS_KNOWN`; `hexkl_moe_flags_dma_q(flags)` returns 1..4. A skel without the field masks it off, so the ARM-side echo check throws (LEDGER rule 21) |
| `hexkl_mm_u8i4_moe.c:616-632` (`moe_m1_ctx`) and new helpers near `:833-882` | `moe_m1_qrun`: {pending pushes (≤ 2: dst, src, row, nrows, bypass), nq, inner func/ctx, nq_used}. `moe_m1_q_worker(n, i, v)`: when `i < min(nq, n)`, issue row slice `[nrows·i/nq, nrows·(i+1)/nq)` of each pending push on this lane's chain (`static hexkl_dma_desc2d g_m1q_desc[4][2]`, aligned(128), 1 KiB). Then run the inner stage worker (or nothing). Then `hexkl_dma_lane_wait` on the lane's last descriptor. The slicing is `replay_rows`' rule |
| `hexkl_mm_u8i4_moe.c:1119-1127, 1327-1341, 1428-1490` | the N > 1 branch of the feed (§3): GU(0) in its own N-unit run before the scan, timed as `DMA_FIRST`; each later push rides the first pool run that starts after the join freeing its slab. The N = 1 branch is today's lines, untouched. `DMA_KB` is counted by the caller when it posts a push. Lanes never touch `hexkl_probe_us` |
| `nntrainer/tensor/htp_backend/hmx/hexkl_probe.h:146`, `test/htp/nntr_hvx_mm_u8i4.c:909,940` | `HEXKL_PROBE_M1_FEED` / `MOE_T_M1_FEED` changes value from "1 = feed" to "queues the feed used, 0 = arena read". N = 1 still writes 1. Same slot, same stage count, **no IDL change** |
| `nntrainer/tensor/htp_backend/htp_moe_opts.h` | `HTP_MOE_DMA_Q_SHIFT/BITS`, `HTP_MOE_DMA_QUEUES_DEFAULT 1u`, `htp_moe_opts_dma_queues(const char *env, uint32_t *bits)`: unset gives the default; `1`..`4` gives (N − 1) << 19; anything else returns an error. This is a trust boundary: a mistyped variant must not run as A |
| `nntrainer/tensor/htp_backend/htp_compute_ops.cpp:1760-1810` (`sendMoeOptsOnce`) | read `NNTR_MOE_DMA_QUEUES` and OR in the bits; an invalid value throws `std::runtime_error`. The banner gains `dma_q=%u` (taken from `applied`) between `dma_bypass=` and `source=` |
| `htp_compute_ops.cpp:188, 385, 476-481, 685-692` | bucket `m1_feed_q_sum += stage_us[HTP_MOE_T_M1_FEED]`; `m1_feed_calls` still counts values ≠ 0. The M==1 row prints `feed=C/C dmaq=%.2f` (the sum divided by the feed calls) |
| `test/htp/host/moe_layer_host_check.c:52-130, 240-320, 562-722, 845-870` | stand-ins for the two lane functions. The scoreboard gains slices: a push is complete when all of its slices are waited. A slice must be **waited by the lane that issued it, inside the same pool run** (a run counter bumped by the pool stand-in). That is the design's safety invariant (§3), checked. The flags loop adds N ∈ {2, 3, 4} to the feed cells; outputs must be byte-equal to the N = 1 run of the same cell, and `fed == nq_used`. New pass line `M1 FEED QUEUES OK (n=2,3,4)` |
| `test/htp/host/hvx_scalar_stubs.c:29-45` | no-op lane stand-ins (the conv-block check links the MoE file, `run_host_checks.sh:51-60`) |
| `test/htp/host/moe_opts_host_check.c` | cases: unset → `0x703e1` unchanged; `NNTR_MOE_DMA_QUEUES=2` → `0xf03e1`; `4` → `0x1f03e1`; `0`, `5`, `x` → error |
| `test/htp/host/replay_stub/hexkl_dma_standin.h` | nothing: the in-process E2E compiles the real `hexkl_dma_ring.c` over this stand-in (`inproc/hexagon_protos.h:20`), so the lane functions run there as they are |

Consumers checked that do **not** move: the IDL `test/htp/nntr_hvx.idl`
(`moe_set_opts(in uint32 flags, rout uint32 applied)`, `:478`, is already a
uint32), so there is no `generate_stub.sh` regeneration beyond what `build.sh`
always does. Also unchanged: the quantizer's `QS4CX_WH` format tag and the
loader check (no byte layout changes), `tools/htp_fc_report.py` (it does not
parse the M==1 row or the banner), the M>1 HMX path (it never reads the new
bits), `nntr_hvx_graph.c:117` (it passes the session's flags through), and
the dspq transport (it carries `stage_us` unchanged).

## 3. Design

**Chosen: each issuing lane polls its own slices before its unit returns.**
When N > 1, every weight matrix is split by rows into `nq = min(N, n_threads)`
contiguous slices. Slice `i` is issued by pool lane `i` on its own queue
(lane 0 is the caller). The DMA is folded into a pool run:

```
run GU0   : lanes <nq issue GU(0) slices, poll                (timed as DMA_FIRST)
scan, pack (unchanged)
run A(0)  : issue GU(1) → slab 1 | compute A(0) from slab 0 | poll
run A(1)  : issue GU(2) → slab 0 | compute A(1) from slab 1 | poll
run A(2)  : issue GU(3) → slab 1 | compute A(2)             | poll
run A(3)  : issue D(0), D(1) → slab 0 | compute A(3)        | poll
run B     : issue D(2), D(3) → slab 1 | requant             | poll
run C(0..3): compute (D(j + 3) → slot (j − 1) & 3 when n_active > 4)
```

The rule is that a push rides the first run that starts after the join freeing
its destination. That is the same slab ownership `moe_m1_gu_off` / `moe_m1_dn_off`
(`:837-850`) already proves (`FEED push onto a live slab`). The consumer waits
through the **pool join**: a matrix is read only by a run that starts after
the run whose lanes polled every one of its slices done. The GEMV units, the
lane slicing (`moe_m1_slice`), the requantization and the caller's scatter
order do not change, and each VTCM byte has exactly one writer per push. So
the int32 sums and the f32 output are the same bytes as for N = 1 (G2). Each
matrix's DMA still runs behind the previous stage's compute (doc 45 §3.2).
Only GU(0) is exposed. Today it hides behind the scan and pack, which take
≈ 7 µs at M=1 (`quant 6.8`, #168 profile), so the loss is negligible.
Expected, not promised: at 4 queues the call is bound by 21.5 MB ÷ the in-app
N-queue rate, plus the C tail and ≈ 9 fork/joins. If the S26 in-app rate is
0.82 × the probe (the S25 ratio, rule 43), that is ≈ 350 µs `dsp` against
593 today.

Contract checks: nothing on the M>1 path changes, and neither do the arena
budget or the slabs (7 MiB of VTCM, as today). The only new memory is 1 KiB
of static descriptors, no heap, no address-space growth. `QS4CX_WH` keeps no
CPU fallback. No quantizer input changes, so no `_det` spec is needed.
Bit-identity is gated on the host (step 3) and on silicon (G1, G2).

**Rejected: lanes `dmstart` their slices and return, and the caller polls
every slice's done bit at today's wait points.** This is the smallest diff:
today's schedule verbatim, with one ~µs issue run per push. It was rejected
because it relies on a queue continuing to run after its issuing thread blocks
on the pool futex, and nothing we have measured shows that. Every multi-queue
reading so far (`dma_probe_worker` `:94-114`, `replay_worker`) polls on the
issuing thread. The ISA has `dmpause` / `dmresume` for context switches. If
QuRT pauses a blocked thread's queue, the caller's spin runs out on the
50 M-poll guard (`hexkl_dma_ring.c:34-39`) and falls through with stale VTCM,
which produces plausible wrong text and no error. It could come back later as
a measured follow-up (a replay flag that detaches the issuer), but not as the
first step. Also rejected: long-lived DMA lanes as a background job
(`hvx_worker_pool_submit_bg`). The pool's contract is "nothing spins inside a
unit" (`hvx_worker_pool.h:91`), and a spinning unit holds a worker that the
next foreground run waits for.

`ponytail:` D(2) and D(3) are polled inside run B, so C(0) cannot overlap them
the way today's single chain does. The cost is about one C(j) compute. If the
profile shows B's wall ≫ its requant, the fix is to move D(3) into C(0)'s run.

## 4. Steps

Every shell starts with:
```
cd /home/j2z0-lee/nntrainer && git checkout -b htp/177-m1-dma-queues origin/htp_moe_v81
export HEX_ARCH=v81 && source tools/htp/env.sh && export HEXKL_ROOT HEXKL_SDK_VER
S=R5KL20NFRCK; W=/local/mnt/workspace/htp_moe/177; R=/local/mnt/workspace/htp_moe/168
D=/data/local/tmp/nntrainer/causallm/s177; MR=/data/local/tmp/nntrainer/causallm/models; DD=/data/local/tmp/s177dump
therm() { adb -s $S shell "echo \$(date +%T) bat=\$(dumpsys battery | sed -n 's/^  temperature: //p') zone0=\$(cat /sys/class/thermal/thermal_zone0/temp)"; }
```

**Step 1 — device go/no-go, before any code (≈ 5 min; agent-run, G0).**
Use the tree as it is: `htp_moe_v81`'s v81 skel (`$R/libnntr_hvx_skel.so`,
already on disk; `hexagon-readelf -h` must show `Flags: 0x81`) and
`test/jni/obj/local/arm64-v8a/unittest_hvx_dma_probe`. Rebuild the probe with
rung 3's `ndk-build … unittest_hvx_dma_probe` if its sources are newer.
```
mkdir -p $W/probe && cp $R/libnntr_hvx_skel.so test/jni/obj/local/arm64-v8a/unittest_hvx_dma_probe $W/probe/
(cd $W/probe && md5sum libnntr_hvx_skel.so unittest_hvx_dma_probe > md5.txt)
adb -s $S shell "mkdir -p $D/probe" && adb -s $S push $W/probe/* $D/probe/
for r in 1 2; do therm | tee -a $W/probe/therm.log
  adb -s $S shell "cd $D/probe && md5sum -c md5.txt && chmod 755 unittest_hvx_dma_probe && \
    LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_dma_probe --gtest_filter='*DmaSettings*'" 2>&1 | tee $W/probe/settings_r$r.log | grep -E 'DMA_SETTINGS_(MEAN|DECISION)|OK$|FAILED'
done; therm | tee -a $W/probe/therm.log
```
Read `src_bypass=1 queues=1/2/4` from both runs. If G0 fails, comment the
table on #177 and stop (label `needs-user`). `valid=n INVALID` with
`checksum_ok=y` on a cell above 85.3 GB/s is the S25's DDR ceiling constant
(§5), not a failure by itself. Record it.

**Step 2 — ARM side: knob, echo, banner, row.** `htp_moe_opts.h`,
`sendMoeOptsOnce`, bucket / row print, `moe_opts_host_check.c`.
Gate, rung 0 and rung 1 subset: `ninja -C build`,
`bash test/htp/host/run_host_checks.sh` → `ALL CHECKS PASS`, with the new
`MOE M1 GEMV OPTS` lines for 1/2/4/invalid.

**Step 3 — DSP side: the lane functions, the flags field, the N > 1 feed
branch, and the host scoreboard.** Gate rung 1 in full: `run_host_checks.sh` →
`ALL CHECKS PASS`, `WORKER POOL LANES OK`, `M1 GEMV VTCM FEED SCHEDULE OK`
(N = 1, unchanged), and the new `M1 FEED QUEUES OK (n=2,3,4)`, whose
f32 output must be byte-equal to N = 1 on every feed cell (tiny and real shapes,
3 and 5 experts, bypass on and off). `tools/htp_syntax_check.sh` exits 0.
`run_inproc_e2e.sh` prints all of its pass lines twice: unset, and then
with `NNTR_MOE_DMA_QUEUES=4` exported (`E2E eval golden … bit_identical=1`,
`E2E tokens htp==cpu 8/8`, `INPROC E2E PASS`). Commit the kernel/app work
separately from the docs (contract §5).

**Step 4 — the negative check once, pasted in the PR.** Put a lane's wait
after the run (so the "same lane, same run" invariant is broken) and show that
the scoreboard fails. Then revert.

**Step 5 — skel (rung 2).** `HEX_ARCH=v81 ./test/htp/build.sh` →
`UNDEFINED SYMBOLS OK`, plus `ARCH OK (V81)` if #173 is merged, or else
`$HEXAGON_TOOLS_ROOT/bin/hexagon-readelf -h test/htp/build/libnntr_hvx_skel.so | grep 'Flags:.*0x81'`.
Record the md5. Build v79 once (`HEX_ARCH=v79`) to show that `htp_moe`'s arch
still compiles, then rebuild v81 last.

**Step 6 — app (rung 3).** `(cd Applications/CausalLM && ./build_android.sh --htp --cache)`
with the `NEEDED` / `strings` / md5 checks. `strings libcausallm_core.so`
(or `libnntrainer.so`) must contain `NNTR_MOE_DMA_QUEUES`.

**Step 7 — the device A/B (the unavoidable silicon step, agent-run on
`R5KL20NFRCK` under the contract's exception; ≈ 60 min including cool-downs).**
Record it as `docs/measurements/177-m1-dma-queues.md` in the handoff skill's
shape. Three variants, **one binary set**, one full directory copy each:

| variant | `$D/<v>/variant.env` | expected banner |
|---|---|---|
| A (reference, N = 1 = today's code path) | *(empty)* | `applied=0x703e1 … dma_bypass=1 dma_q=1 source=default` |
| Q2 | `export NNTR_MOE_DMA_QUEUES=2` | `applied=0xf03e1 … dma_q=2` |
| Q4 | `export NNTR_MOE_DMA_QUEUES=4` | `applied=0x1f03e1 … dma_q=4` |

Stage `$W/set/`: the step 5 skel, the step 6 binaries (`nntrainer_causallm`,
`libcausallm_core.so`, `libnntrainer.so`, `libccapi-nntrainer.so`,
`libc++_shared.so`, `libsdkl.so`), `prompt512.txt`, `bitset-0{2..8}-*.txt`
(copy from `$R`), and `md5.txt` over all of them. Never stage `libcdsprpc*`.
```
for v in A Q2 Q4; do mkdir -p $W/$v && cp $W/set/* $W/$v/; : > $W/$v/variant.env; done
echo 'export NNTR_MOE_DMA_QUEUES=2' > $W/Q2/variant.env; echo 'export NNTR_MOE_DMA_QUEUES=4' > $W/Q4/variant.env
adb -s $S devices -l; adb -s $S shell df -h /data; therm        # ≥ 1 GB free; record every serial
adb -s $S shell "md5sum $MR/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin"   # 7b7867fa…; greedy + bad_word_ids as #168 left them:
adb -s $S shell "grep -H do_sample $MR/q40-qs4cx-wh/generation_config.json; grep -H -E 'bad_word_ids|_engine' $MR/q40-qs4cx-wh/nntr_config.json"
for v in A Q2 Q4; do adb -s $S shell "rm -rf $D/$v && mkdir -p $D/$v" && adb -s $S push $W/$v/. $D/$v/ >/dev/null
  adb -s $S shell "chmod 755 $D/$v/nntrainer_causallm; cd $D/$v && md5sum -c md5.txt" | tee $W/$v/md5_device.log; done
cool() { for k in $(seq 30); do t=$(adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'); [ "$t" -le 45000 ] && break; sleep 10; done; echo "cool zone0=$t waited=$((k*10))s"; }
run() { # run <variant> <G> <log> <prompt> [extra env]
  L=$W/logs; mkdir -p $L; echo "pre $(therm)" > $L/$3.therm; adb -s $S logcat -c
  adb -s $S shell "cd $D/$1 && md5sum -c md5.txt >/dev/null || { echo MD5 FAIL; exit 9; } && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' $MR/q40-qs4cx-wh/nntr_config.json && \
    . ./variant.env && NNTR_NUM_THREADS=8 $5 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $MR/q40-qs4cx-wh \"\$(cat $4)\"; echo EXIT=\$?" 2>&1 | tee $L/$3.log \
    | grep -E '^(prefill|generation)|moe m1 gemv|dspq: close|MD5 FAIL|EXIT='
  echo "post $(therm)" >> $L/$3.therm; adb -s $S logcat -d -v time > $L/$3.logcat
}
```
The per-run checks are in 7.6. Any `MD5 FAIL`, a missing `dspq: close … bad=0`,
or a wrong `dma_q=` voids that run.

7.1 **Sanity, G=8:** `for v in A Q2 Q4; do run $v 8 sanity_$v prompt512.txt; done`
→ the three banners from the table, `calls=176 served=176 bad=0`.

7.2 **tok/s, prompt 512, mirrored:** G=512 twice (both rounds 6 runs), then
G=64 and G=1024 once each:
```
for g in 512 512 64 1024; do k=$((k+1)); r=0; for v in A Q2 Q4 Q4 Q2 A; do r=$((r+1)); cool; run $v $g tps_${v}_G${g}_s${k}r$r prompt512.txt; done; done
```
Every G runs in the order A Q2 Q4 Q4 Q2 A. G=512 has two such rounds, so it
gets four runs per variant.

7.3 **Level-2 profile, G=64, one per variant (mirrored A Q2 Q4):**
`run $v 64 prof_$v prompt512.txt NNTR_HTP_PROFILE=2`. Record the `K=2048 N=2048
M==1` row (`dsp=`, `mm`, `requant`, `feed=1408/1408 dmaq=`) and its `weight
DMA: … first 3584 KB took X us = Y GB/s`. Under N > 1 the `DMA ring:` line
covers only the two copy descriptors, and its `engine` / `first expert ready`
figures are not comparable with A's (§5).

7.4 **Bits:**
```
for v in A Q2 Q4; do adb -s $S shell "rm -rf $DD/$v && mkdir -p $DD/$v"; run $v 64 dump_$v prompt512.txt NNTR_HTP_DUMP=$DD/$v
  adb -s $S pull $DD/$v $W/dump/$v >/dev/null && adb -s $S shell "rm -rf $DD/$v"; done
python3 tools/htp/htp_dump_eval.py --label A_vs_168 $R/dump/v81 $W/dump/A | tail -1     # G1 bit_identical=1
for v in Q2 Q4; do python3 tools/htp/htp_dump_eval.py --label $v $W/dump/A $W/dump/$v | tail -1; done   # G2
adb -s $S shell "rm -f $DD/cont.ids"; run A 512 ppl_A prompt512.txt NNTR_PPL_DECODE=$DD/cont.ids
for v in Q2 Q4; do run $v 512 ppl_$v prompt512.txt NNTR_PPL_DECODE=$DD/cont.ids; done
nll() { grep -o '\[PPL\] decode step=.*' "$1"; }
cmp <(nll $W/logs/ppl_A.log) <(nll $R/logs/ppl_npu.log) && echo "G1 nll A == 168"
for v in Q2 Q4; do cmp <(nll $W/logs/ppl_$v.log) <(nll $W/logs/ppl_A.log) && echo "G2 nll $v == A"; done
```
7.5 **Text, 8 prompts, G=64, per variant:**
`P="prompt512.txt bitset-02-code.txt … bitset-08-short.txt"; i=0; for p in $P; do i=$((i+1)); for v in A Q2 Q4; do run $v 64 text_${v}_p0$i $p; done; done`.
Compare with #168's `strip` (`docs/plans/168-s26-v81-bringup.md` §4 step 5):
Q2 ≡ A, Q4 ≡ A, and A ≡ `$R/logs/text_npu_p0$i.log`.

7.6 **Checks (workstation):**
```
grep -L 'dspq: close calls=\([0-9]*\) served=\1 bad=0' $W/logs/*.log          # nothing
for v in A Q2 Q4; do grep -h 'moe m1 gemv' $W/logs/*_$v*.log | sort | uniq -c; done   # one banner kind per variant, as in the table
grep -l 'MD5 FAIL\|dspq: off\|moe m1 gemv: off\|HTP arena: cannot\|EXIT=[1-9]' $W/logs/*.log   # nothing
grep -h -E '^(prefill|generation)' $W/logs/tps_*.log                           # into the G4/G5 table with the .therm pairs
```
Fill the measurement doc (commit, md5s, serial, therm before/after every run,
G0–G5 tables, and the texts beside A's), commit it on the work branch, open
the PR into `htp_moe_v81`, set `state:review`. If G4 passes, propose the flip
commit for the user (§1).

**Clean-up:** remove `$D/{A,Q2,Q4,probe}` and `$DD`. Keep `$MR/q40*`. Touch
nothing outside `s177`, `s177dump`.

## 5. Risks (host vs device) and how the table shows them

| risk | how it shows up | handling |
|---|---|---|
| The S25 result repeats on the S26: with `src_bypass=1`, more queues add nothing | step 1's `DMA_SETTINGS_MEAN` | G0 stops the work before any code is written |
| In-app rate below the probe: HVX VTCM reads and N engines share the fabric (S25 in-app ≈ 0.82 × probe) | 7.3 first-chunk GB/s vs step 1 | recorded as the ratio; the verdict is `dsp=` and tok/s, not the probe |
| A queue pauses while its thread blocks (untested) | would be wrong bits, not slowness | the design never lets that happen; host step 3 checks "same lane, same run"; G2 checks on silicon |
| The v81 pool has fewer lanes than N | `dmaq=` < N in the M==1 row | the variant is read at the `dmaq` it really had |
| Fork/join cost: ≈ 9 runs, each now ending on a DMA poll | `mm` vs bytes ÷ first-chunk rate | ponytail in §3; if the gap is > 50 µs/call, the follow-up is to issue two matrices per run |
| First-chunk semantics differ: A's `DMA_FIRST` times the exposed remainder after the scan; Q's times the whole GU(0) transfer | A's GB/s reads high | diagnostic only; G4 reads `dsp=` and tok/s |
| DVFS and thermal drift within and between sittings (#168: 30.8* outlier; zone0 reached 68 °C) | the `.therm` pairs per run | mirrored order per G, cool-down to zone0 ≤ 45 °C (max 5 min, time logged), no cross-sitting comparison except G1's bit checks |
| Stale or foreign skel (`ADSP_LIBRARY_PATH=.` loads another one silently if the file is missing) | `md5sum -c` before every run; Q2/Q4 echo (an old skel masks bits 19–20 → the app throws) | A cannot be caught by the echo, which is why md5 runs before every run |
| DDR ceiling constant `NNTR_DDR_CEILING_GBS 85.3` is the S25's | step 1 prints `INVALID` above 85.3 | recorded; SM8850's ceiling is an open LEDGER item, not decided here |
| Address-space budget | none | 1 KiB of static descriptors; no heap, no arena change |

## 6. Docs to update (supervisor, from the filled measurement doc)

* `docs/htp_moe/BENCHMARK.md`, in the S26 Ultra (v81) block that #168 opens:
  rows A / Q2 / Q4 × G 64 / 512 / 1024 (prefill, decode all, last 64,
  therm pre/post), tagged `R5KL20NFRCK`, plus the M==1 `dsp` / first-chunk
  column. Read only as an A/B inside this sitting.
* `docs/htp_moe/LEDGER.md`: a rule on what multi-queue does on v81 (the step 1
  `DmaSettings` cells next to the S25's rule-43 cells, and the in-app `dsp`
  delta). The `dmaq` / `dma_q` columns and the `M1_FEED` slot meaning. An
  open item for the untested "blocked issuer" question (rejected alternative,
  §3). An open item for SM8850's DDR ceiling. If the default flips: the
  "now" row for v81 and the unset banner `dma_q=N`.
