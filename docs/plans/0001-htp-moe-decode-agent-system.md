# 0001 — Agent system and contract for the HTP MoE decode project

Status: agreed 2026-09-21 (grill-me session, 4 rounds after the PR head
moved). This is the ground truth for how the `htp_moe` work is organised.
Every agent definition under `.claude/` links here instead of restating it.
It supersedes `docs/plans/0000-agent-system-and-env.md` on the `hvx_impl`
branch, which stays there, frozen, as the record of the Qwen3 HVX line.

## 1. Goal

Make LFM2.5-8B-A1B **decode** on the Hexagon HTP of a Galaxy S25 Ultra
(Snapdragon 8 Elite, v79) at least **2× faster than the upstream PR runs it
today, and faster than the same phone's CPU**.

| item | value | source |
|---|---|---|
| Model | LFM2.5-8B-A1B: 24 layers = 18 conv + 6 attention, 22 MoE FFN layers (32 experts, top-4), hidden 2048, vocab 128000, tied lm_head | `config.json` of `LiquidAI/LFM2.5-8B-A1B` |
| "Now", NPU (MoE FFN on HTP, `htp_moe` @ `2a75f7d9`, switch off) | decode **17.72 / 17.03 / 17.30 tok/s** at gen 64 / 512 / 1024, prefill 389–527 tok/s (prompt 512) | #94 sitting 2, `R3CY205ZMND`, 2026-09-22 (`dad0f476`); replaces the PR author's 20.8 / 523 (doc 49 §1, prompt 444) |
| "Now", CPU (all Q4_0, 8 threads) | decode **52.43 / 49.22 / 48.31 tok/s**, prefill 268–340 tok/s | same sitting; replaces the PR author's 48 / 334 |
| First lever measured | M=1 HVX GEMV (PR #86, switch on): **18.83 / 18.33 / 17.59** (+6.2 / +7.7 / +1.7 %), text identical | same sitting, variant C (LEDGER ⑯) |
| **Goal** | decode **≥ 50 tok/s** at every measured generation length (2.9× the NPU "now" at gen 512, above the CPU) | user decision Q1 (ii) |
| Physical ceiling | 730 MB of weights per token ÷ 34–38 GB/s measured DDR rate = 19–21 ms ⇒ **48–52 tok/s**. The CPU already sits on it | doc 48 §1 |
| Prefill gate | never below **−5 % of the current NPU prefill** (≈ 523–532 tok/s, i.e. ≥ 497) | user decision Q12 |
| Accuracy gate | doc 45 §3.4, all three: (a) kernel bit-identical to its scalar spec, (b) real-model diff (`NNTR_L2_DIFF`) = 0, (c) generated text identical to the CPU run of the same weights | user decision Q4 |

The "now" numbers were replaced on 2026-09-22 (cycle 5) by #94 sitting 2,
the first sitting with all 12 control cells on one binary set (§4.3's
intent; #77 had measured the same cells a day earlier on the same unit:
NPU 18.2–21.3, CPU 46.4–54.1, prefill 403–541 — all in BENCHMARK.md).
They are read against the sitting they came from: one unit drifts up to
±9 % (CPU) / −16 % (NPU tok/s) between sittings while its DSP profile
columns move ≤ 4 % (LEDGER rules 9, 20), and two S25 Ultra units differed
by 6–8.6 % on one binary in `hvx_impl`. Hence every verdict is a
same-sitting A/B (§4.2) and no unit is named (decision 2026-09-22).

2× the CPU (96 tok/s) is **not** a goal: it needs 70 GB/s from a 38 GB/s
memory system. Anything that raises the ceiling itself (fewer bytes per
token, two DDR readers) is a separate track (§3.3).

### 1.1 Measurement definition

Prompt fixed at 512 tokens (the PR used 444; we fix 512 so the prefill
column is comparable across handoffs). Generation length **64 / 512 /
1024** tokens. Report, per run: prefill tok/s, decode tok/s as the median
over the whole generation and separately over the last 64 tokens, peak RSS,
and the accuracy columns. `NNTR_NUM_THREADS=8` for both CPU and NPU runs
(doc 49 §1 measured both that way). Never read tok/s from a `--profile`
build (it inflates prefill by 83 %, doc 46 §43.4).

## 2. Facts we build on (do not re-derive)

From the PR's own device work (`docs/htp_attention/44`–`50`):

* **Why NPU decode is slow — three walls, all three must fall** (doc 48 §3).
  Per MoE call at M=1: host 1.91 ms = DSP 1.35 + transport 0.57.
  1. HMX computes 64-row tiles for 1 row: 1.03 ms of the 1.35. Fix: an HVX
     GEMV for M ≤ 6 reading the WH tiles in VTCM directly (int32 sums are
     order-independent, so it can be gated bit-identical to the HMX path).
  2. Weight DMA runs at 18 GB/s where the arena probe reached 38. Cause
     unknown (descriptor shape / engine count / DVFS) — measurement C of
     doc 48 §5 decides.
  3. FastRPC transport 0.57 ms × 22 layers = 12.5 ms/token. Fix: dspqueue
     or a resident DSP worker; measurement B of doc 48 §5 decides which.
  Fixing all three gives MoE ≈ 15 ms/token; the CPU's MoE share is 19.
* **Then the ARM remainder** (FC, attention, norm, lm_head ≈ 19 ms/token,
  not yet measured — measurement A of doc 48 §5) must shrink to a few ms,
  which means one FastRPC call per token with the whole layer stack
  resident on the DSP (doc 45 Phase E), M=1 shape first (§3.1).
* **Moving an FC to the HTP is not a lever by itself**: conv in_proj on
  the HTP saved 13 ms of 848 in prefill because the round trip (4.9 ms)
  is larger than the HMX work (2.0) (doc 50 §3.4). Removing round trips is
  the lever.
* **Weights already live in DDR the DSP can see**: all 1408 expert
  weights (3.7 GiB, `QS4CX_WH` = per-channel int4 pre-baked into HMX tile
  order by the quantizer) sit in an ION arena of 256 MiB chunks, 3840 MiB
  mapped, 0.09 ms per registration; DSP heap ≈ 182 MiB remains (doc 46
  §41, §48). There is no residency wall left for the MoE weights; the
  32-bit DSP address space is the limit for anything else we add.
* **Two weight formats, two model files** (doc 46 §35, §48.7):
  `QS4CX_WH` has **no CPU fallback** (`moe_htp_layers` must be empty, a
  wrong layout yields plausible wrong text, not an error); plain `Q4_0`
  is the CPU control. `NNTR_MOE_HTP_DECODE=1` is a measurement-only switch
  on the non-WH `QS4CX` model.
* **Runtime switches** live in `nntr_config.json`: `moe_engine`,
  `moe_htp_layers`, `conv_in_proj_engine`, `conv_out_proj_engine`,
  `attn_proj_engine`, `dense_ffn_engine` and their `*_htp_layers` lists.
  Instrumentation: `NNTR_HTP_PROFILE=2|3` (per-stage DSP breakdown; 3 =
  5× repeat, min), `NNTR_M0_PROFILE=1` (ARM side of the MoE layer),
  `--profile` build (per-node TYPE totals).
* **Traps** (doc 46 §48.7): rebuild the skel (`test/htp/build.sh`) whenever
  the IDL or DSP sources change — `build_android.sh` never does, and a
  stale skel fails with `AEE_EBADPARM (0x8000040E)`; `build_android.sh
  --clean` silently drops `-Denable-htp`; never push the link-time stub
  `libcdsprpc.so` to the phone; read `min`, not `avg`, from profiles.

## 3. Approach and order

### 3.1 Decode first (user decision Q3)

Doc 45's phase order (A MoE → B conv → C attention → D orchestration →
E decode) is prefill-centric. We invert it:

1. **Measure before touching code**: doc 48 §5's A / B / C plus the
   two-reader DDR probe, in the first handoff (§4.3).
2. **The three walls**, in the order the measurements dictate (DMA path,
   M=1 HVX GEMV, transport).
3. **One FastRPC call per token**: M=1 kernels for the rest of the layer
   (RMSNorm, conv1d + gating, RoPE, attention at M=1, dense FFN, lm_head)
   and the per-token orchestration entry. M=1 kernels are HVX, not HMX,
   and small; they may coexist with the prefill kernels as a second path
   (llama.cpp does the same).
4. Prefill-shape residency (doc 45 B/C/D) afterwards.

Every step carries the prefill gate (§1) because the DMA ring, the worker
pool and the weight layout are shared with the prefill path.

### 3.2 NPU-only first (user decision Q10)

Decode runs entirely on the DSP until the NPU-only ceiling is reached.
CPU+NPU expert splitting (the CPU takes one of the four active experts
per layer while the DSP takes three, synchronised once per layer) is a
**later** track. Its only justification is a two-reader DDR bandwidth
above one reader's 38 GB/s, which the first handoff's probe measures.

**Open decision (Q11, user, later):** the rule for when NPU-only stops
and the split begins. Proposed, not decided: after the three walls and the
one-call-per-token step, if the measured decode is < 50 tok/s **and** the
two-reader probe exceeded 45 GB/s.

### 3.3 Separate tracks, filed as issues when the time comes

Raising the ceiling (fewer bytes per token), prefill residency, CPU+NPU
split, upstreaming.

## 4. Machines and the measurement loop

### 4.1 Workstation (all agent work, all build gates)

Ubuntu, 8 cores, 30 GB RAM, 822 GB free under `/`. `source tools/htp/env.sh`
sets everything below; agents put that line first in every shell.

| thing | where | note |
|---|---|---|
| Hexagon SDK | `/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1` (toolv19, hexagon-clang 19.0.04) | the PR was developed on 6.4.0.2; 6.3.0.0 is also installed and not used |
| HexKL 1.0 beta.2 | `~/Qualcomm/hexkl-1.0-beta.2/hexkl_addon` (`HEXKL_ROOT`), `HEXKL_SDK_VER=6.4.0.1`: `lib/6.4.0.1/hexagon_toolv19_v79/libhexkl_micro.a` for the skel, `lib/6.4.0.1/armv8_android26/libsdkl.so` for the app | `~/Qualcomm/hexkl_addon` is the `hvx_impl`-style symlink view of the same package; `test/htp/build.sh` and `build_android.sh --htp` take the package root |
| Android NDK | `~/android-ndk-r30` (`ANDROID_NDK`; `.bashrc` exports it for interactive shells only) | the PR expects r26d; r30 compatibility is verified by the first app build (§9 rung 3) |
| Host build | `build/` (`meson setup build -Denable-transformer=true -Denable-tflite-backbone=false -Denable-tflite-interpreter=false`; no `flatc` here) | host gtests, `nntr_quantize_stream`, `run_host_checks.sh` |
| Models | `/local/mnt/workspace/models/lfm2.5-8b-a1b/{hf,fp32,q40,q40-qs4cx-wh}` (§11) | `NNTR_MODEL_DIR` |
| clang-format-14 | `~/.local/bin/clang-format-14` | AGENTS.md rule, changed lines only |
| adb | `/usr/bin/adb` | **agents never run it** |

**No Hexagon simulator in this project** (user decision 2026-09-21). Kernel
correctness is decided by host scalar specs and bit-identity checks (gate 1)
and by the device (gate 4); nothing in between.

### 4.2 Device (user only)

Galaxy S25 Ultra — **any unit** (user decision 2026-09-22; two units,
`R3CY205ZMND` and `R3CY10WM83Y`, exist and the PR author used
`R5KL30G6MLT`). A handoff never names a serial; the filled handoff
records the serial that ran (`adb devices`), BENCHMARK.md tags every row
with it, and results are compared only inside one sitting (A/B); when two
sittings happen to share a unit the same-unit drift is noted (LEDGER
rule 13). Agents never touch the phone. A task that needs
silicon numbers ends in a **measurement handoff**
(`docs/measurements/<issue#>-<slug>.md`, template in
`.claude/skills/hexagon-handoff`) that the user runs in one sitting.

Handoff rules (user decisions Q3, Q13, Q18, Q19):

* Every handoff is a **full-model end-to-end inference** command set
  (`nntrainer_causallm` on the phone with the real 8B model), never a
  kernel microbenchmark alone. Microbenchmarks may be added next to it.
* **Control run first, same sitting**: the unchanged reference binary
  (variant A) runs before any variant, and every result is read as an A/B
  inside that sitting (the PR's decode moved 20.8 → 16.5 between sessions
  with no code cause found, doc 50 §3.4; `hvx_impl` #53 measured ±5 %).
* At most 4 variants per handoff; prebuilt artifacts with md5 and commit;
  expected log lines; an empty result table with the reference numbers
  in it; estimated minutes at the top. Measurements happen on demand.
* Accuracy columns (text identical to CPU y/n, `NNTR_L2_DIFF`) are
  mandatory even on a speed sweep.

### 4.3 First handoff (issue: the tracker's first child)

1. A control: PR head unchanged, CPU `Q4_0` and NPU `QS4CX_WH`, prompt
   512, generation 64 / 512 / 1024, each twice → replaces the provisional
   "now" (§1).
2. Doc 48 §5's A (`--profile` decode breakdown of the ARM remainder), B
   (`NNTR_HTP_PROFILE=3` transport floor), C (arena DMA probe: linear vs
   2D descriptors, 1–4 engines, bus vote).
3. The two-reader DDR probe: CPU and DSP streaming disjoint buffers at
   once, aggregate GB/s.

## 5. Repository, branches, upstream

* Fork `dlwlzzero/nntrainer`. Integration branch **`htp_moe`**, created from
  upstream PR nntrainer/nntrainer#4327's head `2ce38d65` (branch
  `claude/htp-lfm2-moe-ffn` on `Seunghui98/nntrainer`, 2026-09-21) and
  **frozen there**. The supervisor reports new commits on the PR each
  cycle; merging them is the user's decision (Q16).
* Work branches `htp/<issue#>-<slug>`, PRs into `htp_moe`, linear history
  (rebase, never merge commits — the static check rejects empty bodies).
* `hvx_impl` is frozen: no cycles, no deletion. Its issues are closed or
  labelled out of the queue (§7).
* **Upstream-shaped** (Q14): follow AGENTS.md (DCO sign-off, `[component]`
  subjects, clang-format-14, no `subprojects/` edits, cross-platform),
  keep the PR's file structure (`nntrainer/tensor/htp_backend/**`,
  `test/htp/**`, `Applications/CausalLM/**`), and keep **kernel/app
  commits separate from agent-system commits** (`.claude/**`,
  `docs/plans/**`, `docs/measurements/**`, `docs/htp_moe/**`) so the former
  can be cherry-picked upstream.
* Everything is written in **English** (Q17): plans, handoffs, benchmark,
  agent files, commit messages, code comments. The PR's own
  `docs/htp_attention/*` stay as they are (mostly Korean) and are read,
  not edited.
* Commits: `git commit -s` in the user's name plus
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; one topic per
  commit; body ≥ 8 words; new `.c/.h/.cpp/.py` files carry a doxygen
  `@file`/`@brief` header.
* GitHub writes go through the claude.ai GitHub MCP or the `gh` CLI
  (logged in as the owner); both are acceptable.

## 6. Roles

One Claude Code session on the workstation is the orchestrator; it runs
`/hexagon-cycle`, which spawns the roles below as subagents. State lives
in issue labels; agents talk through issues, plans, handoffs and PRs.

| Role | May write | Must not |
|---|---|---|
| `hexagon-supervisor` | issues and labels, `docs/htp_moe/BENCHMARK.md`, `docs/htp_moe/LEDGER.md` | commit source code |
| `hexagon-planner` | `docs/plans/<n>-<slug>.md`; may run host builds and host checks for exploration | edit source |
| `hexagon-implementer` | source on a `htp/<issue#>-<slug>` branch, handoff docs, PRs into `htp_moe` | push to `htp_moe`, force-push, edit `.github/workflows` |
| `hexagon-guide-writer` | `docs/htp_moe/guide/*.html` (English, self-contained, **decode only**) | anything else |

Supervision scope: `nntrainer/tensor/htp_backend/**`, `nntrainer/htp_context.cpp`,
`test/htp/**`, `Applications/CausalLM/models/lfm2*/**`,
`Applications/CausalLM/layers/**` (only where the LFM2 decode path passes),
`tools/htp/**`, `tools/htp_*`, `docs/htp_moe/**`, `docs/plans/**`,
`docs/measurements/**`. Anything in nntrainer core outside these paths
(`nntrainer/tensor/float_tensor.cpp`'s dispatch, `compute_ops.h`, the
graph executor) is allowed **when the plan names it**, because the
one-call-per-token step cannot avoid it, and is otherwise an issue.

## 7. Issue state machine

Labels: every issue carries `hexagon`, one `state:*`, one `prio:*`.

```
state:needs-plan → state:planned → state:in-progress → state:review → (closed)
                                        │
                                        ├→ state:needs-measurement → state:measured → state:in-progress
                                        │
                                        └→ needs-user  (a decision or action only the user can take)
```

At most one `state:in-progress` issue. `state:measured` is processed
first. A merged PR closes its issue; the user merges.

Queue reset (Q6, Q12 of the session): one tracker issue for this project;
the `hvx_impl` issues #65 #69 #70 #71 (the old port plan) are closed as
superseded by the tracker; the Qwen3-only issues (#26 #41 #57 #58 #59 #60,
#37 #38 #40, #51) are closed as "frozen with hvx_impl" and stay
searchable. The supervisor lists only open `hexagon` issues, so nothing
else needs a new label.

## 8. One cycle (`/hexagon-cycle`)

1. Supervisor: process `state:measured` (update BENCHMARK.md rows and
   goal column, LEDGER.md rules, hand the issue back or close it); close
   issues whose PR merged; report new commits on upstream PR #4327; keep
   at least two `needs-plan`/`planned` issues, derived from the goal
   distance, the three walls, and LEDGER.md's open items.
2. Planner: top `state:needs-plan` → plan file → `state:planned`.
3. Implementer: top `state:planned` (or a handed-back `measured`) →
   gates → PR (`state:review`) or handoff (`state:needs-measurement`).
4. Guide writer: on `--guide`, or when a PR merged or a handoff was filled
   since the guide's last commit.
5. The cycle ends with the user's to-do list.

## 9. Verification gates (summary; commands in `.claude/skills/hexagon-gates`)

0. `clang-format-14` on changed C/C++ files.
1. **Host** (seconds to minutes): `ninja -C build`; the cpu-backend
   `*qs4cx*` gtests; `unittest_causallm_models --gtest_filter='*Lfm2Moe*'`
   (FP32 and Q4_0 differential on the tiny fixture);
   `test/htp/host/run_host_checks.sh` (scalar stand-ins for the kernel loop
   structure and worker pool); `tools/htp_syntax_check.sh`.
   A new DSP kernel ships with a scalar spec and a host check that proves
   bit-identity (doc 45 §3.3/§3.4).
2. **DSP skel** (`test/htp/build.sh`, v79, `-Wall -Werror`), md5 recorded.
3. **Android app + device gtest binary** (`Applications/CausalLM/build_android.sh --htp`,
   `ndk-build unittest_hvx_mm_u8i4`), md5 recorded. `readelf -d
   libnntrainer.so` lists `libsdkl.so` and `libcdsprpc.so`.
4. **Device** (user, handoff). Performance verdicts come only from filled
   handoffs.

## 10. Documentation locations

* Contract: this file. Plans: `docs/plans/`. Measurements: `docs/measurements/`.
* Goals and results table: `docs/htp_moe/BENCHMARK.md`.
* Rules learned on silicon, design verdicts, open items: `docs/htp_moe/LEDGER.md`.
* Beginner guide (decode only): `docs/htp_moe/guide/*.html`.
* The PR's design ledger `docs/htp_attention/*` is read-only history.

## 11. Weights

`/local/mnt/workspace/models/lfm2.5-8b-a1b/`:

| dir | content | how |
|---|---|---|
| `hf/` | `LiquidAI/LFM2.5-8B-A1B` (bf16 safetensors, 17.0 GB, 2 shards) | `hf download` |
| `fp32/` | `nntr_lfm2_8b_a1b_fp32.bin` (≈ 31 GB) + `config.json`, tokenizer files, an authored `nntr_config.json` | `Applications/CausalLM/res/lfm2_moe/lfm2-8b-a1b/weight_converter.py --model_path hf/` |
| `q40/` | all-Q4_0 model, the **CPU control** | `nntr_quantize_stream fp32/ -o q40/ --fc_dtype Q4_0 --embd_dtype Q4_0 --lmhead_dtype Q4_0 --isa ARM` |
| `q40-qs4cx-wh/` | FC/embed/lm_head Q4_0, MoE `QS4CX_WH`, the **NPU model** | same plus `--moe_dtype QS4CX_WH`; `nntr_config.json` gets `"moe_engine": "htp"`, `"moe_htp_layers": ""` |

Always `--isa ARM`; the `_ARM.bin` suffix must be present (an x86 packing
runs silently wrong on the phone). md5s of the produced `.bin` files are
recorded in BENCHMARK.md's artifact section once built.

## 12. Decisions log

| date | decision |
|---|---|
| 2026-09-21 | No simulator in this project: host bit checks + device only |
| 2026-09-22 | **Any device serial.** The same-device requirement (`R3CY10WM83Y` as the anchor unit, per-cell unit ratio) is dropped: any S25 Ultra may run an `htp_moe` handoff, handoffs name no serial, the filled handoff records the serial used, results are compared only inside one sitting (A/B) plus same-unit drift where the unit happens to repeat. The second-unit anchor sitting (#91 → #94 ⑮) is withdrawn, not re-filed. Same day: #94 sitting 2 replaces the provisional "now" (§1); its variant C (M=1 GEMV) is accepted as the LEDGER ⑯ verdict although the device skel was the user's own build of the same sources (rule 14 → rule 21) |
| 2026-09-21 | Base tree = PR #4327 head (Q1 b); model LFM2.5-8B-A1B; goal ≥ 50 tok/s decode (Q1 ii); decode on the NPU is the product condition, CPU decode stays the control (Q2); decode-first order (Q3); NPU-only first, CPU+NPU later (Q10); Q11 rule deferred to the user; prompt 512 + gen 64/512/1024 (Q6'); prefill gate −5 % of NPU now (Q12); control run per sitting (Q13); upstream-shaped commits (Q14); hvx_impl frozen (Q15); htp_moe frozen at the PR head, merges by user (Q16); English + decode-only guide (Q17); first handoff bundles control + doc 48 A/B/C + two-reader probe (Q18); on-demand measurements, ≤ 4 variants (Q19) |
