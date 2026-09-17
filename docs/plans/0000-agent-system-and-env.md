# 0000 — Agent system and development environment for the Hexagon backend

Status: agreed 2026-09-16 (grill-me session, 3 rounds). This is the ground
truth for how the Hexagon (`hvx_impl`) work is organised from here on.
Every agent definition under `.claude/` links back to this file instead of
restating it.

## 1. Goal

Fill the `nntrainer` rows of [HEXAGON_BENCHMARK.md](../backend_guide/HEXAGON_BENCHMARK.md)
on a Galaxy S25 (Snapdragon 8 Elite) for Qwen3-0.6B, in two stages:

| Stage | Target | Why this target |
|---|---|---|
| 1 (HVX only) | W8A8: decode ≥ 60 tok/s @512 (provisional weight-stream bound, replaced by the ceiling #25 measures; issue #49, 2026-09-18) and ≥ 27.5 @4096; w4a8: decode ≥ 70.3 @512 and ≥ 27.5 @4096 (the GENIEX_LLAMACPP q4_0 NPU rows, issue #51); prefill interim goal 1000 tok/s @512 | Decode is bandwidth + host-path bound: 596 MB of int8 weights per step cannot reach 70.3 within the cDSP DDR ceiling, a 4-bit stream can (follow-ups ⑫ ① ⑨ ⑩, then #51). Prefill is compute bound; the 18× gap to LLAMACPP needs HMX. |
| 2 (HVX + HMX, after HexKL) | prefill ≥ 1000 tok/s @512, decode ≥ 60 tok/s, then chase the GENIEX_LLAMACPP prefill numbers | HMX is the only lever with that much MAC headroom. HexKL must be obtained first (needs-user). |

Decode reaching its goal does **not** imply prefill follows: they have
different bottlenecks (see HEXAGON.md §8.2 and the benchmark doc).

The `nntrainer` CPU rows are skipped for now.

## 2. Machines and the measurement loop

* **Mac (this machine)** is the editing/review client. It runs Claude Code
  and Docker (OrbStack) only. No Hexagon SDK, adb or model weights are
  installed natively.
* **Docker container** (`tools/docker/`) does everything that does not
  need a phone: x86 reference build and `hexagon_ref_run`, simulator golden
  tests, DSP skel + Android host harness cross-builds, clang-format-14.
  The Hexagon SDK is *mounted* from the host (`~/Qualcomm/Hexagon_SDK`),
  never baked into the image (licence). Android NDK r26d is baked in.
* **Workstation + Galaxy S25 Ultra** are operated by the user only. Agents
  never run adb. When a task needs silicon numbers, the implementer writes a
  *measurement handoff* (`docs/measurements/<issue#>-<slug>.md`, template in
  `.claude/skills/hexagon-handoff`) that lists the prebuilt artifacts (path
  + md5 + commit), the exact commands, expected log lines and an empty
  result table. The user runs it, fills the table, commits it on the same
  branch and moves the issue to `state:measured`.

### Closing the simulator ↔ device gap

Three kinds of gap have been recorded; each gets a standing rule:

1. **Performance gap** (the simulator does not charge DDR/DMA bandwidth,
   `--timing` is impractically slow). Simulator pcycles are a correctness
   gate and a *relative* signal only. Every performance verdict comes from a
   device measurement. To keep the number of handoffs small, one handoff
   bundles up to 4 skel variants (`HEX_EXTRA_CFLAGS` build matrix); sweeps
   run at 512 tokens only, goal checks run 512/1024/4096.
2. **Numerical gap** (v79 IEEE/qf32 chain misbehaves on silicon, HEXAGON.md
   §7; Qfloat tie rounding; upstream qf32 drift ⑮). Kernel PRs are reviewed
   against the §7 rule list. Every handoff carries accuracy columns
   (`--eval` PPL / top-1, `find_divergence.py` bound) next to tok/s so the
   three-way x86-ref / sim / device comparison is always on record. A
   device-only divergence becomes the next issue and follows the P4 ⑮
   bisection procedure (1-layer image, `--dump-op`).
3. **Environment gap** (stale image on the device, same size, wrong
   layout). `push_if_changed` already compares md5. Handoff docs list the
   commit hash and the md5 of every artifact; the result table repeats the
   md5 of the skel that actually ran. One toolchain (the container's SDK)
   builds everything.

Whenever a new gap is found, the supervisor appends a rule to HEXAGON.md §7.

## 3. SDK

Hexagon SDK **6.4 or newer**, installed once by the user with the Linux
`qpm-cli` *inside* the container (wizard: `tools/docker/setup_wizard.sh`)
into the host directory that `run.sh` mounts. The workstation's 6.0.0.2 is
no longer used for builds. Issue #23 rebuilds `hvx_impl` HEAD with 6.4,
reruns the 13 simulator tests on v75 and v79, builds both skels and hands
off a regression measurement against HEXAGON.md §8.2 (P4: 192.1 / 27.7
tok/s @512). Matching numbers close follow-up ⑭; the v79 run doubles as the
first data point for ④.

HexKL (`libhexkl_micro.a`) goes under the same host directory
(`~/Qualcomm/hexkl_addon`) when obtained; the wrapper mounts it if present.

## 4. Roles

One Claude Code session on the Mac is the orchestrator. It reads the role
files in `.claude/agents/` and spawns them as subagents. State lives in
GitHub issue labels on `dlwlzzero/nntrainer`; agents talk through issues,
plans, handoff docs and PRs, never directly.

| Role | May write | Must not |
|---|---|---|
| `hexagon-supervisor` | issues, labels, `docs/backend_guide/HEXAGON_BENCHMARK.md`, HEXAGON.md §7/§8/§9 | commit source code |
| `hexagon-planner` | `docs/plans/<n>-<slug>.md`; may run exploratory builds in the container | edit source |
| `hexagon-implementer` | source on a `hvx/<issue#>-<slug>` branch, handoff docs, PRs into `hvx_impl` | push to `hvx_impl`, force-push, edit `.github/workflows` |
| `hexagon-guide-writer` | `docs/backend_guide/hexagon-guide/*.html` (self-contained, English) | anything else |

Supervision scope is the Hexagon subtree only: `nntrainer/tensor/hexagon/**`,
`Applications/CausalLM/hexagon/**`, `tools/hexagon/**`, `tools/docker/**`,
`test/hexagon/**`, `test/htp/**`, `docs/backend_guide/HEXAGON*.md`,
`docs/backend_guide/hexagon-guide/**`. Anything in nntrainer core is filed as
an issue, not touched.

GitHub access: the claude.ai GitHub MCP returned `403` for issue and PR
creation on this fork on 2026-09-16 but wrote issues, comments and labels
fine on 2026-09-17; either path is acceptable, and the `gh` CLI (logged in
as the repo owner) is the fallback, so every role that writes to GitHub
has `Bash`. Work done from a remote claude.ai session lands on a
`claude/<slug>` branch that can push nowhere else; the next Mac cycle
re-cuts it into per-issue `hvx/<issue#>-<slug>` branches before any PR
(2026-09-17 handover, issue #42). Labels were created
on 2026-09-16; the first queue is issues #23–#28 and PR #22.

All roles inherit the session model. Commits use `git commit -s` in the
user's name plus `Co-Authored-By: Claude ... <noreply@anthropic.com>`, one
topic per commit, `[component] message` subjects (AGENTS.md).

## 5. Issue state machine

Labels (all issues also carry `hexagon`, and one of `prio:p0` `prio:p1`
`prio:p2`):

```
state:needs-plan → state:planned → state:in-progress → state:review → (closed)
                                        │
                                        ├→ state:needs-measurement → state:measured → state:in-progress
                                        │
                                        └→ needs-user  (a decision or action only the user can take)
```

Rules: at most one `state:in-progress` issue at a time. `state:measured`
is processed before anything else. A PR that merges closes its issue
(`completed`). Merging is done by the user.

## 6. One cycle (`/hexagon-cycle`)

1. Supervisor: process `state:measured` (update benchmark + HEXAGON.md §8,
   re-plan or close); close issues whose PR merged; if fewer than two
   `needs-plan`/`planned` issues exist, derive new ones from the benchmark
   goals, HEXAGON.md §9 and the follow-up ledger; on `--architecture`, run a
   clean-architecture sweep of the subtree and file issues.
2. Planner: turn the top `state:needs-plan` into a plan file, label
   `state:planned`.
3. Implementer: if nothing is `in-progress` (a `needs-measurement` issue
   does not block a planned issue that needs no device step; still at most
   one `in-progress`), take the top `state:planned`, implement through the
   gates (`.claude/skills/hexagon-gates`), then either open a PR
   (`state:review`) or write a handoff (`state:needs-measurement`).
4. Guide writer: on `--guide`, when a PR merged since the last guide
   update, or when a measurement handoff was filled since then, refresh
   `docs/backend_guide/hexagon-guide/`; every device measurement lands in
   the guide's `06-performance.html` (history table + chart).
5. The cycle ends and prints the user's to-do when only
   `needs-measurement`, `review` or `needs-user` issues remain.

The first cycles are run by hand; `/loop /hexagon-cycle` once the prompts
are stable.

## 7. Verification gates (summary; details in `hexagon-gates`)

x86 ref tests → simulator (only when DSP bytes change, see below) → skel +
host harness compile (v79, the primary arch since #35 — the target is the
S25 Ultra's v79 silicon; v75 only when a change touches an
`__HVX_ARCH__` branch, or on request) → device numbers only via handoff.
clang-format-14 on changed lines.

Simulator budget (agreed 2026-09-17; the Mac runs `hexagon-sim` under
Rosetta, `profile acc` alone is ~10 min and the 13 tests ~5 min, so sim
time is the scarcest agent resource after device time):

* A change that touches no file under `nntrainer/tensor/hexagon/htp/`,
  `test/hexagon/sim_*` or the packer/lowering (host, tools, docs, tests
  outside the sim harness) runs **no simulator test**. The PR states
  "no DSP bytes changed" and the skel md5 from rung 4 proves it.
* A change that touches one op kind runs that kind's sim test per step
  (`run_sim_test.sh <kind>`, seconds to a minute) and `profile acc` +
  the 13 tests **once**, right before the PR, never per commit.
* `profile acc` is a correctness gate (STAT bit-identity), not a timing
  tool; `SIM_TIMING` is never enabled by an agent.
* A planner probe on the simulator is limited to one targeted test; if
  more is needed, it becomes step 1 of the implementation plan.
* Every simulator run is v79 (user decision 2026-09-17, after #35: the
  device is the S25 Ultra). v75 is neither built nor simulated per PR;
  its last record is HEXAGON.md §5.2 / §8.3 "v75 final record", and it
  is run only when a change touches an `__HVX_ARCH__` branch or the
  user asks.

## 8. Fork CI

PRs into `hvx_impl` currently trigger the full upstream matrix. Reducing it
for `hvx/*` branches is filed as a `needs-user` issue: workflow edits are
the one change that always needs explicit user approval.

## 9. Documentation locations

* Comparison table + goals: `docs/backend_guide/HEXAGON_BENCHMARK.md`
  (English). The blog copy at `dlwlzzero.github.io/_study/2026-09-16-WTD.md`
  is a published snapshot, updated by the user.
* Design and results: `docs/backend_guide/HEXAGON.md` (unchanged role).
* Plans: `docs/plans/`. Measurements: `docs/measurements/`.
* Beginner guide: `docs/backend_guide/hexagon-guide/index.html` and
  siblings, English, no build step, viewable by opening the file (GitHub
  shows the source; enabling Pages on the fork would render it).
* The `docs/superpowers/specs` tree stays the historical design ledger.
