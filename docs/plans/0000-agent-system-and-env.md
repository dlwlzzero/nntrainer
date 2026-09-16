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
| 1 (HVX only, W8A8) | decode ≥ 70.3 tok/s @512 and ≥ 27.5 tok/s @4096 (the GENIEX_LLAMACPP NPU rows); prefill interim goal 1000 tok/s @512 | Decode is bandwidth + host-path bound and reachable without new hardware paths (follow-ups ⑫ ① ⑨ ⑩). Prefill is compute bound; the 18× gap to LLAMACPP needs HMX. |
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
no longer used for builds. Issue #1 rebuilds `hvx_impl` HEAD with 6.4,
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
3. Implementer: if nothing is `in-progress` and nothing is
   `needs-measurement`, take the top `state:planned`, implement through the
   gates (`.claude/skills/hexagon-gates`), then either open a PR
   (`state:review`) or write a handoff (`state:needs-measurement`).
4. Guide writer: on `--guide`, or when a PR merged since the last guide
   update, refresh `docs/backend_guide/hexagon-guide/`.
5. The cycle ends and prints the user's to-do when only
   `needs-measurement`, `review` or `needs-user` issues remain.

The first cycles are run by hand; `/loop /hexagon-cycle` once the prompts
are stable.

## 7. Verification gates (summary; details in `hexagon-gates`)

x86 ref tests → `profile acc` on the simulator (per task) → all 13 sim
tests (before PR) → skel + host harness compile (v75, and v79 once ⑭ is
done) → device numbers only via handoff. clang-format-14 on changed lines.

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
