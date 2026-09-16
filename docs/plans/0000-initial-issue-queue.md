# Initial issue queue for the Hexagon work

Issues could not be filed on 2026-09-16 because the fork has Issues
disabled (`has_issues: false`). Once the user enables them (repository
Settings → General → Features → Issues), the supervisor creates these on
its first `/hexagon-cycle` run, in this order, and then deletes this file.
Labels: every issue gets `hexagon`; the state and priority are given per
item.

## 1. Rebuild hvx_impl with Hexagon SDK 6.4+ in the dev container and re-measure the M6 P4 baseline (ledger ⑭, ④)

`state:needs-plan` `prio:p0`

Goal: every Hexagon build moves to the container (`tools/docker/`) with
SDK 6.4 or newer; prove nothing regressed against the last device numbers
recorded with SDK 6.0.0.2.

Acceptance:
1. In the container: the 13 simulator tests + `profile acc` pass on v75
   **and** v79 (6.4 ships a v79 QuRT image).
2. v75 and v79 skels + host harness build.
3. Handoff `docs/measurements/1-sdk64-baseline.md`, run by the user at
   512/1024/4096 with `--chunk 128 --steps 64` and `--eval`: the v75 skel
   reproduces HEXAGON.md §8.2 M6 P4 within noise (192.1 / 27.7 @512,
   118.2 / 22.9 @1024, 27.3 / 9.4 @4096; §8.1 PPL band). The v79 skel run
   records whether the §7 rule-1 IEEE/qf32 misbehaviour reproduces — data
   for ④, not a pass/fail here.
4. HEXAGON.md §5.2 / §8.3 SDK and toolchain notes updated; hard-coded
   `/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.0.0.2` paths replaced by
   `HEXAGON_SDK_ROOT`.

Prerequisite (user): `tools/docker/setup_wizard.sh` completed. Checks
from ledger ⑭: `setup_sdk_env.source` handling, `run_main_on_hexagon`
`hexagon_toolv*_v79` present, `-mhvx-ieee-fp` compatibility,
`hvx_hexagon_protos.h` intrinsic renames, HexKL (6.0.0.2 libs) coexistence.

## 2. Host/RPC decode path: stop copying 151,936 logits per step (ledger ⑫)

`state:needs-plan` `prio:p0`

Device data (HEXAGON.md §8.2, 2026-09-16): DSP decode 26.8 ms but host
wall 45.8 ms; generation mode 35 ms DSP / 76.4 ms host. About 40 ms per
step is FastRPC return + logits copy + host argmax. Candidates in the
ledger's order: (c) logits into an rpcmem ACT slot read zero-copy by the
host, then (a) return top-k (value + index) from `forward()`.

Acceptance: decode host wall per step at 512 within 5 ms of the DSP
pcycle time; generated ids identical to the current path for greedy;
`hexagon_rpc_test` still passes; ABI bump documented in §1.4. Handoff
required (host-side timing is device-only).

## 3. Cross-op weight prefetch for decode (ledger ①) with `MM_TB` / chunk sweep (⑨ ⑩)

`state:needs-plan` `prio:p1`

Kick the next matmul's first DMA chunk while ATTN/RMSNORM/eltwise run;
promote `dma_queue` to graph lifetime; reserve worker slab buf[0] for the
next op. Bundle in the same handoff: `MM_TB` 1/2/4/8 and `MM16_R` 2/4/8 ×
`MM16_TB` 1/2/4 at 512 tokens (sim cannot decide these, §8.3).

Acceptance: decode @512 improves by the non-matmul share measured in the
P1 profile or the sweep picks defaults with a device number behind each;
outputs bit-identical to the current path.

## 4. Prefill attention: K^T reuse across query rows (P5)

`state:needs-plan` `prio:p1`

HEXAGON.md §9 first item; `ponytail:` note in `hvx-attn.c`. The only cost
that grows with context; dominates the 4096-token rows (27.3 / 9.4).
Includes the ⑮ bisection of the layer-0 SILU_MUL 2.65 % rel-RMS drift
while the kernel is open.

Acceptance: prefill @4096 Mcycles/tok drops measurably on the device and
the 8-token accuracy gate holds; the drift source is named in §8.

## 5. Obtain HexKL (libhexkl_micro.a) for the HMX stage

`needs-user` `prio:p1`

Stage 2 needs HexKL 1.0 Beta.2 or newer (`specs/hexagon-hmx/00-overview.md`).
User action: check the Qualcomm account / NDA path and whether `qpm-cli`
lists a HexKL package; place it under `~/Qualcomm/hexkl_addon` (mounted by
`run.sh`). If it cannot be obtained, the fallback investigation is
"HMX intrinsics directly from SDK 6.4 headers" and becomes a planner issue.

## 6. Reduce fork PR CI for `hvx/*` branches

`needs-user` `prio:p2`

PRs into `hvx_impl` run the full upstream matrix (Android, Tizen, Windows,
Yocto). Proposal: for `hvx/*` head branches run only Ubuntu meson (gcc,
clang), `cpp_linter`, `static.check`. Workflow edits are the one change
that always needs explicit user approval; the implementer does it only
after the user says yes on the issue.
