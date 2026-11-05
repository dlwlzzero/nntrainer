# #4456 refactoring notes

Working notes for the `mha_core.cpp` part of #4451 (umbrella #4456), written for whoever continues the work. Status as of 2026-11-05.

- [HANDOFF.md](HANDOFF.md): open PR stack, what to do when each PR merges, the remaining steps, known issues, working conventions and how to verify locally (Korean). The current state is the table in §2 and §12.11.
- One file per PR, each with its status, open review items, history, and the PR description as posted:
  - `mha_core` stack, in merge order:
    - [PR_4462.md](PR_4462.md): `unittest_mha_core`, the characterization test
    - [PR_4464.md](PR_4464.md): `rotary_embedding` whole-op
    - [PR_4465.md](PR_4465.md): softcap fix for the host flash and CUDA arms (#4461)
    - [PR_4466.md](PR_4466.md): `kv_cache_write` whole-op, plus the design of the remaining steps
    - [PR_4469.md](PR_4469.md): `attention` whole-op, host per-row arm (6c-1)
    - [PR_4477.md](PR_4477.md): host flash arm and the host arm choice into the op (6c-2)
    - [PR_4478.md](PR_4478.md): K-OHWI and int8 KV writes through `kv_cache_write` (6d)
    - [PR_4479.md](PR_4479.md): OpenCL attention, RoPE and concat KV write in `ClComputeOps` (7a)
  - Side PRs:
    - [PR_4473.md](PR_4473.md): no-RoPE staged-plane settle fix plus the device mode of `unittest_mha_core` (on #4462)
    - [PR_4472.md](PR_4472.md): drop the unused `hsumq_f16` (independent)
    - [PR_4459.md](PR_4459.md): recurrent cells declare their output lifespan (another #4456 item, independent)
  - Separate track, kept for reference: [PR_4480.md](PR_4480.md) (Gemma4 MoE HTP in QuickDotAI without the OpenCL series, draft)
- `cuda_stubs/`: CUDA 툴킷 없이 `nntrainer/cuda/*.cpp`를 `-fsyntax-only`로 검사할 때 쓰는 stub 헤더 (HANDOFF §9).
