# #4456 refactoring notes

Working notes for the `mha_core.cpp` part of #4451 (umbrella #4456), written for whoever continues the work. Status as of 2026-11-02.

- [HANDOFF.md](HANDOFF.md): open PR stack, what to do when each PR merges, the review items still open on #4469, the remaining steps, known issues, working conventions and how to verify locally (Korean).
- One file per PR, each with its status, open items, history, and the PR description as posted:
  - [PR_4462.md](PR_4462.md): `unittest_mha_core`, the characterization test
  - [PR_4464.md](PR_4464.md): `rotary_embedding` whole-op
  - [PR_4465.md](PR_4465.md): softcap fix for the host flash and CUDA arms (#4461)
  - [PR_4466.md](PR_4466.md): `kv_cache_write` whole-op, plus the design of the remaining steps
  - [PR_4469.md](PR_4469.md): `attention` whole-op, host per-row arm (6c-1)
