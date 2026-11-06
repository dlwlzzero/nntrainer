# #4456 refactoring notes

Working notes for the `mha_core.cpp` part of #4451 (umbrella #4456), written for whoever continues the work. Status as of 2026-11-05.

- [HANDOFF.md](HANDOFF.md): open PR stack, what to do when each PR merges, the remaining steps, known issues, working conventions and how to verify locally (Korean). The current state is the table in §2 and §12.11.
- One file per PR, each with its status, open review items, history, and the PR description as posted:
  - PR_4462.md, PR_4464.md and PR_4472.md instead hold the user prompts and AI replies that produced each PR, in time order (this session and the user's other session), followed by the commits and the PR description.
  - `mha_core` stack, in merge order:
    - [PR_4462.md](PR_4462.md): `unittest_mha_core`, the characterization test (merged 11-05)
    - [PR_4464.md](PR_4464.md): `rotary_embedding` whole-op (merged 11-06)
    - [PR_4465.md](PR_4465.md): softcap fix for the host flash and CUDA arms (#4461)
    - [PR_4466.md](PR_4466.md): `kv_cache_write` whole-op, plus the design of the remaining steps
    - [PR_4469.md](PR_4469.md): `attention` whole-op, host per-row arm (6c-1)
    - [PR_4477.md](PR_4477.md): host flash arm and the host arm choice into the op (6c-2)
    - [PR_4478.md](PR_4478.md): K-OHWI and int8 KV writes through `kv_cache_write` (6d)
    - [PR_4479.md](PR_4479.md): OpenCL attention, RoPE and concat KV write in `ClComputeOps` (7a)
  - Side PRs:
    - [PR_4473.md](PR_4473.md): no-RoPE staged-plane settle fix plus the device mode of `unittest_mha_core` (on #4462)
    - [PR_4472.md](PR_4472.md): drop the unused `hsumq_f16` (independent, merged 11-06)
    - [PR_4459.md](PR_4459.md): recurrent cells declare their output lifespan (another #4456 item, independent)
  - Separate track, kept for reference: [PR_4480.md](PR_4480.md) (Gemma4 MoE HTP in QuickDotAI without the OpenCL series, draft)
- [agents_rules/SESSION.md](agents_rules/SESSION.md): 별도 트랙. upstream용 폴더별 `AGENTS.md` 규칙 파일 작업의 세션 기록 (대화 흐름, 결정, 근거 수집 방법) + `agents_rules/records/` 근거 자료. 규칙 파일 자체는 main에서 딴 `dlwlzzero:docs/agents-rules` 브랜치에 있고 upstream PR은 미생성.
- [ARCH_MAP_SESSION.md](ARCH_MAP_SESSION.md): 별도 트랙. `qc_npu_rebase` 기준 현재 백엔드 구조와 `ARCHITECTURE_REFACTOR.md` 목표 구조를 비교한 HTML 페이지 작업의 세션 기록 (사용자 프롬프트와 AI 응답 시간순, 결정 요약). 페이지 링크와 버전은 [ARCH_MAP_PAGE.md](ARCH_MAP_PAGE.md), PDF 사본은 [ARCH_MAP_PAGE.pdf](ARCH_MAP_PAGE.pdf).
- `cuda_stubs/`: CUDA 툴킷 없이 `nntrainer/cuda/*.cpp`를 `-fsyntax-only`로 검사할 때 쓰는 stub 헤더 (HANDOFF §9).
