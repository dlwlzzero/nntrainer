# Handoff — Hexagon HVX M2 커널 (Task 10까지 완료)

날짜: 2026-08-18 · 브랜치: `hvx_impl` · HEAD: `2a05feb4`
계획: `docs/superpowers/plans/2026-08-18-hexagon-hvx-m2-kernels.md`
스펙: `docs/superpowers/specs/hexagon-hvx/02-hvx-kernels.md`
진행 원장: `.superpowers/sdd/progress.md` (git clean -fdx 시 소실 주의 — 이 문서와 git log로 복구)

## 1. 진행 상태 (Task 1–10 완료 / 11–15 남음)

| Task | 내용 | 커밋 | 검증 |
|---|---|---|---|
| 1 | sim 테스트 하네스 (build/run_sim_test.sh, cmp_f) | 505d4b7e | smoke PASS |
| 2 | 와이어 포맷 v2 + oplist validate | fecc1c2e | x86 gcc PASS |
| 3 | QuRT 워커 풀 (워커별 start sem) | b3e20c7f | pool PASS (n=4) |
| 4 | ggml-hexagon 차용 (hex/ dma/ hvx/ 11파일, MIT) | 2aee7dc5 | exp PASS |
| 5+6+7 | hvx-quant.h + htp_ops.h + W8A8 matmul (DDR+VTCM DMA) | ce5d1533 | quant/matmul/matmul_dma PASS (bit-exact) |
| 8 | RMSNORM (+hvx-f16-math.h, cmp_f STAT) | 6eff9897 | rmsnorm PASS |
| 9 | ROPE (rotate-half, 테이블 기반) | fd711eed | rope PASS |
| 10 | ADD + SILU_MUL | 2a05feb4 | eltwise PASS |
| 11 | EMBED | — | 다음 작업 |
| 12 | ATTN (KV append + causal SDPA + GQA) | — | |
| 13 | MATMUL_LOGITS | — | |
| 14 | htp_graph 실행기 + 합성 golden 테스트 | — | M2 최종 게이트 |
| 15 | FastRPC 글루 + skel 빌드 + 문서 | — | |

미커밋 워킹트리: `.gitignore`의 `docs/superpowers/` ignore 한 줄만 (의도적으로 커밋 제외 중).

## 2. 워크플로 규칙 (사용자 계약 — 반드시 준수)

- **승인 게이트**: 에이전트는 빌드/테스트 직접 실행 금지. 검증 명령 제시 → 정지 → 사용자가 실행한 출력으로 판정. 사용자 승인 후에만 커밋. 커밋 메시지는 매번 초안을 먼저 보여주고 승인받음.
- **subagent-driven-development**: Task마다 구현 서브에이전트(ponytail full 적용) → 리뷰 서브에이전트(스펙+품질) → Critical/Important는 픽스 후 재리뷰 → 사용자 검증 → 승인 커밋. 컨트롤러는 직접 구현하지 않음(결정적 도구 실행·git 정리는 예외).
- **커밋 형식**: `[htp] 제목` + 바디 + `Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>` + `Co-authored-by: Claude Fable 5 <noreply@anthropic.com>`. 커밋 전 사용자가 히스토리 재작성(reset 후 재커밋)을 요구하는 경우 있음 — Task 4·5+6 전례 참조.
- **plan/spec/handoff 문서는 커밋하지 않음** (docs/superpowers/는 gitignore 대상).

## 3. 명명·배치 규칙 (사용자 결정 사항)

1. **디렉토리 = 하드웨어 유닛**: `htp/hex/`(스칼라 유틸), `htp/dma/`(user-DMA 큐), `htp/hvx/`(HVX 프리미티브), `htp/ops/`(와이어 op 핸들러). 벤더 차용분도 이 기준으로 분산 배치됨.
2. **op 커널 파일**: `ops/hvx-<op>.c` (구현 유닛 접두사). 남은 것: `hvx-embed.c`, `hvx-attn.c`, logits는 `hvx-matmul.c`에 추가.
3. **op 구현 함수**: `hvx_op_*` (예: `hvx_op_matmul_w8a8`). **계약명은 `htp_*` 유지**: `htp_exec_ctx`, `htp_op_fn`, `htp_op_table`, `htp_ref_ptr`, `htp_m`, `htp_ops.h`.
4. **hvx/ 승격 기준**: 소비자 2곳 이상인 프리미티브만 hvx/ 헤더로. op 전용 벡터 코드(인트린식 포함)는 해당 ops/ 파일에 인라인 (예: rope_rotate는 hvx-rope.c 내부).
5. 파일명은 대시(`hvx-f16-math.h`), 벤더 함수 접두사도 유닛 기준(`hmx_ceil_div`→`hex_ceil_div` 전례).

## 4. 툴체인 함정 (전부 실제로 밟은 것)

- hexagon-clang에서 `int32_t`/`uint32_t` = long 계열 → printf에 `(int)`/`(unsigned)` 캐스트 필수.
- `<malloc.h>` 없음 — `memalign`은 `<stdlib.h>`.
- HVX 인트린식은 `hvx_hexagon_protos.h`(hexagon_protos.h 아님)에서 철자 확인 필수.
- **qf16 덧셈/뺄셈은 상쇄 구간에서 위험**: 비정규화 포맷이라 결과 정밀도가 큰 피연산자 ulp로 고정. 합산은 IEEE hf add(`Q6_Vhf_vadd_VhfVhf`) 또는 qf32/sf 누산 사용. 곱셈은 qf16 안전. (Task 10에서 FAIL로 실증)
- `qurt_hvx_get_units()`는 개수가 아닌 인코딩(0x400 → bits[15:8]=128B 유닛 4개). 디코딩 필수.
- QuRT sim 실행에는 qtimer/l2vic cosim(q6ss.cfg) + osam.cfg 필수 — 없으면 부팅 직후 행업. run_sim_test.sh가 자동 생성.
- sim은 user-DMA 지원 확인됨(matmul_dma PASS). 단 `dma_queue_pop`은 무한 busy-wait — DMA 계열 실패는 행업으로 나타남(timeout 걸고 실행).
- 벤더 `hvx_vec_f16_to_f32/f32_to_f16`은 pre-shuffle로 **위치 기반 lo/hi** 반환(even/odd 인터리브 아님) — 왕복 시 lane 원위치 보존.

## 5. CI 규칙 (신규 파일마다)

- `// SPDX-License-Identifier: Apache-2.0` + doxygen 블록(`@file @date @brief @see @author dlwlzzero <dlwlzzero@gmail.com> @bug`) — 벤더 파일은 MIT + 출처 문구를 doxygen 블록 안에.
- repo `.clang-format` 준수(벤더 파일은 재포맷 금지), 영문 주석만.
- 체커: `.github/workflows/static.check.yml`(doxygen-tag advanced, exec-bit, hardcoded-path 등) + `cpp_linter.yml`(clang-format).

## 6. 검증 절차

```bash
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.0.0.2/setup_sdk_env.source
./tools/hexagon/build_sim_test.sh          # v75 크로스빌드
./tools/hexagon/run_sim_test.sh <name>     # smoke pool exp quant matmul matmul_dma rmsnorm rope eltwise
```

cmp_f가 비교마다 `STAT max_abs=.. max_rel=..` 출력. max_rel은 참값≈0 원소에서 폭발 가능(정상, atol이 커버). 판정은 혼합식 `|d| ≤ atol + rtol·|ref|`.

## 7. 남은 Task 주의사항

- **Task 12 (ATTN)**: 출력 누산은 반드시 qf32 페어(계획대로) — qf16 합산 금지(§4). `hvx_dot_fp16`을 `hvx-f16-math.h`에 추가(두 번째 소비자 승격 규칙). 워커 분할은 kv_head 단위. KV 레이아웃/GQA 매핑은 공통 설계 참조.
- **Task 14 (실행기) 필수 추가 사항(사용자 요구)**: `htp_graph_forward` 진입부에서 `n_tokens==0 || n_tokens>max_chunk || pos+n_tokens>max_seq` 거부 + test_graph에 위반 케이스. 근거: pos/n_tokens는 런타임 인자라 init validate가 못 막고, 위반 시 rope 테이블 밖 읽기·KV cache 밖 쓰기 발생.
- **Task 15**: M1 더미 경로(`n_ops==0`) 유지 필수 — 디바이스 검증은 S25 Ultra(R3CY205ZMND), S26은 불안정(메모리 참조).
- 이연 항목(최종 whole-branch 리뷰에서 triage): ATTN KV 검사 주석, validate 도달불가 default, matmul DMA per-call 큐 할당(최적화 후보), rows_per_buf<1 폴백 미테스트, hvx-f16-math.h 디버그 assert 부재, kMaxExp 상수 중복, test_quant @brief 줄바꿈.

## 8. 브리프/리포트 위치

- Task 브리프: `.superpowers/sdd/task-N-brief.md` (task-brief 스크립트로 계획에서 추출)
- 공통 설계 발췌: `.superpowers/sdd/m2-common-design.md` (구현·리뷰 디스패치에 항상 동봉)
- 구현/리뷰 리포트: `.superpowers/sdd/task-N-report.md`, 리뷰 diff: `task-N-review.diff` (add -N으로 untracked 포함해 생성)
