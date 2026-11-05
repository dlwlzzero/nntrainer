# Agent 규칙 파일 (AGENTS.md) 작업 노트

- 기준 시각: 2026-11-05
- 목표: nntrainer에서 AI 에이전트가 코드 작업할 때 따를 규칙을 md 파일로 만든다. 코드베이스를 단위별로 나누고 단위마다 규칙 파일을 둔다. **upstream main (nntrainer/nntrainer)에 PR로 올린다.**
- 상태: PR 1 커밋 2개를 fork `dlwlzzero:docs/agents-rules` @ `e961fa7b7`에 push함. **upstream PR은 아직 안 열었음** (사용자 지시 대기).

## 1. 결정 사항

| 항목 | 결정 | 이유 |
|---|---|---|
| 대상 브랜치 | upstream main | 사용자 결정. 그래서 범용 규칙만 담고 HTP/Gemma fork 실험 규칙(bit-identity, byte-identical 그래프 등)은 뺐다 |
| 나누는 기준 | 디렉토리 전체가 아니라 **규칙이 실제로 다른 곳만** | 디렉토리마다 파일을 만들면 코드를 옮겨 적는 문서가 되고 금방 낡는다 |
| 규칙의 원천 | 코드 읽기가 아니라 **리뷰 증거** | 코드만 읽고 만든 규칙은 "에러 처리 잘 해라" 같은 no-op이 된다 |
| 파일 형식 | 폴더별 `AGENTS.md` + 루트 `AGENTS.md`의 포인터 표 | upstream에는 루트 `AGENTS.md`만 있고 `CLAUDE.md`가 없다. 포인터 표는 Claude Code, Codex 등 어느 에이전트에서나 동작한다 |
| PR 분할 | 작게 시작: 루트 수정 + CausalLM 파일 1개 | 형식이 받아들여지면 나머지를 이어서 올린다. 처음부터 다 쓰면 형식 거절 시 전부 다시 써야 한다 |

## 2. 방법 (근거 수집)

1. `gh api repos/nntrainer/nntrainer/pulls/comments`로 inline 리뷰 코멘트 4000개 수집 (API 상한). 실제 기간은 **2024-11 ~ 2026-11** (API가 생성일 순으로 정렬하지 않아서 처음 예상한 6개월보다 넓었다).
2. 봇, PR 작성자 본인 코멘트, 답글을 빼고 사람 리뷰어의 최상위 코멘트 **1,678개**를 경로별 단위로 나눴다.
3. 서브에이전트 5개가 그룹별로 분석: apps (CausalLM, 기타 앱), tensor (cpu_backend, tensor 코어), core (layers, models, compiler/graph, 기타 nntrainer), accel (OpenCL, QNN/HTP), cross (test, api, CI, 빌드/패키징).
4. 규칙 후보 조건: **2개 이상 PR에서 반복** 또는 **메인테이너가 컨벤션으로 명시**, 코드 한 번 보고 알 수 없는 것, 기본 동작을 바꾸는 것, upstream 범용. 모든 후보는 upstream main 코드/커밋과 대조.
5. upstream main `640ac6e0e`의 `fix`/`revert` 커밋도 같이 봤다.

단위별 코멘트 수 (사람 최상위 코멘트 / PR 수):

| 단위 | 코멘트 | PR |
|---|---|---|
| Applications/CausalLM | 287 | 105 |
| nntrainer/tensor/cpu_backend | 225 | 79 |
| nntrainer/tensor (코어) | 211 | 97 |
| nntrainer/layers | 182 | 79 |
| 기타 nntrainer/ | 161 | 60 |
| test | 135 | 67 |
| OpenCL (cl_operations + opencl) | 125 | 53 |
| models / compiler+graph / api / qnn+htp | 82 / 49 / 26 / 4 | 36 / 27 / 16 / 1 |

## 3. 결과: 파일별 규칙 수

| 파일 | 규칙 | 상태 |
|---|---|---|
| 루트 `AGENTS.md` | 정정 5건 + 공통 규칙 8개 + 크로스플랫폼 함정 + CI 미검사 목록 | **PR 1** |
| `Applications/CausalLM/AGENTS.md` | 8 | **PR 1** |
| `nntrainer/tensor/cpu_backend/AGENTS.md` | 7 | 후속 |
| `nntrainer/tensor/AGENTS.md` | 4 | 후속 |
| `nntrainer/opencl/AGENTS.md` (cl_operations 포함) | 6 | 후속 |
| `nntrainer/layers/AGENTS.md` | 7 | 후속 |
| `nntrainer/AGENTS.md` (models, compiler/graph, utils, qnn) | 7 | 후속 |
| `test/AGENTS.md` | 7 | 후속 |
| `api/AGENTS.md` | 3 | 후속 |

만들지 않는 것: `htp_backend`/`qnn` 별도 파일 (근거 부족, upstream에 htp_backend 파일 5개뿐), 기타 샘플 앱, `docs/`, `.github/`, optimizers/dataset/mem_allocator (규칙 0~1개).

## 4. PR 1 내용

- 브랜치: `dlwlzzero:docs/agents-rules`, upstream main `640ac6e0e` 위
- 로컬 worktree: `~/nntrainer-agents-rules`

### 커밋 1: `14eca6e9b` [docs] AGENTS.md: correct the CI facts and add rules for every change

루트 `AGENTS.md`의 틀린 내용 (워크플로 파일에서 직접 확인):

- `check_count`는 테스트 수가 아니라 **승인 리뷰어 수**를 센다 (`pull_request_review` 트리거). "테스트 추가/삭제 시 count 갱신" 문장은 틀렸다.
- `pr-desc`는 PR 본문을 **검증하지 않는다**. `ai-describe` 라벨이 있을 때 AI 요약 코멘트만 단다.
- minja는 `subprojects/`가 아니라 `Applications/CausalLM/third_party/`에 있다. "수정 금지" 목록에 이 폴더, `xgrammar`, `nntrainer-windows-resource` 서브모듈 추가.
- `cpp_linter`는 clang-format 전에 `git diff --check`로 모든 변경 파일의 공백/EOF를 검사한다 (`.github/`, CausalLM third_party 제외).
- `clang-format-14 -i`는 파일 전체를 포맷해서 리뷰어가 거절하는 churn을 만든다 (#3813, #3560, #3212). `git clang-format-14 <base>`만 권장.

추가한 공통 규칙: 파일 위생 (EOF newline, trailing whitespace), doxygen 파일 헤더, dtype 분기 (모든 타입 처리 또는 throw, `ENABLE_FP16`), 설치 파일 변경 시 `debian/*.install` + `packaging/nntrainer.spec`, meson 옵션/플래그, 개인 경로 금지, `ml_log*`, 크기 곱셈 전 캐스트. 크로스플랫폼 함정: `_FP16`이 `nntrainer::Half` wrapper일 수 있음, `(std::min)`, Android fp16 기본값, Tizen armv7l. static.check 실제 항목과 "PR CI가 검사하지 않는 것" (CausalLM은 fp16-off leg에서만, Android unit test, OpenCL 런타임).

루트는 95줄에서 168줄이 됐다.

### 커밋 2: `e961fa7b7` [CausalLM] Add AGENTS.md with rules for agents editing the app

`Applications/CausalLM/AGENTS.md` (117줄) + 루트의 "Directory rules" 포인터 표. 규칙 8개:

1. weight 파일은 위치 기반: converter 쓰기 순서 = 레이어 생성 순서, `createMlp` (up, gate, down) 고정. 2026년에 두 번 revert (87f5fea19, 45330052b).
2. 새 소스는 빌드 목록 4곳에: in-tree meson, `app_build/meson.build`, `jni/Android.mk` (테스트 하네스), `models/model_registry.cpp`.
3. `incremental_forwarding(from, to)`는 절대 위치, step 텐서는 `pos - from`, prefill 판정 `!from || (to - from) > 1`.
4. dtype/arch 분기를 함께 유지. CI는 CausalLM을 x86 fp16-off에서만 테스트.
5. `parallel_for`는 한 단계만 (중첩하면 thread manager deadlock).
6. 공개 모델 이름만. `api/quick_dot_ai_api.h` enum은 append만.
7. `res/<model>/`에는 converter, `nntr_config.json`, README, 빌드 스크립트만.
8. `BATCH_SIZE` 존중 또는 batch > 1 명시 거부.

### 검증 중 바로잡은 서브에이전트 초안

- "CI가 CausalLM Android 빌드를 안 한다"는 틀림. `causallm_android.yml`이 `app_build/meson.build`를 빌드한다. 빌드 안 되는 건 `jni/Android.mk` (테스트 전용)뿐.
- "Android는 항상 fp16"을 "`tools/package_android.sh` 기본값이 fp16"으로 수정 (preset JSON으로 끌 수 있음).
- Windows 제외 가드: meson 쪽은 `_WIN32`가 아니라 `get_option('platform') != 'windows'`.

## 5. 확인 못 한 것

- **빌드/테스트 미실행.** worktree에서 서브모듈 고정 커밋을 받지 못했다 (`--depth 1`로는 고정 커밋 fetch 실패, xgrammar만 직접 fetch로 해결). 문서만 바꾸는 변경이라 `git diff --check`와 테스트 타깃 이름이 `meson.build`에 있는지만 확인했다. PR 본문 Self evaluation은 "Skipped (documentation only)".
- 규칙 파일의 Verify 명령은 실행해 보지 않은 제안이다.
- 커밋 87f5fea19 메시지에 따르면 Qwen2/Qwen3/LFM2 `FP32MatchesHFReference`는 fixture 재생성 전까지 실패한다. 이후 fixture 업데이트 커밋 없음. 미실행.

## 6. 덤으로 찾은 것

- upstream `.github/workflows/static.check.scripts/hardcoded-path.sh`는 **절대 실패하지 않는다**. `$VIOLATION`은 정의되지 않고 `[[ "failed" = "1" ]]`은 문자열 두 개를 비교한다. 이번 PR에서는 안 고치고 PR 본문 "Also found"에만 적었다. 별도 upstream fix 후보.

## 7. 다음 할 일

1. 사용자 지시 시 upstream PR 생성: `gh pr create -R nntrainer/nntrainer --head dlwlzzero:docs/agents-rules --base main --body-file ~/agents-rules-evidence/pr_body.md`. 본문은 `.github/PULL_REQUEST_TEMPLATE.md` 형식 (커밋별 `<details>`).
2. 메인테이너가 형식을 받아들이면 §3의 후속 파일 7개를 PR로 이어서 올린다. 루트 포인터 표에 행을 추가한다.
3. 후속 파일 작성 시: 파일당 50~80줄, 규칙은 긍정형으로 ("무엇을 하라"), 각 규칙에 이유 한 줄, 단위 끝에 검증 명령.

## 8. 자료 위치 (워크스테이션 로컬, 커밋 안 됨)

- `~/agents-rules-evidence/candidates.md`: 단위별 규칙 후보 통합본 (한국어)
- `~/agents-rules-evidence/rules_{apps,tensor,core,accel,cross}.md`: 규칙별 근거 (리뷰 코멘트 URL, 커밋 SHA, 리뷰어, 강도)
- `~/agents-rules-evidence/unit_*.json`: 단위별 원본 코멘트
- `~/agents-rules-evidence/BRIEF.md`: 서브에이전트에 준 분석 지침
- `~/agents-rules-evidence/pr_body.md`: PR 1 본문
