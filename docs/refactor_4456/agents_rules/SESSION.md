# AGENTS.md 규칙 파일 작업: 세션 기록

- 세션: 2026-11-05 ~ 11-06, Claude Code (Opus 5.5)와 사용자 대화
- 목적: nntrainer에서 AI 에이전트가 코드 작업할 때 따를 규칙을 md 파일로 만든다. 이 문서는 그 결정에 이른 **대화 내용과 기록**을 정리한 것이다.
- 산출물 위치:
  - 규칙 파일 자체: upstream main에서 새로 딴 브랜치 `dlwlzzero:docs/agents-rules` (커밋 `14eca6e9b`, `e961fa7b7`). upstream PR은 아직 안 열었다.
  - 이 기록: 이 폴더 (`docs/refactor_4456/agents_rules/`), 근거 자료는 [`records/`](records/)

---

## 1. 대화 흐름

### 1.1 사용자 계획 제시 → 검토

**사용자:**

> 이번 세션에서는 nntrainer에서 Agent가 코드 작업을 할 때 따라야할 규칙을 md 파일로 만드는 작업을 수행할 거임
> - nntrainer 코드베이스가 크니까 코드 구조를 우선 세분화해
> - 세분화한 코드 단위별로 따라야할 규칙 파일을 만들어
>
> 위의 계획 검토해줘

**검토 결과 (Claude):** 방향은 맞지만 **나누는 기준을 "코드 구조"가 아니라 "규칙이 실제로 달라지는 곳"으로** 바꾸자고 제안했다.

1. **차이(delta)만 담기.** 루트 `AGENTS.md`(upstream main에 이미 있음, 95줄)가 DCO, clang-format-14, 커밋 형식, `subprojects/` 금지 등 공통 규칙을 갖고 있다. 단위 파일에는 그 단위에서만 다른 규칙을 넣고, 차이가 없으면 파일을 만들지 않는다.
2. **코드에서 안 보이는 것만 담기.** 디렉토리 구조, 클래스 목록, 빌드 옵션은 에이전트가 직접 찾을 수 있고 적어두면 낡는다. 담을 것은 불변식, 반복되는 리뷰 지적, 플랫폼 함정.
3. **규칙은 증거에서 뽑기.** 코드만 읽고 만든 규칙은 "에러 처리 잘 해라" 같은 일반론(no-op)이 된다. 원천: 과거 PR 리뷰 코멘트, `fix`/`revert` 커밋 이력, 이미 겪은 함정.
4. **단위마다 검증 방법 한 줄.** "이 단위를 고쳤으면 이 테스트를 돌려라"가 있어야 에이전트가 끝을 판단한다.
5. **로딩 방식 먼저 결정.** Claude Code는 하위 `CLAUDE.md`를, Codex 계열은 하위 `AGENTS.md`를 읽는다.

단위 후보는 6~8개로 제안했다 (수십 개로 쪼개면 유지보수 불가). 파일당 50~80줄.

**결정 요청:** 결과물이 어느 브랜치로 가는지 (upstream main PR / fork 전용 / 둘 다)에 따라 담을 내용이 완전히 달라진다.

### 1.2 대상 결정: upstream main

**사용자:** upstream main에 push할 예정

**Claude:** upstream main (`640ac6e0e`)을 받아 실제 구조를 확인하고 계획을 세 가지 바꿨다.

1. **브랜치:** fork 작업 브랜치(`htp_decode`)가 아니라 **upstream main에서 새 worktree**. 브랜치 전용 `CLAUDE.md`, `docs/htp_*`가 딸려 가지 않게 하고, HTP/Gemma 실험 규칙(bit-identity, byte-identical 그래프 등)은 전부 뺀다.
2. **형식:** upstream에는 루트 `AGENTS.md`만 있고 `CLAUDE.md`가 없다. 그래서 **폴더별 `AGENTS.md` + 루트에 "X 아래를 고치기 전에 `X/AGENTS.md`를 읽어라" 포인터 표.** 어느 에이전트에서나 동작한다.
3. **단위:** upstream 실제 크기로 재조정. `htp_backend`는 upstream에 파일 5개뿐이라 독립 단위가 안 된다.

| 후보 단위 | 파일 수 (upstream main) |
|---|---|
| `Applications/CausalLM` | 293 |
| `nntrainer/layers` / `nntrainer/tensor/cpu_backend` | 178 / 174 |
| `nntrainer/tensor/cl_operations` + `nntrainer/opencl` | 125 |
| `test` / `api` / `compiler`+`graph` / `qnn`+`htp_backend` | 383 / 31 / 45 / 21 |

진행 순서 제안: (1) 근거 수집 → (2) 후보 목록을 사용자가 검토 → (3) 루트 포인터 + 단위 파일 1~2개로 **작은 첫 PR**, 반응 보고 나머지 → (4) 커밋 규칙 준수 (`[docs]`, `-s`, `Co-authored-by`, PR 템플릿).

### 1.3 근거 수집 (사용자: "진행해줘")

1. `gh api repos/nntrainer/nntrainer/pulls/comments`로 inline 리뷰 코멘트 4000개 수집 (API 상한, PR 784개).
2. 봇, PR 작성자 본인, 답글을 빼고 **사람 리뷰어의 최상위 코멘트 1,678개**를 경로별 단위로 분류.
3. 서브에이전트 5개를 병렬로 돌렸다. 공통 지침은 [`records/BRIEF.md`](records/BRIEF.md).
   - 규칙 후보 조건: 2개 이상 PR에서 반복 **또는** 메인테이너가 컨벤션으로 명시 / 코드 한 번 보고 알 수 없음 / 기본 동작을 바꿈 / upstream 범용.
   - 모든 후보는 upstream main 코드와 대조, 코드가 반박하면 탈락.

| 단위 | 사람 최상위 코멘트 | PR 수 |
|---|---|---|
| Applications/CausalLM | 287 | 105 |
| nntrainer/tensor/cpu_backend | 225 | 79 |
| nntrainer/tensor (코어) | 211 | 97 |
| nntrainer/layers | 182 | 79 |
| 기타 nntrainer/ | 161 | 60 |
| test | 135 | 67 |
| OpenCL | 125 | 53 |
| models / compiler+graph / api / qnn+htp | 82 / 49 / 26 / 4 | 36 / 27 / 16 / 1 |

**정정 (대화 중 사용자에게 보고):** 수집 기간은 처음 말한 6개월이 아니라 **2024-11 ~ 2026-11 (약 2년)**이었다. API가 생성일 순으로 정렬해 주지 않았다. 오래된 지적은 코드 대조에서 걸러지게 했다.

그룹별 결과 (도착 순서대로 사용자에게 보고):

- **accel** ([`records/rules_accel.md`](records/rules_accel.md)): OpenCL 6개 + 약한 1개, 파일 가치 있음. `nntrainer/qnn`은 2개 (리뷰 코멘트 4개가 모두 미머지 PR이라 커밋과 코드에서 뽑음). `htp_backend`은 근거 부족으로 skip. 가장 강한 것: SVM map/unmap 핸드오프, host dispatcher가 커널 shape 가정 검사, OpenCL은 `-Denable-opencl=true` + 실 GPU에서 테스트 (CI에 GPU 없음).
- **core** ([`records/rules_core.md`](records/rules_core.md)): layers 7, utils+engine 3, models 1 (다른 파일에 합침), compiler/graph 2 (애매), optimizers/dataset/mem_allocator 0. **루트와 충돌 발견:** 루트는 `clang-format-14 -i` (파일 전체)를 권하지만 리뷰어는 무관한 포맷 변경을 계속 거절 (#3813, #3560, #3212) → `git clang-format-14 <base>`로.
- **apps** ([`records/rules_apps.md`](records/rules_apps.md)): CausalLM 8개, 하위 폴더 전체를 파일 하나로. 기타 샘플 앱 1개 (skip). 가장 강한 것: weight 파일은 위치 기반 (2026년 revert 2회), 새 `.cpp`는 빌드 목록 4곳 (fix 커밋 5개).
- **cross** ([`records/rules_cross.md`](records/rules_cross.md)): test 6, api 3. **루트 `AGENTS.md`에 틀린 내용 9건:** `check_count`는 테스트 수가 아니라 승인 리뷰 수를 셈, `pr-desc`는 PR 본문을 검증하지 않음, minja는 `subprojects/`가 아니라 `Applications/CausalLM/third_party/`, format gate는 `git diff --check`로 모든 파일 공백 검사 등.
- **tensor** ([`records/rules_tensor.md`](records/rules_tensor.md)): cpu_backend 8, tensor 코어 4. arch 하위 폴더별 파일은 불필요 (규칙이 "세 백엔드를 함께 고쳐라"라서 나누면 서로 못 본다). tensor와 cpu_backend는 파일 두 개로.

### 1.4 후보 통합 → 첫 PR 범위 결정

다섯 그룹 결과를 [`records/candidates.md`](records/candidates.md)로 합쳤다. 여러 그룹에서 같은 규칙(EOF newline 14+ PR, doxygen 헤더, "바꿀 줄만 바꾸기", `ENABLE_FP16` 가드, 패키징 매니페스트)이 반복돼서 루트 공통 규칙으로 올렸다.

| 파일 | 규칙 |
|---|---|
| 루트 `AGENTS.md` | 정정 9건 + 공통 규칙 10개 + 포인터 표 |
| `Applications/CausalLM/` | 8 |
| `nntrainer/tensor/cpu_backend/` / `nntrainer/tensor/` | 7 / 4 |
| `nntrainer/opencl/` (cl_operations 포함) / `nntrainer/layers/` | 6 / 7 |
| `nntrainer/` / `test/` / `api/` | 7 / 7 / 3 |

만들지 않는 것: `htp_backend`/`qnn` 별도 파일, 기타 샘플 앱, `docs/`, `.github/`.

첫 PR 범위 선택지로 (추천) 루트 수정 + CausalLM 1개 / 루트만 / 전부 한 번에를 제시했다. 추천 이유: 루트 정정은 틀린 정보를 고치는 것이라 거절 위험이 가장 낮고, CausalLM은 근거가 가장 강하다.

### 1.5 PR 1 작성 (사용자: "추천안으로 진행해줘")

1. upstream main `640ac6e0e`에서 worktree `~/nntrainer-agents-rules`, 브랜치 `docs/agents-rules`.
2. 루트 정정 사항을 워크플로 파일에서 **직접 다시 확인**: `check_count.yml` (APPROVED 리뷰어 수, `pull_request_review` 트리거), `pr-desc.yml` (`ai-describe` 라벨, "never edit PR body"), `cpp_linter.yml` (`git diff --check`, `.github`/third_party 제외), `ubuntu_clean_meson_build.yml` (`required_tests`, CausalLM은 fp16-off leg만), `static.check.yml` (30 MiB 등), `.gitmodules`, `hardcoded-path.sh`.
3. 확인하면서 서브에이전트 초안 3곳을 바로잡았다.
   - "CI가 CausalLM Android 빌드를 안 한다" → 틀림. `causallm_android.yml`이 `app_build/meson.build`를 빌드한다. ndk-build가 안 되는 건 `jni/Android.mk` (테스트 전용 하네스)뿐.
   - "Android는 항상 fp16" → "`tools/package_android.sh` 기본값이 fp16" (preset JSON으로 끌 수 있음).
   - Windows 제외 가드: meson 쪽은 `_WIN32`가 아니라 `get_option('platform') != 'windows'`.
4. 커밋 2개 (DCO `-s`, `Co-Authored-By` 트레일러):
   - `14eca6e9b` **[docs] AGENTS.md: correct the CI facts and add rules for every change** — 정정 5건 (`check_count`, `pr-desc`, minja 위치, `git diff --check`, 파일 전체 clang-format), 공통 규칙 8개, 크로스플랫폼 함정, static.check 실제 항목, "PR CI가 검사하지 않는 것". 95줄 → 168줄.
   - `e961fa7b7` **[CausalLM] Add AGENTS.md with rules for agents editing the app** — `Applications/CausalLM/AGENTS.md` 117줄 (규칙 8개) + 루트 "Directory rules" 포인터 표.
5. PR 본문은 `.github/PULL_REQUEST_TEMPLATE.md` 형식으로 작성: [`records/pr_body.md`](records/pr_body.md).

### 1.6 push (사용자: "우선 fork에 push만, PR 생성은 나중에")

`docs/agents-rules`를 fork `dlwlzzero/nntrainer`에 push. 원격 head `e961fa7b7` 확인. **upstream PR은 열지 않았다.**

### 1.7 기록 정리 (이 문서)

첫 시도에서는 현황 노트 형태로 `docs/refactor_4456/AGENTS_RULES.md`를 올렸다 (`bff0d5828`). 사용자가 "AGENTS.md는 main에서 새 브랜치로 작업하는 것이고, 이 브랜치에는 대화 내용과 기록을 정리한 md를 넣어달라"고 다시 짚어서, 대화 기록 중심으로 이 문서를 다시 쓰고 근거 자료를 [`records/`](records/)에 함께 넣었다.

---

## 2. 확인 못 한 것

- **빌드/테스트 미실행.** worktree에서 서브모듈 고정 커밋을 받지 못했다 (`--depth 1`로 고정 커밋 fetch 실패, xgrammar만 직접 fetch로 해결). 문서만 바꾸는 변경이라 `git diff --check`와 테스트 타깃 이름이 `meson.build`에 있는지만 확인했다. PR 본문 Self evaluation은 "Skipped (documentation only)".
- 규칙 파일과 `records/rules_*.md`의 Verify 명령은 실행해 보지 않은 제안이다.
- 커밋 87f5fea19 메시지에 따르면 Qwen2/Qwen3/LFM2 `FP32MatchesHFReference`는 fixture 재생성 전까지 실패한다. 이후 fixture 업데이트 커밋 없음. 미실행.

## 3. 덤으로 찾은 것

upstream `.github/workflows/static.check.scripts/hardcoded-path.sh`는 **절대 실패하지 않는다**. `$VIOLATION`은 정의되지 않고 `[[ "failed" = "1" ]]`은 문자열 두 개를 비교한다. PR 1에서는 안 고치고 PR 본문 "Also found"에만 적었다. 별도 upstream fix 후보.

## 4. 다음 할 일

1. 사용자 지시 시 upstream PR 생성:
   `gh pr create -R nntrainer/nntrainer --head dlwlzzero:docs/agents-rules --base main --title "..." --body-file docs/refactor_4456/agents_rules/records/pr_body.md`
2. 메인테이너가 형식을 받아들이면 §1.4 표의 후속 파일 7개를 PR로 이어서 올리고, 루트 포인터 표에 행을 추가한다. 내용은 [`records/candidates.md`](records/candidates.md), 근거 링크는 `records/rules_*.md`.
3. 후속 파일 작성 원칙: 파일당 50~80줄, 규칙은 긍정형 ("무엇을 하라"), 규칙마다 이유 한 줄, 단위 끝에 검증 명령. 작성 전에 해당 규칙을 upstream main 최신 코드와 다시 대조한다 (이 기록은 `640ac6e0e` 기준).

## 5. 기록 파일

| 파일 | 내용 |
|---|---|
| [`records/BRIEF.md`](records/BRIEF.md) | 서브에이전트 5개에 준 분석 지침 (입력, 방법, 규칙 조건, 출력 형식). 안의 `/tmp/...` 경로는 세션 당시 scratchpad |
| [`records/candidates.md`](records/candidates.md) | 단위별 규칙 후보 통합본 (한국어), 루트 정정 9건 |
| [`records/rules_apps.md`](records/rules_apps.md) | CausalLM, 기타 앱 |
| [`records/rules_tensor.md`](records/rules_tensor.md) | cpu_backend, tensor 코어 |
| [`records/rules_core.md`](records/rules_core.md) | layers, models, compiler/graph, 기타 nntrainer |
| [`records/rules_accel.md`](records/rules_accel.md) | OpenCL, QNN/HTP |
| [`records/rules_cross.md`](records/rules_cross.md) | test, api, CI, 빌드/패키징, 루트 정정 |
| [`records/pr_body.md`](records/pr_body.md) | PR 1 본문 (템플릿 형식) |

`rules_*.md`는 규칙마다 Rule / Why / Evidence (리뷰 코멘트 URL, 커밋 SHA, 리뷰어) / Strength / Verify를 담고 있다.

원본 코멘트 JSON (`unit_*.json`, 약 750 KB)은 커밋하지 않았다. 다시 만들려면:

```bash
for p in $(seq 1 40); do
  gh api "repos/nntrainer/nntrainer/pulls/comments?sort=created&direction=desc&per_page=100&page=$p" > rc_$p.json
  [ "$(jq length rc_$p.json)" -lt 100 ] && break
done
jq -s add rc_*.json > review_comments.json
# PR 작성자: gh api repos/nntrainer/nntrainer/pulls/<n> --jq .user.login
# 필터: user가 [bot]이 아니고, in_reply_to_id가 없고, PR 작성자가 아닌 코멘트
```
