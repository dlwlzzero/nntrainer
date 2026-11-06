# 백엔드 구조 지도: 현재 구조 · 목표 구조 · 남은 일 (세션 기록)

- 결과물: [ARCH_MAP_PAGE.md](ARCH_MAP_PAGE.md) (HTML 페이지 링크와 버전), [ARCH_MAP_PAGE.pdf](ARCH_MAP_PAGE.pdf) (PDF 사본)
- 기준 tree: `qc_npu_rebase` @ `d061fcd20` (upstream main `4cc8dd5cd` + qc_npu 커밋 720개)
- 목표 문서: `docs/backend_guide/ARCHITECTURE_REFACTOR.md` (사용자가 "Refactoring.md"라고 부른 문서)
- 기간: 2026-11-06 07:40 UTC (11-06 16:40 KST) ~ 2026-11-06 10:51 UTC (11-06 19:51 KST)

이 문서는 nntrainer의 현재 백엔드 구조와 목표 구조를 HTML 페이지로 정리하고, 목표 구조를 다듬은 과정을 **사용자 프롬프트와 AI 응답을 시간순으로** 옮긴 것이다.

- 출처: 이 세션의 대화 기록. 같은 세션 앞부분의 nntrainer_dev main 정리와 `qc_npu_rebase` 브랜치 생성은 refactoring 논의가 아니라서 뺐다.
- 사용자 프롬프트와 AI의 마지막 응답은 원문 그대로다. 이메일 주소만 가렸다.
- AI가 작업하면서 중간에 남긴 짧은 메시지는 "작업 중 메시지"로 접어 두었다. 도구 호출 내용과 결과는 넣지 않았다.
- 작업 중 메시지 일부는 대화 기록 파일에 남지 않아 빠졌다. 사용자 질문("언제쯤 끝나니?")에 대한 답 하나만 대화 화면 내용으로 보완하고 그렇게 표시했다.
- 로컬 명령(`/login`, `/remote-control`)과 알림은 사용자 입력이 아니어서 따로 표시했다.
- 시각은 UTC이고 괄호 안은 KST다.

> 10:12 UTC에 다른 계정으로 다시 로그인했다. 그 뒤로 첫 링크(https://claude.ai/artifact/99ZP8UjFiQUp1TZQiDxR4f)는 이 세션에서 고칠 수 없어서, 10:27 UTC부터 새 링크(https://claude.ai/artifact/7zoytDaUeHhrRLJA6AsPTa)에 게시했다. 같은 때 작업 폴더가 비워져 페이지 원본이 지워졌고, 대화 기록에 남은 파일 작성 · 수정 내역을 순서대로 다시 적용해 복원했다.

## 결정 요약

| # | 결정 | 나온 곳 |
|---|---|---|
| 1 | 목표 구조의 기준은 `ARCHITECTURE_REFACTOR.md`. 저장소에 `Refactoring.md`라는 파일은 없다 | 1 |
| 2 | layer는 없애지 않고 **계산만 뺀다.** layer에는 설계도(속성, shape, weight, 상태, 학습 설정, `lower()`)만 남고, 계산은 compile 때 만든 op 목록이 한다 | 7~9 |
| 3 | layer 설계도와 Plan은 맡는 질문이 다르다(무엇을 vs 어디서 · 어떻게). `lower()`는 하드웨어를 모르고, Plan은 shape · weight를 정하지 않는다. 겹치는 곳은 ModelFeatures이고, op 목록에서 자동으로 뽑는 쪽을 제안 | 9 |
| 4 | backward도 op 목록으로 옮긴다. 다만 forward 다음인 **2단계**이고, 방식은 op 종류별 규칙표다. backward가 없는 backend(HTP, QNN)는 Plan이 학습 모드를 거절한다 | 11~13 |
| 5 | prefill과 decode는 **한 경로**(`runStep`)다. 실행 방식(eager, capture-replay, offload)은 Plan이 step 모양(M)을 보고 고른다 | 14~16 |

2~5는 문서 §9 결정 2(얇은 layer가 forwarding에서 ComputeOps를 부름)와 SEAM-2(decode hook)보다 한 걸음 더 나간 안이다. upstream에 가져가려면 그 결정을 바꾸는 제안으로 내야 한다.

## 목차

| # | 시각 (UTC) | 입력 | 내용 |
|---|---|---|---|
| 1 | 11-06 07:40 | 사용자 | nntrainer-qc-rebase 기준으로 아래 내용들 정리해서 html 파일로 시각화해서 보여줘 - … |
| 2 | 11-06 07:57 | 사용자 | 아래 내용도 추가해줘 - 현재 전체 코드베이스의 class diagram도 그려줘 - 목표한 아키텍쳐에 … |
| 3 | 11-06 08:07 | 사용자 | 지금 class diagram이 너무 커서 전체가 한눈에 안 보임 해결해 |
| 4 | 11-06 08:36 | 사용자 | 그리고 목표 구조 쪽에 있는 diagram을 현재 구조쪽에 있는 diagram처럼 각 backend별로 … |
| 5 | 11-06 10:11 | 로컬 명령/알림 | 10:11 UTC · 이전 세션의 백그라운드 명령(`bnuei73p0`)이 중단됐다는 알림; 10:12 … |
| 6 | 11-06 10:13 | 사용자 | ?? |
| 7 | 11-06 10:14 | 사용자 | 목표 구조에 대해서 layer가 없어져야 한다고 생각하는데, 무작정 동의하지 말고 한 번 검토해봐 |
| 8 | 11-06 10:19 | 사용자 | 잘 이해가 안 됨 좀 더 쉽게 설명해줘 |
| 9 | 11-06 10:21 | 사용자 | 2번 뜻이었어 페이지에 반영해줘 그리고 2번 뜻이라면 목표 구조에서 layer에 뭐가 포함되는지 설명해줘 |
| 10 | 11-06 10:30 | 로컬 명령/알림 | 10:30 UTC · `/remote-control` → `Remote Control disconnect… |
| 11 | 11-06 10:31 | 사용자 | 근데 forward랑 backward 쪽이랑 무슨 구조 차이가 있길래 layer를 못 없앰? 내가 잘못 … |
| 12 | 11-06 10:34 | 사용자 | backward도 op로 만들어야하지 않을까? 무작정 동의하지 말고 검토해줘 |
| 13 | 11-06 10:38 | 사용자 | 지금 내용을 페이지에 반영하고 필요하다면 목표 구조 이미지도 업데이트해줘 |
| 14 | 11-06 10:45 | 사용자 | 목표 구조에서 Decode가 왜 따로 분류되어 있는 이유가 있음? Prefill은 없는데 말이야 |
| 15 | 11-06 10:47 | 사용자 | 둘이 왜 분리되어야 함? 그냥 통합되어 있으면 안 됨? |
| 16 | 11-06 10:49 | 사용자 | 응 그렇게 페이지에 반영해줘 |
| 17 | 11-06 10:52 | 사용자 | 아래의 지시대로 2개의 md 파일 만들어서 https://github.com/dlwlzzero/nntra… |

---

## 프롬프트와 응답 (시간순)

### 1. 2026-11-06 07:40 UTC (11-06 16:40 KST) · 사용자 프롬프트

> nntrainer-qc-rebase 기준으로 아래 내용들 정리해서 html 파일로 시각화해서 보여줘
> - 설명은 사람이 이해하기 쉽게 너무 verbose하지 않게 설명해야 함
> - 우선 현재 architecture 구조가 어떻게 되어있는지 시각화해서 설명해줘
> - Refactoring.md에서 지향하고 있는 nntrainer 구조가 적용되었을 때의 architecture 구조에 대해서 시각화해서 설명해줘
> - 두 구조를 비교했을 때, 앞으로 해결해야할 부분이 뭔지 설명해

> 참고: 이 턴에서 AI가 그림 작성 지침(artifact-diagramming skill)을 불러왔다. 스킬 본문은 사용자 입력이 아니라서 뺐다.

<details><summary>작업 중 메시지 1개</summary>

> Grep으로 확인을 끝냈고, 이제 HTML 페이지를 작성해서 게시합니다.

</details>

**AI 응답:**

> **정리한 페이지를 게시했습니다: https://claude.ai/artifact/99ZP8UjFiQUp1TZQiDxR4f**
>
> **→ 기준 문서:** 저장소에 `Refactoring.md`라는 이름의 파일은 없습니다. 그래서 `docs/backend_guide/ARCHITECTURE_REFACTOR.md`를 그 문서로 보고 작성했습니다. 다른 파일을 말씀하신 거면 알려 주세요.
>
> **→ 구성:** 그림 4개로 되어 있습니다.
> - 현재 구조: 호출 순서 × backend 5개 matrix
> - 목표 구조: L5~L0 층 구조
> - 결정 흐름: 지금과 목표 비교
> - 남은 일: 의존 관계
>
> **→ 현재 구조:** 아래쪽 사슬(Context → ComputeOps → MemAllocator → kernel)은 이미 목표 모양입니다. 문제는 위쪽입니다. layer fork가 **11개** 있고(cl 5 · gpu 2 · cuda 1 · htp 3), HTP는 **모델 class 자체가 따로**(Gemma4/LFM2 HTP 전용) 있습니다.
>
> **→ 가장 큰 차이:** `resolveExecPlan`은 계산은 하지만 **로그만 찍고 아무도 읽지 않습니다.** 실제 kernel, 메모리, decode 경로는 **env 변수 275개**와 `#if`가 정합니다. 275개 중 260개는 upstream에서 온 것이고, qc_npu가 15개를 더했습니다.
>
> **→ 남은 일, 제안 순서:** 문서 자체에는 순서가 없어서 의존 관계를 보고 정한 제 제안입니다.
> 1. 이름 비교 2곳과 downcast 4곳 정리. 작고 다른 일과 얽히지 않습니다.
> 2. `rmsnorm` · `attention` whole-op virtual 추가. 이게 들어가야 fork 11개를 접을 수 있습니다.
> 3. HtpContext에 `caps()`를 붙이고, env 변수를 caps 값으로 옮겨서 ExecPlan이 실제로 결정하게 만들기.
> 4. 마지막으로 HTP를 공통 graph로 합치기.
>
> **→ 문서 내용과 다른 점:** 문서 §10은 AppContext downcast를 **1곳**으로 적었습니다. 실제로는 upstream main에 이미 2곳이 있고, qc_npu_rebase에는 **4곳**이 있습니다.
>
> **→ 미확인:** 숫자는 전부 `d061fcd20` tree를 grep해서 센 값입니다. 빌드나 실행으로 확인한 내용은 없습니다. 페이지 렌더링도 미리보기 없이 게시해서, 깨진 곳이 있으면 말씀해 주시면 고치겠습니다.
>
> 페이지는 비공개입니다. 다른 분께 보여 주려면 페이지의 Share 메뉴에서 공유해야 합니다.

**결과:** 페이지 첫 게시: https://claude.ai/artifact/99ZP8UjFiQUp1TZQiDxR4f (Version 1)

---

### 2. 2026-11-06 07:57 UTC (11-06 16:57 KST) · 사용자 프롬프트

> 아래 내용도 추가해줘
> - 현재 전체 코드베이스의 class diagram도 그려줘
> - 목표한 아키텍쳐에 대한 class diagram도 그려줘 
> - 그리고 HTP뿐만 아니라 CPU/GPU에 대해서도 구조 안 맞는 부분 다 설명해줘

<details><summary>작업 중 메시지 3개</summary>

> 페이지에 세 가지를 추가합니다: 현재 class diagram, 목표 class diagram, CPU/GPU/CUDA까지 포함한 구조 불일치 전체 목록. 먼저 class 관계를 코드에서 뽑겠습니다.

> class 선언에 `WIN_EXPORT` 접두어가 붙어 있어서 패턴을 바꿔 다시 뽑는 중입니다.

> Class 관계와 backend별 근거를 다 모았습니다. 페이지에 class diagram 4개(현재 3, 목표 1)와 backend별 불일치 목록을 넣고 다시 게시합니다.

</details>

**AI 응답:**

> **같은 링크에 세 가지를 추가했습니다: https://claude.ai/artifact/99ZP8UjFiQUp1TZQiDxR4f**
>
> **→ 현재 class diagram (그림 2a~2c):** 코드베이스 전체 class는 수백 개라서, backend 구조에 관련된 class만 세 장으로 나눴습니다.
> - 2a 실행 사슬: Engine, Context, ComputeOps, MemAllocator
> - 2b graph · layer: fork class 포함
> - 2c 모델: CausalLM 계열
>
> 2b에는 "같은 type 이름을 backend마다 다른 class가 등록"하는 관계를 점선으로 표시했습니다. 예를 들어 RMSNorm 하나에 class가 4개입니다.
>
> **→ 목표 class diagram (그림 4):** 문서 §3을 기준으로 그렸습니다. 아래에 사라지는 class, core로 올라오는 class, 새로 생기는 것을 칩으로 정리했습니다. HTP decode를 `runDecode`의 offload mode 뒤에 둔 부분은 문서에 없는 제 해석이고, 페이지에도 그렇게 적었습니다.
>
> **→ backend별 불일치:** 공통/CPU 9건, OpenCL 6건, CUDA 4건, HTP 8건, QNN 3건입니다. 각 항목에 어긴 문서 조항과 근거 파일을 달았고, backend마다 이미 목표 모양인 부분도 한 줄씩 넣었습니다. HTP 외에 눈에 띄는 것:
> - **RMSNorm 구현이 5벌**이고, 그중 op table을 부르는 것은 하나도 없습니다.
> - **CUDA가 OpenCL class(`FullyConnectedLayerCl`)를 FC로 빌려 씁니다.**
> - **gemma2와 gemma4가 모델 graph에 `gpu_decode_attn` 같은 backend 이름 속성을 넣습니다.**
> - **layer가 process 전역 engine 문자열을 직접 읽어 분기합니다.** 예: `nntr_engine_env() == "cuda"`.
>
> **→ 새로 찾은 HTP 문제 2건:**
> - 중립이어야 할 core의 `Fp16Attention`이 `HtpAttentionState`와 `beginHtpPrompt()` 같은 HTP 전용 member를 직접 갖고 있습니다.
> - ComputeOps virtual이 upstream 104개에서 **161개**로 늘었고, 그중 상당수가 연산이 아니라 weight 등록·해제 같은 자원 관리 함수입니다.
>
> **→ 미확인:** class diagram은 Mermaid로 그렸는데, 로컬에 Mermaid가 없어서 렌더링을 확인하지 못했습니다. 문법 오류가 있으면 그 그림만 소스 텍스트로 보일 수 있으니, 깨진 그림이 있으면 알려 주세요.
>
> **→ 공유 상태:** 이 페이지는 지금 "링크가 있는 누구나" 볼 수 있습니다. 내부 브랜치 내용이 들어 있으니 의도한 설정인지 확인해 주세요.

**결과:** https://claude.ai/artifact/99ZP8UjFiQUp1TZQiDxR4f Version 2 (Mermaid class diagram 4개, backend별 불일치)

---

### 3. 2026-11-06 08:07 UTC (11-06 17:07 KST) · 사용자 프롬프트

> 지금 class diagram이 너무 커서 전체가 한눈에 안 보임 해결해

> 참고: 08:15 UTC에 API 연결이 끊겨 응답이 끝나지 못했다. 작업은 다음 턴에서 이어졌다.

**AI 응답 (중단됨):**

> API Error: Connection lost mid-response. The response above may be incomplete.

---

### 4. 2026-11-06 08:36 UTC (11-06 17:36 KST) · 사용자 프롬프트

> 그리고 목표 구조 쪽에 있는 diagram을 현재 구조쪽에 있는 diagram처럼 각 backend별로 표현해줘.
>
> 언제쯤 끝나니?

- 알림 (사용자 입력 아님): 08:19 UTC · Artifact 감시가 연결 끊김으로 종료됐다는 알림 (이 턴 중간에 전달됨)

<details><summary>작업 중 메시지 1개</summary>

> **새 요청(목표 구조도 backend별 표로)까지 반영해서 10~15분 안에 끝납니다.**
>
> 진행 순서는 이렇습니다:
> - class diagram 4개를 각각 화면 한 장에 들어가는 크기의 SVG로 다시 그립니다.
> - 목표 구조에 그림 1과 같은 모양의 backend × 층 표를 추가합니다.
> - 로컬 Chrome으로 한 번 렌더링해서 확인하고 게시합니다.

> (이 메시지는 대화 기록 파일에 남지 않아 대화 화면의 내용으로 보완했다.)

</details>

**AI 응답:**

> **두 가지 모두 반영해서 같은 링크에 다시 게시했습니다: https://claude.ai/artifact/99ZP8UjFiQUp1TZQiDxR4f**
>
> **→ Class diagram 크기:** Mermaid는 배치를 제어할 수 없어서 4개 모두 직접 배치한 SVG로 다시 그렸습니다.
> - 각 그림이 데스크톱에서 화면 한 장(높이 약 340~640px) 안에 다 들어갑니다.
> - 그림 2b(layer)는 상속 트리 대신 "op × backend" 표 형태로 바꿨습니다. 한 행에 상자가 여러 개면 그 op가 backend마다 fork된 것입니다.
> - 목표 class diagram에서는 구현 class(Cpu · Cl · Cuda · Htp)를 상자 안 "구현:" 한 줄로 줄였습니다. 지금과 똑같은 부분이라서 줄였고, 상자 수가 줄어 한눈에 보입니다.
>
> **→ 목표 구조를 backend별로 (새 그림 3):** 그림 1과 같은 backend × 층 표입니다. 파란 칸이 지금과 달라지는 곳입니다.
> - **위쪽 3줄(모델 · 실행 계획 · layer)**은 backend 구분 없이 하나로 합쳐집니다.
> - **backend 차이는 경계선 아래에만 남습니다:** Context, ComputeOps, 메모리, decode mode, kernel 줄입니다.
>
> **→ 그림 번호:** 목표 구조 쪽 그림이 하나 늘어서 그 뒤 번호가 하나씩 밀렸습니다. 지금은 그림 1~7입니다.
>
> **→ 렌더링 확인:** 로컬 headless Chrome으로 라이트 모드에서 한 번 렌더링해 겹침이나 잘림이 없는 걸 봤습니다. 다크 모드와 폰 화면은 확인하지 않았습니다. 폰에서는 그림이 가로 스크롤됩니다.
>
> **→ 확인 필요:**
> - HTP 메모리 칸의 "rpcmem MemAllocator"는 문서 §5 4단계를 따라 제가 붙인 가칭입니다.
> - 이 페이지는 아직 "링크가 있는 누구나" 볼 수 있는 공유 상태입니다.

**결과:** https://claude.ai/artifact/99ZP8UjFiQUp1TZQiDxR4f Version 3 (class diagram을 SVG로 다시 그림, 목표 구조 backend별 표 추가)

---

### 5. 2026-11-06 10:11 UTC (11-06 19:11 KST) · 로컬 명령과 알림 (사용자 프롬프트 아님)

- 10:11 UTC · 이전 세션의 백그라운드 명령(`bnuei73p0`)이 중단됐다는 알림
- 10:12 UTC · `/login` → `Login successful` (다른 계정으로 로그인)
- 10:12 UTC · `/remote-control` → 활성화

---

### 6. 2026-11-06 10:13 UTC (11-06 19:13 KST) · 사용자 프롬프트

> ??

**AI 응답:**

> **마지막으로 요청하신 작업은 끝났고, 지금 기다리는 작업은 없습니다.**
>
> **→ 완료된 것:** class diagram 4개를 화면 한 장에 들어가게 다시 그렸고, 목표 구조에 backend별 표(그림 3)를 추가했습니다. 둘 다 Version 3으로 게시했습니다: https://claude.ai/artifact/99ZP8UjFiQUp1TZQiDxR4f
>
> **→ 계정이 바뀌었습니다:** `/login` 뒤로 다른 계정([이메일 생략])으로 로그인돼 있습니다. 위 페이지는 이전 계정(dlwlzzero)으로 만든 것이라, 지금 계정에서는 열리지 않을 수 있습니다. 이 세션에서 그 페이지를 다시 수정해 게시하는 것도 실패할 수 있습니다. 링크는 "링크가 있는 누구나"로 공유돼 있어서 브라우저에서 보는 것 자체는 될 가능성이 높습니다.
>
> **→ Artifact 서버가 지금 연결 안 됨:** `api.anthropic.com` 주소를 못 찾는(DNS) 오류가 나고 있습니다. 네트워크가 돌아오기 전까지는 페이지를 다시 게시할 수 없습니다. 페이지 원본 파일은 로컬에 그대로 있습니다: `/tmp/claude-1002/-home-j2z0-lee-nntrainer/698a41a0-34fb-4f9d-9100-1f5f67cb2218/scratchpad/nntrainer-arch-map.html`
>
> **→ `/remote-control`:** 출력이 비어 있어서 연결됐는지는 여기서 알 수 없습니다.
>
> "??"가 어떤 뜻이었나요? 위 내용 중 하나에 대한 질문이면 그걸 말씀해 주세요. 페이지를 지금 계정으로 다시 게시하길 원하시면, 네트워크가 돌아온 뒤 새 링크로 올리겠습니다.

---

### 7. 2026-11-06 10:14 UTC (11-06 19:14 KST) · 사용자 프롬프트

> 목표 구조에 대해서 layer가 없어져야 한다고 생각하는데, 무작정 동의하지 말고 한 번 검토해봐

**AI 응답:**

> **layer를 "backend별 실행 단위"로서 없애는 건 맞습니다. 하지만 layer라는 개념 자체를 없애는 건 nntrainer에서는 반대합니다.** 둘을 구분해서 보면 이렇습니다.
>
> ### 맞는 부분
>
> **1 → 목표 구조의 layer는 이미 껍데기에 가깝습니다.** 목표대로라면 `RMSNormLayer::forwarding()`은 `ops->rmsnorm()` 한 줄입니다. 그런데도 op마다 class를 하나씩 만들면 property 파싱과 shape 계산 같은 반복 코드만 쌓입니다.
>
> **2 → NPU decode는 이미 layer 없이 돕니다.** HTP는 첫 token이 나온 뒤에는 layer를 거치지 않습니다. 대신 op 11종(`RMSNORM`, `FC`, `ROPE`, `ATTN_M1`, `MOE` …)으로 된 DSP 상주 graph로 돕니다. QNN도 graph를 통째로 실행합니다.
>
> **3 → fork 11개는 layer가 backend 등록 단위라서 생겼습니다.** `Context::createLayer`가 layer를 backend마다 따로 등록합니다. layer가 backend를 모르게 만들면 fork는 구조적으로 생길 수 없습니다.
>
> ### 반대하는 부분: layer가 맡은 일 중 대신 할 곳이 없는 것
>
> **4 → Training (on-device training이 nntrainer 본업):** backward는 layer에 있습니다. core layer 75개가 `calcDerivative`를 구현합니다. 반면 ComputeOps에는 backward용 virtual이 **0개**입니다. layer를 없애면 op마다 forward, backward, shape 정의를 다시 만들어야 하고, 그 결과물은 결국 이름만 바뀐 layer입니다.
> - 이 근거가 약해지는 경우도 있습니다. CausalLM layer는 대부분 backward가 `throw`이거나 빈 함수입니다. 그래서 **LLM 추론만 보면** 이 근거는 약합니다.
>
> **5 → 메모리 계획:** 출력 shape, weight, 임시 tensor와 그 수명은 `finalize(InitLayerContext)`가 선언합니다. memory planner는 이 선언을 보고 메모리를 배치합니다. 상태가 없는 kernel table인 ComputeOps는 이 정보를 갖고 있지 않습니다.
>
> **6 → Weight 이름과 로딩:** weight 파일은 layer 순서대로 읽습니다. HTP decode도 weight를 `"<layer>:<weight>"` 이름으로 찾습니다(`HtpModelBinding`).
>
> **7 → 호출 사이에 남는 상태:** KV cache, sliding-window ring, LFM2 conv 상태, MoE expert cache는 모두 layer 객체가 들고 있습니다.
>
> **8 → Public API와 문서:** layer를 없애면 `ml::train::Layer`, `createLayer(type, props)`, INI 모델 파일, 다른 Applications(MNIST, ResNet 등)가 모두 깨집니다. 문서 §9 결정 2도 "Layer는 얇게 두고 연산은 op table에"를 이미 확정된 결정으로 기록하고 있습니다.
>
> ### 진짜 문제는 "graph가 두 벌"이라는 점
>
> **9 → HTP 모델은 같은 모델을 두 번 기술합니다.** prefill용 layer graph가 있고, 그와 별도로 decode용 op 목록을 손으로 짭니다. `htp_graph_gemma_build()`와 `htp_graph_lfm2_build()`에 hidden, n_heads, window 같은 shape를 직접 넣어서 만듭니다. 게다가 이 모델별 builder는 backend 파일인 `htp_graph_desc.h` 안에 있습니다. "layer를 없애자"는 직관이 실제로 겨냥해야 할 건 이 중복이라고 봅니다.
>
> ### 제안: layer는 정의 단위로만 남기고 실행에서는 빼기
>
> - **compile 때 layer graph를 op 목록으로 변환:** 실행은 ExecPlan의 op 목록이 합니다. 그러면 HTP decode나 QNN 같은 offload backend는 실행 중에 layer를 전혀 거치지 않습니다.
> - **decode op 목록 자동 생성:** 각 layer가 자기 op 설명(op 종류, weight 이름, 속성)만 내놓으면 됩니다. 그러면 모델별로 손으로 짠 decode builder가 사라집니다.
> - **단순 op는 범용 layer 하나로:** rmsnorm, add, activation 같은 것은 범용 `OpLayer` 하나로 묶습니다. "layer class 수가 줄어야 한다"는 직관은 이걸로 반영됩니다.
> - **Training은 그대로:** 지금처럼 layer의 backward를 씁니다. op 목록 변환은 추론 graph에만 적용합니다.
>
> 이 방향은 문서와 충돌하지 않습니다. 문서도 whole-graph offload를 "compile 단계의 graph 재작성"으로 정의하고 있어서, 그걸 일반화한 것에 가깝습니다. 페이지에 반영한다면 목표 표의 "Layer" 줄은 "op 정의 (compile 때만)"로, 그 아래에 "ExecPlan op 목록" 실행 줄을 새로 넣게 됩니다.
>
> 말씀하신 "layer가 없어져야 한다"가 위 제안(실행에서만 빼기)과 같은 뜻인지, 아니면 training까지 포함해서 layer 개념을 아예 없애자는 뜻인지 알려 주세요. 앞쪽이면 이 내용으로 페이지를 고치겠습니다. 뒤쪽이면 backward를 어디로 옮길지부터 같이 정해야 합니다.

---

### 8. 2026-11-06 10:19 UTC (11-06 19:19 KST) · 사용자 프롬프트

> 잘 이해가 안 됨 좀 더 쉽게 설명해줘

**AI 응답:**

> **layer는 두 가지 일을 하는데, "직접 계산하는 일"은 빼는 게 맞고 "모델이 어떻게 생겼는지 적어두는 일"은 남겨야 합니다.**
>
> **→ layer가 지금 하는 일 두 가지**
> 1. **설계도:** 이 층의 weight가 무엇인지, 입출력 크기, 학습 때의 역계산(backward), KV cache 같은 기억을 적어 둡니다.
> 2. **계산:** 실제 숫자 계산을 CPU, GPU, NPU에서 돌립니다.
>
> **→ 2번(계산)은 빼야 합니다. 여기까지는 같은 의견입니다.** 계산을 layer가 직접 하다 보니 backend마다 layer를 복사했습니다. RMSNorm 하나에 class가 4개인 게 그 결과입니다. 계산은 backend별 계산 담당인 ComputeOps로 넘기면 됩니다.
>
> **→ 1번(설계도)은 없앨 수 없습니다.** 설계도가 없으면 weight를 어디서 읽을지, 메모리를 얼마나 잡을지, 학습을 어떻게 할지, KV cache를 누가 들고 있을지를 아무도 모릅니다. 지우면 결국 다른 이름으로 다시 만들게 됩니다.
>
> **→ 요리에 비유하면:**
> - layer는 지금 "레시피 카드 + 요리사"를 겸하고 있고, 주방(CPU/GPU/NPU)마다 레시피 카드를 복사해서 고쳐 쓰고 있습니다.
> - 목표는 **레시피 카드는 하나만 두고, 요리는 주방마다 있는 요리사(ComputeOps)가** 하는 겁니다.
> - 레시피 카드까지 버리면 재료(weight)와 분량(크기)을 알 방법이 없어집니다.
>
> ```
> 지금:  [설계도+계산] layer  ×  backend마다 복사
> 목표:  [설계도] layer 하나  →  계산 목록 자동 생성  →  backend별 ComputeOps가 계산
> ```
>
> **→ HTP에서 실제로 보이는 낭비:** 같은 모델을 두 번 적고 있습니다. prefill용 layer 설계도와는 따로, decode용 계산 목록을 손으로 다시 씁니다. 설계도 하나에서 이 목록을 자동으로 뽑으면 중복이 없어집니다.
>
> 말씀하신 "layer가 없어져야 한다"가 2번(계산)을 빼자는 뜻이었나요, 아니면 1번(설계도)까지 없애자는 뜻이었나요?

---

### 9. 2026-11-06 10:21 UTC (11-06 19:21 KST) · 사용자 프롬프트

> 2번 뜻이었어 페이지에 반영해줘 그리고 2번 뜻이라면 목표 구조에서 layer에 뭐가 포함되는지 설명해줘

**작업 중에 보낸 추가 프롬프트 (2026-11-06 10:23 UTC (11-06 19:23 KST)):**

> 목표 구조에서 1번의 내용과 Plan layer가 겹치는 작업 진행하는 거 아님?

<details><summary>작업 중 메시지 1개</summary>

> 2번(layer에서 계산을 빼는 것) 기준으로 페이지를 고치겠습니다. 바꿀 곳은 목표 표(그림 3), 층 그림(그림 4), 목표 class diagram(그림 5), 비교 표, HTP 항목이고, "목표 구조에서 layer에 남는 것" 설명을 새로 추가합니다.

</details>

**AI 응답:**

> **layer 설계도와 Plan은 거의 겹치지 않습니다. 맡는 질문이 다르기 때문입니다. 실제로 겹치는 곳은 ModelFeatures 한 곳입니다.** 페이지는 지금 계정으로 새 링크에 올렸습니다: https://claude.ai/artifact/7zoytDaUeHhrRLJA6AsPTa
>
> ### layer(설계도)와 Plan이 겹치나
>
> **→ 맡는 질문이 다릅니다.**
> - layer는 **"무엇을 계산하나"**를 적습니다. op, shape, weight가 여기에 해당합니다.
> - Plan은 **"이 하드웨어에서 어디서·어떻게 하나"**를 정합니다. kernel, 메모리 위치, 동기화, decode mode가 여기에 해당합니다.
>
> **→ 안 겹치려면 지켜야 할 경계 규칙 두 개:**
> - **lower()는 하드웨어를 모릅니다.** 항상 같은 기본 op로 펼치기만 합니다. fusion, kernel, 메모리 종류를 정하면 Plan의 일을 침범하는 겁니다.
> - **Plan은 shape나 weight를 새로 정하지 않습니다.** layer가 선언한 op와 tensor에 kernel과 메모리 위치만 붙입니다.
>
> **→ 실제로 겹치는 곳은 ModelFeatures입니다.** sliding_window, qk_norm, mlp 종류 같은 항목은 layer 속성에 이미 있는 정보를 모델이 한 번 더 적는 셈입니다. 이런 항목은 op 목록에서 자동으로 뽑고, `decode_gpu`처럼 측정으로 정한 정책값만 따로 선언하는 게 맞다고 봅니다. 문서에는 없는 제 제안입니다.
>
> ### 목표 구조에서 layer에 들어가는 것
>
> - **남는 것 (설계도):**
>   - 속성
>   - shape 계산
>   - weight 선언 (이름, 크기, 양자화 형식, tied 관계)
>   - weight 파일 읽기·쓰기
>   - 상태 선언 (KV cache 등을 얼마나, 언제까지 쓰는지)
>   - 임시 tensor 수명
>   - **`lower()`** (새로 추가): 이 layer를 어떤 op로 펼칠지
>   - 학습 규칙 (backward)
> - **빠지는 것 (계산):**
>   - forwarding 본문
>   - backend 분기 (`#if`, `*_cl`, `Htp*` class)
>   - HTP cache handle 같은 backend 자원
>   - decode 전략
>   - 모델별로 손으로 짠 decode op 목록
>
> ### 페이지에 반영한 것
>
> **→ 바뀐 곳:**
> - **그림 3:** "Layer = 설계도" 줄과 "Ops = 실행 목록" 줄을 나눴습니다.
> - **그림 4:** 실행 층(L3b)을 "op 목록 실행"으로 바꿨습니다.
> - **그림 5:** `lower()`와 `OpNode`를 추가했습니다.
> - **새 설명 블록:** "layer에 남는 것 / 빠지는 것", RMSNorm 예시, layer와 Plan의 경계를 넣었습니다.
> - **비교 표:** "계산 위치" 행을 추가했습니다.
> - **HTP 항목:** "같은 모델을 두 번 기술" 문제를 추가했습니다.
>
> **→ 문서와 다른 점:** layer가 계산을 아예 하지 않는 건 문서 §9 결정 2보다 한 걸음 더 나간 안입니다. 페이지에도 그렇게 적어 두었습니다.
>
> **→ 확인:** 로컬 Chrome으로 바뀐 그림과 블록을 렌더링해 봤고, 보인 겹침 두 곳은 고쳤습니다.
>
> **→ 링크와 원본 파일:**
> - **이전 링크(99ZP8…)는 이 계정으로는 더 수정할 수 없습니다.** 앞으로는 새 링크를 쓰면 됩니다.
> - 새 링크는 **비공개**입니다. 다른 분께 보여 주려면 Share 메뉴에서 공유해야 합니다.
> - 재로그인할 때 작업 폴더가 비워져서 페이지 원본이 지워졌습니다. 세션 기록에 남은 수정 내역을 순서대로 다시 적용해서 복원했습니다.

**결과:** 새 링크로 게시: https://claude.ai/artifact/7zoytDaUeHhrRLJA6AsPTa (Version 1). 이전 링크는 새 계정에 수정 권한이 없었다.

---

### 10. 2026-11-06 10:30 UTC (11-06 19:30 KST) · 로컬 명령과 알림 (사용자 프롬프트 아님)

- 10:30 UTC · `/remote-control` → `Remote Control disconnected.`
- 10:30 UTC · `/remote-control` → 다시 활성화

---

### 11. 2026-11-06 10:31 UTC (11-06 19:31 KST) · 사용자 프롬프트

> 근데 forward랑 backward 쪽이랑 무슨 구조 차이가 있길래 layer를 못 없앰? 내가 잘못 이해한 건가?

**AI 응답:**

> **잘못 이해하신 게 아닙니다. forward와 backward 사이에 "그래서 layer가 꼭 있어야 한다"고 할 만한 구조 차이는 없습니다.** 앞에서 제가 training을 근거로 든 건 설계도(1번)까지 없애는 경우에 대한 얘기였습니다. 그것도 "불가능하다"가 아니라 "다시 만들 게 많다"는 뜻이었는데, 표현이 과했습니다. PyTorch나 JAX는 실제로 backward를 layer가 아니라 op 단위로 계산합니다.
>
> **→ 합의한 2번 방향에서는 backward 계산도 layer 밖으로 나갈 수 있습니다.** 다만 순서상 두 단계로 나누는 게 맞습니다.
> - 1단계: 추론(forward)만 op 목록으로 옮깁니다. backward는 당분간 지금 layer 코드에 둡니다.
> - 2단계: backward도 op 목록으로 옮깁니다.
>
> LLM 추론만 보면 backward는 상관이 없습니다. CausalLM layer는 대부분 backward가 `throw`입니다.
>
> **→ backward가 forward보다 어려운 이유 (못 하는 이유는 아님):**
>
> ```
> forward : x → [rmsnorm] → [fc] → [attention] → y    앞으로. 끝난 값은 바로 버려도 됨
> backward: dx ← [rmsnorm'] ← [fc'] ← [attention'] ← dy   거꾸로. forward 때 값이 다시 필요
>                             ↘ dW → optimizer
> ```
>
> 1. **forward 값을 끝까지 보관해야 합니다.** backward는 forward 때의 입력과 중간값을 다시 씁니다. 그래서 어떤 값을 언제까지 메모리에 둘지 정해야 합니다. 지금은 각 layer가 "이 tensor는 `calcDerivative`·`calcGradient` 때까지 살려 둬"라고 선언합니다(`TensorLifespan`의 `CALC_DERIV`, `CALC_GRAD` 등). memory planner는 이 선언을 보고 메모리를 재사용합니다. op 목록으로 옮기면 op마다 같은 정보가 필요합니다.
>
> 2. **결과가 두 가지이고 optimizer와 이어집니다.** 하나는 앞 층으로 넘기는 gradient(`calcDerivative`), 하나는 weight gradient(`calcGradient`)입니다. weight gradient는 그 weight를 가진 쪽의 optimizer로 가야 합니다.
>
> 3. **손으로 최적화한 미분 규칙이 있습니다.** 예를 들어 softmax + cross-entropy는 각각 미분해서 곱하지 않고 "출력 − 정답" 한 줄로 계산합니다(`CrossEntropySoftmaxLossLayer`). 문서 §8도 "fusion한 backward는 fusion한 forward를 거꾸로 읽은 것이 아니다"라고 경고합니다.
>
> 4. **지금 코드에는 backward가 op 단위로 없습니다.** ComputeOps에 backward용 virtual이 0개입니다. 대신 core layer 75개가 `calcDerivative`를 손으로 구현하고 있습니다. 옮기려면 op마다 backward kernel을 만들고, forward 목록에서 backward 목록을 자동으로 만드는 장치(PyTorch의 autograd 같은 것)가 필요합니다.
>
> **→ 결국 목표 구조의 layer에 남는 것은 "backward 규칙"뿐입니다.** 계산은 하지 않습니다. 이 규칙도 나중에는 op마다의 미분 규칙으로 옮길 수 있습니다. 그러면 layer는 `lower()`만 내놓고, backward 목록은 forward 목록에서 자동으로 만들어집니다.
>
> 원하시면 페이지의 "학습 규칙" 줄을 이 2단계 설명으로 더 분명하게 고치겠습니다.

---

### 12. 2026-11-06 10:34 UTC (11-06 19:34 KST) · 사용자 프롬프트

> backward도 op로 만들어야하지 않을까? 무작정 동의하지 말고 검토해줘

<details><summary>작업 중 메시지 1개</summary>

> 백워드도 op로 만들어야 하는지 판단하려면 지금 backward가 실제로 어떻게 계산되는지부터 봐야 해서, 코드를 확인하겠습니다.

</details>

**AI 응답:**

> **backward도 op로 만드는 방향에는 동의합니다. 다만 forward와 동시에 하는 건 반대이고, 2단계로 미루는 게 맞다고 봅니다.** 확인하면서 제가 앞에서 한 말 하나를 바로잡을 것도 나왔습니다.
>
> ### 먼저 정정
>
> **→ "ComputeOps에 backward가 없다"는 말은 반만 맞습니다.** rmsnorm backward 같은 전용 kernel은 없는 게 맞습니다. 하지만 backward 계산 자체는 이미 op table을 거칩니다. 예를 들어 FC backward는 `dot_deriv_wrt_1` / `dot_deriv_wrt_2`, 즉 gemm 두 번이고, 이건 Tensor 연산을 통해 ComputeOps로 갑니다. 그래서 op 목록으로 옮기는 일은 앞서 말한 것보다 작습니다.
>
> ### 찬성하는 이유
>
> **1 → 안 옮기면 실행 경로가 두 벌이 됩니다.** forward는 op 목록이 계산하는데 backward는 layer 메서드가 계산하면, layer가 다시 계산 코드를 갖게 됩니다. 합의한 2번 원칙과 어긋납니다.
>
> **2 → 그 결과 fork가 backward 쪽에서 다시 생깁니다.** 지금 OpenCL fork들이 실제로 그렇습니다. `FullyConnectedLayerCl`의 backward는 CPU와 똑같은 코드를 복사해 둔 것입니다. RMSNorm, transpose는 아예 지원하지 않고, concat은 빈 함수입니다.
>
> **3 → 메모리 수명을 자동으로 계산할 수 있습니다.** 지금은 layer마다 "이 값은 backward 때까지 살려 둬"를 손으로 선언합니다. op 목록이 있으면 각 값이 마지막으로 쓰이는 op를 목록에서 찾아 계산할 수 있어서, 손으로 선언하다 생기는 실수가 줄어듭니다.
>
> **4 → 검증을 한 번만 만들면 됩니다.** op마다 gradient check(수치 미분과 비교)를 만들어 두면 모든 layer가 그걸 재사용합니다.
>
> ### 지금 같이 하는 데 반대하는 이유
>
> **5 → 당장 이득 보는 곳이 CPU 하나뿐입니다.** HTP와 QNN은 학습을 하지 않습니다. CausalLM layer 17개 중 12개가 `supportBackwarding() = false`입니다. OpenCL fork도 backward가 대부분 미구현입니다. 실제로 학습이 도는 건 CPU 경로뿐입니다.
>
> **6 → 기존 학습 앱이 회귀할 위험이 있습니다.** MNIST, ResNet, TransferLearning 같은 앱이 손으로 최적화한 규칙에 기대고 있습니다. 예를 들어 softmax + cross-entropy는 "출력 − 정답" 한 줄로 계산합니다. 이런 규칙을 같은 op으로 1:1 옮기지 않으면 속도, 메모리, 수치가 달라집니다. 75개 layer를 옮기는 동안 gradient가 똑같은지 확인해야 합니다.
>
> **7 → 범위가 두 배가 됩니다.** forward를 op 목록으로 바꾸는 것도 아직 없습니다. 둘을 동시에 하면 작업도 검증도 두 배입니다.
>
> ### 제안
>
> - **목표에는 넣습니다.** 최종 구조에서는 backward도 forward와 같은 실행기가 op 목록으로 돌립니다.
> - **방식:** "아무 op나 자동 미분"하는 autograd가 아니라, op 종류마다 backward 규칙을 표로 정합니다.
>   - fc → [gemm, gemm]
>   - rmsnorm → [rmsnorm_backward]
>   - softmax+CE → [softmax_ce_backward] (손으로 최적화한 규칙을 op 하나로 유지)
> - **순서:** forward 먼저 하고, backward는 그다음입니다. 그 사이에는 "새 backward 코드는 layer가 아니라 op 단위로 쓴다"는 규칙만 둬서 fork가 다시 생기지 않게 합니다.
> - **학습 못 하는 backend:** HTP, QNN처럼 backward가 없는 backend에서는 ExecPlan이 학습 모드를 거절하게 합니다. DeviceCaps에 학습 가능 여부를 넣으면 됩니다.
>
> **Also found:** `ConcatLayerCl::calcDerivative`가 빈 함수입니다. GPU에서 concat이 들어간 모델을 학습하면 오류 없이 gradient가 틀립니다. 적어도 `throw`로 바꿔야 합니다.
>
> 이 내용(backward도 op 목록, 2단계)을 페이지의 목표 구조에 반영할까요?

---

### 13. 2026-11-06 10:38 UTC (11-06 19:38 KST) · 사용자 프롬프트

> 지금 내용을 페이지에 반영하고 필요하다면 목표 구조 이미지도 업데이트해줘

<details><summary>작업 중 메시지 1개</summary>

> 페이지를 고치겠습니다. 목표 표(그림 3)에 학습 줄을 추가하고, 그림 4·5를 backward 기준으로 바꾸고, forward/backward op 목록 그림(그림 5b)과 설명 블록을 새로 넣습니다. 비교 표, OpenCL 항목, 제안 순서에도 반영합니다.

</details>

**AI 응답:**

> **같은 링크(Version 2)에 반영했습니다: https://claude.ai/artifact/7zoytDaUeHhrRLJA6AsPTa**
>
> **→ 바뀐 그림:**
> - **그림 3 (목표 표, backend별):**
>   - **Train 줄**을 새로 넣었습니다. CPU는 backward kernel을 추가하고, OpenCL·CUDA는 지원하는 op만 학습하고 나머지는 Plan이 거절합니다. HTP·QNN은 학습 모드를 Plan이 거절합니다.
>   - Ops 줄은 "forward는 `lower()`로, backward는 op별 규칙표로 생성(2단계)"으로 바꿨습니다.
>   - Plan 줄에 "학습 가능 여부"를 추가했습니다.
> - **그림 4 (층 구조):** 실행 층을 "op 목록 실행 (forward · backward)"으로 바꿨습니다.
> - **그림 5 (목표 class):**
>   - Layer의 `calcDerivative`를 빼고 `trainable`(학습 대상 표시)만 남겼습니다.
>   - `OpNode`에 `backward()`(규칙표의 op들)를 추가했습니다.
>   - `ComputeOps`에 `*_backward()`(2단계)를 추가했습니다.
>   - `DeviceCaps`에 "학습 지원 여부"를 추가했습니다.
> - **그림 5b (새로 추가):** forward op 목록에서 backward op 목록이 만들어지는 모습입니다. fc의 backward는 기존 gemm을 재사용하고, rmsnorm과 attention만 새 kernel이 필요합니다. softmax+CE는 fused 규칙을 그대로 유지합니다.
>
> **→ 바뀐 글:**
> - **새 섹션 "backward도 op 목록으로, 다만 2단계":** 옮겨야 하는 이유 4가지, forward와 동시에 하지 않는 이유 3가지, 방식(규칙표), 중간 규칙, 학습 못 하는 backend는 거절.
> - **"layer에 남는 것":** "학습 규칙"을 "학습 설정(trainable, loss)"으로 바꿨습니다. "빠지는 것" 목록에 backward 계산을 추가했습니다.
> - **비교 표:** backward 행을 추가했습니다.
> - **OpenCL 항목 (6 → 7건):** concat backward가 빈 함수라는 문제를 추가했습니다.
> - **TL;DR, 제안 순서(5단계째), 목차**도 맞춰 고쳤습니다.
>
> **→ 확인:** 바뀐 그림 3장과 새 섹션을 로컬 Chrome으로 렌더링해서 겹침이 없는 걸 봤습니다.
>
> **→ 공유 상태:** 이 페이지는 지금 "링크가 있는 누구나" 볼 수 있습니다.
>
> **Also found:** 페이지 원본은 `/tmp` 아래 scratchpad에 있습니다. 재로그인할 때 한 번 지워진 적이 있으니, 원하시면 홈 디렉터리처럼 지워지지 않는 곳에 사본을 두겠습니다.

**결과:** https://claude.ai/artifact/7zoytDaUeHhrRLJA6AsPTa Version 2 (backward도 op 목록, 2단계)

---

### 14. 2026-11-06 10:45 UTC (11-06 19:45 KST) · 사용자 프롬프트

> 목표 구조에서 Decode가 왜 따로 분류되어 있는 이유가 있음? Prefill은 없는데 말이야

**AI 응답:**

> **이유는 있었지만 분류가 틀렸습니다. 말씀하신 대로 Prefill도 같이 다뤄야 맞습니다.**
>
> ### Decode를 따로 둔 이유
>
> **→ 문서의 용어를 그대로 따랐습니다.** 문서는 backend별 실행 방식을 고르는 자리를 "decode hook"(SEAM-2)이라고 부르고, 지금 코드의 함수 이름도 `Context::runDecode`입니다.
>
> **→ decode에서 차이가 가장 크게 나는 건 사실입니다.** decode는 token마다 작은 계산(M=1)을 반복하기 때문에, op마다 호출하는 비용이 계산 비용보다 커집니다. 그래서 CUDA의 graph capture-replay나 HTP의 상주 DSP graph 같은 방식이 decode에서 효과를 봅니다. prefill은 한 번에 큰 계산이라 op 단위로 돌려도 손해가 적습니다.
>
> ### 그런데 분류가 틀린 이유
>
> **1 → 같은 hook이 이미 prefill에도 불립니다.** `neuralnet.cpp`의 호출부 주석이 "one decode or prefill step is dispatched through the resolved Context"입니다. 이름만 runDecode일 뿐, 실제로는 추론 한 단계마다 거치는 입구입니다.
>
> **2 → prefill에도 backend별 실행 방식 결정이 있습니다.**
> - CUDA는 같은 `runDecode` 안에서 prefill도 graph capture합니다(`prefill_graph`, `from == 0 && to - from > 1`).
> - HTP는 여러 줄을 이어 처리할 때(continuation) 상주 graph를 쓰고, 긴 prompt는 `init_seq_len` 단위로 잘라 넣습니다.
> - QNN은 prefill도 graph를 통째로 실행합니다.
>
> **3 → 결국 "Decode" 줄의 내용은 "실행 방식(mode)"입니다.** 단계(prefill · decode · 학습)마다 mode가 다를 뿐이라, decode만 따로 이름 붙일 이유가 없습니다.
>
> ### 제안
>
> **→ 그림 3의 "Decode" 줄을 "실행 방식" 줄로 바꾸고, 칸마다 prefill과 decode를 같이 적습니다.**
>
> | backend | prefill | decode |
> |---|---|---|
> | CPU · OpenCL | eager | eager |
> | CUDA | 첫 chunk capture | capture-replay |
> | HTP | op 단위 (chunk) | offload (상주 graph) |
> | QNN | offload | offload |
>
> **→ ExecPlan.mode를 단계별로 나눕니다.** prefill · decode · 학습마다 mode를 하나씩 둡니다.
>
> **→ hook 이름을 바꿉니다.** `runDecode` 대신 `runStep(phase)` 같은 이름이 맞습니다. 지금 HTP의 단일 token decode는 이 hook을 거치지 않고 모델 코드(`npuDecodeTokenId`)에서 직접 들어가는데, 이 경로도 같은 hook 뒤로 넣습니다.
>
> **→ Train 줄은 그대로 둡니다.** 그 줄은 실행 방식이 아니라 "backward kernel이 있는가"를 보여 주는 줄이라 성격이 다릅니다.
>
> 이 내용으로 그림 3(목표 표), 그림 5(Context의 runDecode), 관련 설명을 고칠까요?

---

### 15. 2026-11-06 10:47 UTC (11-06 19:47 KST) · 사용자 프롬프트

> 둘이 왜 분리되어야 함? 그냥 통합되어 있으면 안 됨?

**AI 응답:**

> **통합하는 게 맞습니다. 앞에서 제가 prefill과 decode를 표에서 두 칸으로 나누자고 한 건 과했습니다.** 다만 "두 경우를 완전히 같은 방식으로 돌린다"까지 가면 decode 속도를 잃습니다. 그래서 하나로 합치되, 갈라지는 지점을 "단계 이름"이 아니라 "한 번에 몇 줄을 처리하느냐(M)"로 바꾸는 걸 제안합니다.
>
> ### 통합해도 되는 부분
>
> **→ 하는 계산은 같습니다.** prefill과 decode는 같은 op 목록, 같은 weight, 같은 KV cache를 씁니다. 다른 건 한 번에 처리하는 줄 수뿐입니다. prefill은 M=N(prompt 길이)이고 decode는 M=1입니다.
>
> **→ 입구도 이미 하나입니다.** 지금 코드도 prefill과 decode가 같은 hook(`runDecode`)을 지납니다. 그러니 op 목록, hook, ExecPlan을 각각 하나로 두면 됩니다.
>
> ### 완전히 같은 방식으로 돌리면 안 되는 이유
>
> **→ 병목이 다릅니다.**
> - prefill은 큰 행렬곱이라 계산량이 병목입니다.
> - decode는 token마다 weight 전체를 다시 읽는 작은 계산이라 메모리 대역폭과 호출 비용이 병목입니다. HTP 문서도 decode 비용은 거의 전부 weight 대역폭이라고 적고 있습니다.
> - 그래서 같은 op라도 맞는 kernel이 다릅니다. 예를 들어 HTP의 ternary expert는 prefill에서는 HMX용으로 펼쳐서 쓰고, decode에서는 펼치지 않고 바로 씁니다.
>
> **→ 미리 묶어 두는 방식은 모양이 고정돼야 합니다.** CUDA graph replay와 HTP 상주 graph는 "같은 모양을 반복한다"는 전제에서만 쓸 수 있습니다. M=1 decode는 매 token 모양이 같아서 한 번 묶고 계속 재생할 수 있습니다. prefill은 prompt마다 길이가 달라서 그렇게 못 합니다. 실제로 CUDA는 prefill에서 첫 chunk만 capture하고, HTP 상주 graph의 attention op는 M=1 전용인 `ATTN_M1`입니다. 전부 eager로 통일하면 decode를 빠르게 만드는 이 장치들이 사라집니다.
>
> ### 제안: 하나로 합치고, 모양으로 고르기
>
> **1 → 경로는 하나입니다.** op 목록 하나, hook 하나(`runStep(from, to)`), ExecPlan 하나를 씁니다. "prefill 모드"나 "decode 모드" 같은 이름은 두지 않습니다.
>
> **2 → 갈라지는 건 Plan의 결정 하나뿐입니다.** Plan이 step의 모양을 보고 kernel과 실행 방식을 고릅니다. M=1을 반복하면 capture나 상주 graph를 쓰고, M>1이면 op 단위로 돌립니다. 이 패턴은 이미 코드에 있습니다. `accelerates_qs4cx_at_m1()`, 그리고 `supports_gemm_q4_0_batch_fp32() && M > 1` 같은 분기가 그 예입니다(ARCHITECTURE.md §4).
>
> **3 → 모양별 변형은 자동으로 만듭니다.** graph를 미리 묶는 backend(HTP, QNN, CUDA capture)는 같은 op 목록에서 M=1용과 M>1용 변형을 각각 자동으로 만듭니다. 지금처럼 decode용 목록을 손으로 따로 짜지 않습니다.
>
> **4 → 표에서는 "Decode" 줄을 "실행 방식" 한 줄로 바꿉니다.** 칸마다 그 backend가 쓸 수 있는 방식만 적고, 어느 것을 쓸지는 Plan이 모양을 보고 고른다고 적습니다.
>
> 이 내용으로 그림 3(목표 표)과 그림 5(Context의 `runDecode` → `runStep`), 관련 설명을 고칠까요?

---

### 16. 2026-11-06 10:49 UTC (11-06 19:49 KST) · 사용자 프롬프트

> 응 그렇게 페이지에 반영해줘

<details><summary>작업 중 메시지 2개</summary>

> 그림 3·4·5와 관련 설명을 "prefill과 decode는 한 경로(`runStep`), 실행 방식은 Plan이 M으로 선택" 기준으로 고치겠습니다.

> Rendering looks clean. Publishing.

</details>

**AI 응답:**

> **같은 링크(Version 3)에 반영했습니다: https://claude.ai/artifact/7zoytDaUeHhrRLJA6AsPTa**
>
> **→ 그림 3 (목표 표, backend별):**
> - "Decode" 줄을 **"Run (실행 방식)" 줄**로 바꿨습니다. 각 칸에는 그 backend가 쓸 수 있는 방식과 언제 그 방식을 쓰는지를 적었습니다.
>   - CPU: eager
>   - OpenCL: eager, coherence는 caps로 판단
>   - CUDA: M=1이 반복되면 capture-replay
>   - HTP: M=1이 반복되면 상주 graph, 아니면 op 단위
>   - QNN: M별 graph를 자동 생성해 offload
> - Ops 줄은 "op 목록 하나 (prefill · decode 공용)"로 바꿨습니다.
> - Plan 줄에는 "step 모양(M)마다 kernel · 메모리 위치 · 실행 방식"을 넣었습니다.
>
> **→ 그림 4 (층 구조):** SEAM-2를 "step hook (runStep)"으로 바꿨습니다. 현재 상태 설명에는 "hook은 runDecode라는 이름으로 이미 있음"이라고 적었습니다.
>
> **→ 그림 5 (목표 class):**
> - `Context`의 `runDecode()`를 `runStep(from, to)`로 바꿨습니다.
> - `ExecPlan`에 "step 모양(M)별" 표시를 붙였습니다.
>
> **→ 바뀐 글:**
> - **새 섹션 "prefill과 decode는 한 경로로":** 경로는 하나라는 점, kernel은 M으로 고른다는 점, 미리 묶는 방식은 M=1 반복에만 쓴다는 점, 모양별 변형은 자동으로 만든다는 점, 이렇게 네 가지를 넣었습니다.
> - **비교 표:** decode 행을 "실행 방식" 행으로 바꿨습니다. 현재 상태 칸에 "HTP 단일 token decode는 hook 밖(`npuDecodeTokenId`)으로 들어감"을 적었습니다.
> - **"layer에서 빠지는 것", 실행 흐름 5단계, 제안 순서(HTP 단계), 목차**도 맞춰 고쳤습니다.
> - **문서와의 관계:** "runStep 하나로 prefill · decode를 한 경로로 보는 것도 문서(SEAM-2 = decode hook)보다 나간 안"이라는 문장을 추가했습니다.
>
> **→ 확인:** 바뀐 그림 3장과 새 섹션을 로컬 Chrome으로 렌더링해서 겹침이 없는 걸 봤습니다.
>
> **→ 그대로 둔 것:** 현재 구조 쪽(그림 1)의 "Decode" 줄은 지금 코드 상태를 보여 주는 거라 바꾸지 않았습니다.

**결과:** https://claude.ai/artifact/7zoytDaUeHhrRLJA6AsPTa Version 3 (prefill · decode 한 경로, runStep)

---

### 17. 2026-11-06 10:52 UTC (11-06 19:52 KST) · 사용자 프롬프트

> 아래의 지시대로 2개의 md 파일 만들어서 https://github.com/dlwlzzero/nntrainer/tree/claude/nntrainer-refactoring-priority-80xzum-docs/docs/refactor_4456 에다가 push해줘
>
> - 지금 세션에서 refactoring에 대해 논의한 내용 시간 순으로 유저 대화와 AI 답변 내용 정리해서 기록해줘
> - 그리고 지금까지 작업한 html 파일 링크 md에 기록해줘
> - 그리고 지금까지 작업한 html 파일 pdf 사본으로 저장해줘

> 참고: 이 요청의 결과가 이 문서, [ARCH_MAP_PAGE.md](ARCH_MAP_PAGE.md), [ARCH_MAP_PAGE.pdf](ARCH_MAP_PAGE.pdf)다.
