# 백엔드 구조 지도 (HTML 페이지)

`qc_npu_rebase`의 현재 백엔드 구조와 `docs/backend_guide/ARCHITECTURE_REFACTOR.md`가 그리는 목표 구조를 그림으로 비교하고, 남은 일을 정리한 페이지다. 만든 과정과 결정은 [ARCH_MAP_SESSION.md](ARCH_MAP_SESSION.md)에 시간순으로 있다.

## 링크

| 링크 | 상태 | 마지막 버전 |
|---|---|---|
| https://claude.ai/artifact/7zoytDaUeHhrRLJA6AsPTa | **현재본.** 10:12 UTC 재로그인 뒤 새로 만든 링크 | Version 3 (2026-11-06 10:51 UTC) |
| https://claude.ai/artifact/99ZP8UjFiQUp1TZQiDxR4f | 이전본. 재로그인 전 계정으로 만든 링크라 이 세션에서는 더 고칠 수 없음 | Version 3 (2026-11-06 08:43 UTC) |

- 두 링크 모두 claude.ai 페이지라서, 공유 설정에 따라 로그인이 필요하거나 열리지 않을 수 있다. 로그인 없이 보려면 PDF 사본을 쓴다.
- 이전본에는 10:14 UTC 이후의 결정(layer는 설계도만, backward도 op 목록, prefill · decode 한 경로)이 없다.

## PDF 사본

- [ARCH_MAP_PAGE.pdf](ARCH_MAP_PAGE.pdf): 현재본 Version 3의 원본 HTML을 headless Chrome으로 인쇄한 것(A3, 라이트 테마).
- 인쇄용으로 두 가지만 바꿨다. 접혀 있던 "근거 (file: symbol)" 블록을 펼쳤고, 화면에서 가로로 스크롤되던 그림을 페이지 폭에 맞게 줄였다.

## 버전 기록 (2026-11-06, UTC)

| 링크 | 버전 | 시각 | 바뀐 것 |
|---|---|---|---|
| 이전본 | 1 | 07:51 | 첫 게시. 현재 구조 표(그림 1), 목표 층 구조, 결정 흐름 비교, 남은 일과 제안 순서 |
| 이전본 | 2 | 08:06 | 현재 · 목표 class diagram(Mermaid), backend별 구조 불일치 목록 |
| 이전본 | 3 | 08:43 | class diagram을 한 화면에 들어가는 SVG로 다시 그림, 목표 구조를 backend별 표로(그림 3) |
| 현재본 | 1 | 10:27 | layer는 설계도만(계산은 op 목록), layer와 Plan의 경계, HTP "같은 모델을 두 번 기술" 항목 |
| 현재본 | 2 | 10:41 | backward도 op 목록(2단계): Train 줄, 그림 5b, OpenCL concat backward 문제 |
| 현재본 | 3 | 10:51 | prefill · decode 한 경로: Run 줄, `runStep`, Plan이 step 모양(M)으로 실행 방식 선택 |

## 페이지 구성 (현재본 Version 3)

- **1/3 현재 구조:** 그림 1(호출 순서 × backend 표), 그림 2a~2c(class diagram: 실행 사슬, graph · layer, 모델)
- **2/3 목표 구조:** 그림 3(backend별 목표 표), 그림 4(L5~L0 층), 그림 5(목표 class), 그림 5b(forward와 backward op 목록), layer에 남는 것과 빠지는 것, layer와 Plan의 경계, backward 2단계, prefill · decode 한 경로, 규칙 1a · 1b, 실행 흐름
- **3/3 비교와 남은 일:** 그림 6(결정 흐름 비교), 비교 표, backend별 구조 불일치(공통 · CPU 9건, OpenCL 7건, CUDA 4건, HTP 9건, QNN 3건), 그림 7(남은 일의 의존 관계), 제안 순서

## 기준과 한계

- 숫자와 근거 위치는 2026-11-06에 `qc_npu_rebase` @ `d061fcd20`를 grep해서 얻었다. 빌드나 실행으로 확인한 내용은 없다.
- 목표 구조 중 문서에 없는 부분(layer는 설계도만, backward도 op 목록, `runStep`, HTP의 "rpcmem MemAllocator")은 이 세션에서 정한 안이고, 페이지에도 그렇게 표시했다.
