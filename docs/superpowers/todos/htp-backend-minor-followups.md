# HTP 백엔드 — 사소한(Minor) follow-up

> 출처: `docs/superpowers/plans/2026-06-18-htp-backend-port.md` 의 "Task 5 전 미해결 follow-up".
> 모두 **verbatim 원본** 이슈(포팅 중 의도적으로 미수정). 동작에 당장 영향 없는 위생/견고성 항목이며,
> **브랜치 최종 리뷰(requesting-code-review) 단계에서 함께 처리**하는 것을 권장한다.
>
> 참고: 같은 follow-up 목록의 **(Important) int-truncation offset 가드**는 별도로 이미 처리됨
> (`float_tensor.cpp` 의 `htp_offset_fits_int` 가드 — wf16/pwf16/pwqk0 세 호출부).
> 문서의 v79→v75 표기도 별도로 수정됨(`docs/how-to-use-htp-backend.md`).

## 상태
- [ ] 미처리 — 최종 리뷰 시 판단/적용

---

## 1. 메모리 수명/위생 (`nntrainer/tensor/float_tensor.cpp`)

### 1-1. `g_htp_scratch` thread 종료 시 미해제
- `thread_local HtpScratchBuf g_htp_scratch` 는 활성화/출력 staging 버퍼를 lazily grow 하지만,
  스레드가 종료될 때 `free_shared_mem_buf` 로 해제하지 않는다 → 스레드 수명 동안 RPC shared mem 점유.
- 추론은 사실상 단일 스레드라 실사용 영향은 작지만, 워커 스레드가 빈번히 생성/소멸하는 경로에서는 누수.
- **제안:** 스레드 종료 시 정리하는 RAII 래퍼(소멸자에서 `free_shared_mem_buf`) 또는 명시적 teardown 훅.

### 1-2. `g_htp_weight_cache` 미축출 (가중치 rebind 시 누수)
- 가중치 데이터 포인터를 키로 RPC shared mem 매핑을 영구 보관한다. 레이어가 storage 를
  rebind(LoRA reload 등) 하면 옛 포인터 엔트리가 캐시에 남아 **shared mem 이 축출되지 않고 누적**.
- **제안:** rebind/teardown 시 엔트리 invalidate + `free_shared_mem_buf`, 또는 용량 제한 + LRU 축출.

## 2. 중복 include (`nntrainer/tensor/float_tensor.cpp`)
- `q4_0_utils.h` 가 두 번 include 됨: 파일 상단 `#include <q4_0_utils.h>` (항상),
  그리고 `#if ENABLE_HTP` 블록 안 `#include "q4_0_utils.h"`.
- 동작엔 무해(헤더 가드)하나 정리 대상. **제안:** ENABLE_HTP 블록의 중복 줄 제거.

## 3. configure 중단 위험 (`nntrainer/tensor/htp_backend/meson.build`)
- `run_command('printenv', 'HEXAGON_SDK_HOME', check: true)` 사용 →
  `enable-htp=true` 인데 `HEXAGON_SDK_HOME` 가 설정되지 않은 환경에서는 **meson configure 가 중단**된다.
- **제안:** `check: false` 로 받고 결과가 비면 사람이 읽을 수 있는 `error('HEXAGON_SDK_HOME ...')`
  메시지로 실패시키거나, 옵션이 켜졌을 때만 친절한 진단을 내도록 변경.
