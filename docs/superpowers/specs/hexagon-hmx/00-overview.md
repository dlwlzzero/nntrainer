# Hexagon HMX 백엔드 — 개요 (w8a8 matmul → HMX u8i8 전환)

- 날짜: 2026-08-20
- 전제: [hexagon-hvx](../hexagon-hvx/00-overview.md) M1–M5 완료 (qwen3-0.6b가 HVX 경로로 e2e 실행·검증 완료)
- 목표: 모든 w8a8 matmul을 HexKL NPU Micro API의 HMX u8i8 커널로 교체 — prefill·decode 동일 경로
- 의존성: `~/hexkl_addon/` (HexKL 1.0 Beta.2, `libhexkl_micro.a`)

이 문서는 전체 설계의 개요이며, 세부는 마일스톤별 문서로 나뉜다:

| 문서 | 마일스톤 | 내용 |
|---|---|---|
| [01-spike-relocation](01-spike-relocation.md) | H1 | 선행 스파이크: wh 타일 재배치 유효성 판정 + 변환 비용 측정 — **설계 확정 게이트** |
| [02-build-integration](02-build-integration.md) | H2 | libhexkl_micro.a 링크, HAP_compute_res HMX 파라미터, 버전 체크 |
| [03-weight-pipeline](03-weight-pipeline.md) | H3 | init 1회 wh 변환(메모리 중립), W8_CX colsum 출력, zero-point 보정 수식 |
| [04-hmx-kernel](04-hmx-kernel.md) | H4 | HMX matmul 커널, u8 양자화, 스레딩·VTCM 배치 |
| [05-e2e-verification](05-e2e-verification.md) | H5 | 비트 비교 → 3-way 정확도 → 성능 게이트, 디스패치 전환 |
| [06-optimizations](06-optimizations.md) | — | 최적화 포인트: 루프 순서 타일링, wh 버퍼 배치 순서, 누산기 stall 천장, 융합 후보 |

## 핵심 결정

| 결정 | 내용 | 근거 |
|---|---|---|
| 적용 범위 | **HMX everywhere** — `MATMUL_W8A8`, `MATMUL_LOGITS` 전부. prefill·decode 동일 경로 | decode는 대역폭 병목이라 HMX 이득은 없지만 손해도 없음. 레이아웃 단일화로 이중 레이아웃 문제 소멸, "커널 차이는 M뿐" 유지. M>1 decode(speculative 등) 확장 시 자동 수혜 |
| 가중치 배치 | **init 1회 변환, 메모리 중립** — row-major → wh 레이아웃 변환 후 원본 해제 | wh 레이아웃은 HexKL 비공개 내부 포맷이라 호스트 사전 생성 불가. on-the-fly 변환은 decode에서 토큰당 변환 비용 발생 |
| 활성화 | per-token 동적 양자화 유지, int8 대칭 → **u8 (zero-point 128 고정)** | HMX u8i8은 activation이 unsigned. 보정항은 가중치 열합으로 처리 ([03](03-weight-pipeline.md)) |
| HVX 커널 | 삭제하지 않고 **안전망 + 비교 기준**으로 유지 | op 디스패치 플래그로 선택. 비트 동일 출력이 가능해 단위 테스트 기준으로 사용 |
| 자원 확보 | `hexkl_micro_hw_init`/`hmx_lock` **사용 안 함** | 헤더에 "테스트·예제용, 통합자는 자체 루틴 사용" 명시. 기존 `HAP_compute_res` 경로에 HMX 파라미터 추가 |

**H1 스파이크가 게이트다.** wh 타일 재배치가 무효로 판정되면 본 설계 전체가 fallback(prefill-only HMX + on-the-fly 변환, decode는 HVX 유지)으로 전환된다 — [01](01-spike-relocation.md) 참조.

## 마일스톤

1. **스파이크** ([01](01-spike-relocation.md)): wh 타일 재배치 판정 + 변환 비용 측정 → 설계 확정 또는 fallback 전환
2. **빌드 연동** ([02](02-build-integration.md)): libhexkl_micro.a 링크 + HAP_compute_res HMX 파라미터 + 버전 체크
3. **가중치 변환 파이프라인** ([03](03-weight-pipeline.md)): W8_CX colsum 출력 + init 변환 + 원본 해제 — 시뮬레이터 검증
4. **HMX matmul 커널** ([04](04-hmx-kernel.md)): u8 양자화 + 커널 본체 — HVX와 비트 비교 통과
5. **e2e 전환 + 성능** ([05](05-e2e-verification.md)): 디스패치 전환, 3-way 정확도, TPS·init 시간 측정

## 범위 외 (후속 작업)

- `hmx_mm_f16` (SDPA/attention 내부 HMX화)
- `u8i4` (w4a8 — 4bit 가중치)
- QKV / gate-up 융합 (호스트 lowering 변경 — [06 ④](06-optimizations.md))
