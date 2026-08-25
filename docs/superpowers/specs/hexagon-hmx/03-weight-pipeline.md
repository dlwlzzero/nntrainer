# H3 — 가중치 파이프라인 (init 1회 변환, 메모리 중립)

상위: [00-overview](00-overview.md) · 선행: [02-build-integration](02-build-integration.md)

## 변환 흐름

```
[호스트]                                  [DSP]
① HMX 가중치 rpcmem 버퍼 별도 할당
   (K·N을 HMX 블록 배수로 패딩: K→32, N→32/64)
② init RPC ──────────────────────────→ ③ 레이어별·타일별:
                                            row-major 가중치 타일 DMA → VTCM
                                            → rm_to_wh_i8 (VTCM 내 변환)
                                            → DDR HMX 버퍼로 복사
④ 완료 응답 수신 후 row-major
   원본 rpcmem 버퍼 해제  ←──────────────  ⑤ 이후 forward는 wh 버퍼만 참조
```

- 순간 최대 메모리는 원본 + HMX 버퍼가 공존하는 init 구간(≈2×0.6GB). 상주 메모리는 변환 전과 동일(+패딩분)
- DDR HMX 버퍼 내 타일 배치 순서는 **커널 루프 순서(N-타일 → K-타일)와 동일하게** — 실행 중 가중치 DMA가 순수 선형 스트리밍이 되도록 ([06 ②](06-optimizations.md) 필수 반영)

## W8_CX 도구 변경 (1가지)

- per-output-channel **가중치 열합 `colsum(w)` (int32)** 추가 출력 — u8 활성화 zero-point 보정용
- 도구에서 계산이 가장 쌈 (DSP init 계산도 가능하나 불필요)

## zero-point 보정 수식

활성화를 `x_u8 = clamp(round(x/s_x) + 128, 0, 255)`로 양자화하면:

```
acc_u8i8 = Σ_k x_u8[k]·w[k] = Σ_k (x_i8[k]+128)·w[k] = acc_i8i8 + 128·colsum(w)
∴ y = (acc_u8i8 − 128·colsum(w)) · s_w · s_x
```

정수 연산은 정확하므로 dequant 수식·순서를 HVX 커널과 동일하게 맞추면 **비트 동일 출력**이 나온다 — 단위 테스트의 판정 기준 ([04](04-hmx-kernel.md), [05](05-e2e-verification.md)).

## 완료 기준

- 시뮬레이터에서 변환된 wh 버퍼로 `hmx_mm_u8i8` 실행 결과가 CPU 레퍼런스와 일치
- init 변환 시간이 H1 스파이크 추정치 범위 내
