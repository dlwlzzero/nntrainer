# H5 — e2e 전환, 검증과 성능

상위: [00-overview](00-overview.md) · 선행: [04-hmx-kernel](04-hmx-kernel.md)

## 검증 단계

| 단계 | 내용 | 기준 |
|---|---|---|
| 단위 | 시뮬레이터에서 HMX vs HVX w8a8 출력 비교 (`run_sim_test.sh` 프레임 재사용) | **비트 동일** |
| e2e | 기존 3-way 정확도 검증 (hexagon-hvx M4와 동일 절차) | 기존 기준 그대로 통과 (커널 교체이므로 기준 변경 없음) |
| 성능 | prefill/decode TPS + init 변환 시간 측정 | **prefill TPS 개선이 목표. decode TPS는 HVX 대비 회귀 금지** |

- 디바이스 테스트에서 **프로덕션 DMA 경로**(init 변환·wh 타일 스트리밍)의 캐시 일관성 확인 — H1 스파이크는 자체 memcpy 경로를 디바이스에서 확인했을 뿐, 본 구현의 DMA 경로는 여기서 검증
- 보고는 항상 STAT 정확도 + Pcycles/wall/forward_us 표로 병기

## 전환

- 검증 통과 후 op 디스패치 기본값을 HMX 경로로 전환. HVX 경로는 플래그로 잔존 (안전망)
