# M3: 호스트 파이프라인 — 양자화·lowering·가중치 배치

[← 개요](00-overview.md)

목표: W8_CX 양자화 도구 + graph lowering + 가중치 배치 + fake-quant 기준선.

## 오프라인 양자화 (1회)

기존 `nntr_quantize` 도구에 `W8_CX` 타입 추가 — int8 대칭 per-output-channel, scale fp32. 기존 `QScheme::PER_CHANNEL_AFFINE` quantizer 경로 재사용.

```
HF safetensors (fp32/bf16) → nntr_quantize --fc_dtype W8_CX → 양자화 .bin + nntr_config.json
```

**embedding도 int8:** vocab×hidden = 1.56억 파라미터(모델의 ¼). tied이므로 int8 사본 하나(155MB)를 EMBED와 MATMUL_LOGITS가 공유. fp32로 두면 622MB로 배보다 배꼽이 커짐.

**활성화는 오프라인 작업 없음:** dynamic 양자화이므로 calibration 데이터셋(대표 입력으로 활성화 범위를 미리 기록하는 static 방식의 준비물)이 불필요. scale은 런타임에 매 행 absmax로 산출.

## 런타임 초기화 (호스트)

기존 CausalLM 로더로 .bin 로드 → WEIGHTS rpcmem에 DSP 레이아웃 배치(행 128B 정렬 재배열, scale 벡터, norm 가중치 fp16, RoPE 테이블 계산·포함) → 1회 flush → init RPC. 파일 포맷↔DSP 레이아웃 변환은 전부 호스트 담당.

**Graph lowering:** nntrainer 레이어 그래프를 op-list + 정적 버퍼 오프셋 + 타일 파라미터로 바꾸는 초기화 1회 호스트 단계. 모든 오프셋·행 stride를 128B 정렬로 배치하고([02](02-hvx-kernels.md) 정렬 규칙), VTCM 예산에서 타일 크기를 역산한다. 결정은 호스트가 미리, DSP는 실행만.

## fake-quant 기준선

DSP 없이 순수 CPU에서 양자화 손실만 측정하는 기준선 — 가중치에 quant→dequant 왕복만 적용하고 나머지는 fp32로 실행. [04](04-e2e-verification.md)의 3-way 검증에서 ② 역할.

## 테스트 (x86, 디바이스 불필요)

- 호스트 단위: lowering(정렬·경계·op 수), W8_CX 양자화 왕복
- fake-quant 정확도 기준선: 원본 fp32 대비 perplexity 증가 <1% 기대, 초과 시 양자화 설계 재검토
