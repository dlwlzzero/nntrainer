# M2: DSP 실행기와 HVX 커널

[← 개요](00-overview.md)

목표: op 9종 커널 시뮬레이터(hexagon-sim) 통과.

## 모델과 데이터 타입

**qwen3-0.6b:** hidden 1024, 28 레이어, Q 16 / KV 8 헤드(GQA), head_dim 128, FFN 3072, vocab 151,936, embedding·lm_head tied, **QK-Norm** (q,k per-head RMSNorm).

**데이터 타입 흐름:** op 사이는 fp16(ACT 버퍼). int8은 matmul 내부에만:

```
MATMUL_W8A8 내부: fp16 입력 행 → per-token 동적 양자화(int8) → vrmpy 누산(int32)
                  → × (scale_w[ch] · scale_a[row]) → fp16 출력
```

양자화를 matmul에 융합해 별도 QUANT op와 버퍼 왕복을 제거. (q/k/v가 같은 입력을 쓰므로 입력 양자화 1회 공유는 후속 최적화 여지.)

## op 세트 (9종)

| op | 입출력 | 비고 |
|---|---|---|
| `EMBED` | token_ids → fp16 [M×1024] | int8 행 gather + dequant |
| `RMSNORM` | fp16 → fp16 | per-head 플래그로 QK-Norm 겸용 |
| `MATMUL_W8A8` | fp16 [M×K] × int8 [N×K] → fp16 [M×N] | q/k/v/o, gate/up/down 전부 |
| `ROPE` | q,k fp16 → fp16 | sin/cos는 사전 계산 테이블 |
| `ATTN` | q,k,v → fp16 | KV append + causal SDPA + GQA 한 op |
| `SILU_MUL` | gate,up → fp16 | silu(gate) ⊙ up |
| `ADD` | fp16 + fp16 | residual |
| `MATMUL_LOGITS` | fp16 [1×1024] × int8 → fp32 [vocab] | 마지막 토큰만, lm_head(tied) |
| (예약) | | HMX matmul 등 확장용 |

레이어당 16 op (norm, q/k/v proj, q/k norm, rope, attn, o proj, add, norm, gate/up, silu_mul, down, add) × 28 + 앞뒤 3 (embed / final norm / logits) = **~451 op**. 디스패치는 switch 1회라 순회 오버헤드 무시 가능.

## 실행기

op-list 순차 루프. op 내부는 워커 풀로 행/헤드 분할 병렬화, op 사이는 배리어. 파이프라이닝은 범위 외.

**KV cache:** fp16, `[layer][kv_head][seq_pos][head_dim=128]` — attention이 seq 방향 순차 읽기. 행 256B로 정렬 유지. KV 버퍼 헤더에 포맷 필드를 두어 int8 KV(긴 컨텍스트 최적화)를 후속으로 열어둠. int8 KV 전환 시 변경은 ATTN op 내부(append 시 양자화)에 국한되고 proj는 불변.

**op 디스크립터** (접근 2 확장 염두):

```c
struct tensor_ref { uint8_t buf_id; uint32_t offset; };
struct op_desc {
  uint32_t kind;                    // enum, 예약 여유
  struct tensor_ref in0, in1, out;
  uint32_t m, k, n;                 // M은 실행 시 n_tokens로 치환
  uint32_t flags, layer;            // per-head norm, causal 등
};
```

## HVX 커널 전략

**정렬 규칙:** 호스트 lowering이 모든 오프셋·행 stride를 128B(HVX 벡터) 정렬로 배치. 커널에 비정렬 처리 없음.

**MATMUL_W8A8** (전체 FLOPs ~99%):
- `vrmpy`: 벡터(128B)당 32개 int32 lane, lane마다 int8 4쌍 곱-누산 = 명령당 128 MAC. K=1024 행 = 벡터 8개 → vrmpy 8회 + lane reduce 1회로 출력 1개
- 입력 양자화(op 진입 시 1회): absmax → scale → int8. 결과는 VTCM 상주
- 가중치 스트리밍: 워커별 N-슬랩을 DMA 큐로 VTCM 더블 버퍼링(계산과 DMA 중첩) — ggml-hexagon `dma-queue` 차용
- M 처리: VTCM의 가중치 타일에 대해 안쪽 루프를 M으로 — M=1이면 matvec, M=128이면 타일당 128회 재사용. 별도 prefill 커널 불필요
- decode에서 가중치 재사용은 불가능(matvec은 원소당 1회 사용, 토큰 간 재사용은 VTCM 8MB ≪ 0.6GB): 대역폭 병목이 근본이며, 우회는 후속 Q4(w4a8)로 읽기량 자체를 줄이는 것

**기타 op:**

| op | 방식 |
|---|---|
| `RMSNORM` | 제곱합 fp32 위드닝 누산 → rsqrt → fp16 곱 |
| `ROPE` | 호스트가 init 때 sin/cos 테이블 사전 계산(WEIGHTS에 포함), DSP는 곱·셔플만 |
| `ATTN` | KV head(8) 단위 워커 분할. scores fp16(fp32 누산), exp는 ggml `hvx-exp.h`, softmax 후 ×V, KV append 포함 |
| `SILU_MUL` | sigmoid = exp 프리미티브 재사용 |
| `ADD`/`EMBED` | 벡터 add / 행 gather+dequant |

**VTCM 예산:** ~8MB ÷ 워커 수. 워커당 ~1.3MB: Xq(1KB) + 가중치 더블 버퍼(각 ~512KB) + 스크래치. 타일 크기는 lowering이 예산에서 역산.

**ggml-hexagon 차용** (MIT → Apache-2.0에 라이선스 표기 후):
`dma-queue.{c,h}`, `hvx-exp.h`·`hvx-utils` 계열, 워커 풀 패턴(`qurt_hvx_get_units()` 조회). 행렬곱 커널은 포맷이 달라(블록 vs per-channel) 신규 작성.

## 테스트 (x86, 디바이스 불필요)

- 커널 단위: Hexagon 시뮬레이터(hexagon-sim)에서 HVX 커널 vs C 레퍼런스
- 실행기: 합성 op-list를 시뮬레이터에서 golden 비교
