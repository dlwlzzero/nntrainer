# Hexagon HVX 백엔드 — 개요 (qwen3-0.6b e2e)

- 날짜: 2026-08-13
- 목표: nntrainer에서 Snapdragon 8 Elite(cDSP, HVX v75/v79)의 **HVX만** 사용해 qwen3-0.6b를 e2e 실행
- 범위: HMX는 의도적 제외(후속 마일스톤). 기존 HexKL 기반 `htp_backend`를 대체

이 문서는 전체 설계의 개요이며, 세부는 마일스톤별 문서로 나뉜다:

| 문서 | 마일스톤 | 내용 |
|---|---|---|
| [01-rpc-skeleton](01-rpc-skeleton.md) | M1 | FastRPC IDL, rpcmem 버퍼, 캐시 일관성, 세션 설정, 에러 처리 |
| [02-hvx-kernels](02-hvx-kernels.md) | M2 | DSP 실행기, op 세트 9종, KV cache, HVX 커널 전략 |
| [03-host-pipeline](03-host-pipeline.md) | M3 | W8_CX 양자화 도구, graph lowering, 가중치 배치, fake-quant 기준선 |
| [04-e2e-verification](04-e2e-verification.md) | M4–5 | 3-way 정확도 검증, 디바이스 테스트, 성능 튜닝 |
| [05-design-qna](05-design-qna.md) | — | 설계 과정 Q&A 부록 |

후속 프로젝트: HMX matmul 전환은 [../hexagon-hmx/00-overview.md](../hexagon-hmx/00-overview.md) 참조.

## 배경과 핵심 결정

기존 HexKL 경로는 op 단위 오프로드 구조로, 토큰당 FastRPC 호출 횟수가 병목이 되어 decode ~0.16 TPS에 그쳤다. 본 설계는 이를 뒤집는다:

| 결정 | 내용 |
|---|---|
| 오프로드 단위 | **전 그래프 DSP 상주** — embedding부터 lm_head까지 전부 DSP. 호스트는 청크당 RPC 1회. 샘플링만 CPU |
| 실행기 형태 | qwen3에 필요한 op만 가진 **평탄한 op-list 해석 실행기** (접근 3). 범용 op-graph 백엔드(접근 2)의 부분집합으로 설계해 이후 성장 |
| 양자화 | **w8a8**: 가중치 int8 대칭 per-output-channel scale, 활성화 int8 대칭 per-token 동적, int32 누산 |
| 활성화 타입 | op 사이는 fp16. int8은 matmul 내부에만 존재 |
| 기존 백엔드 관계 | `htp_backend`(HexKL) 대체. engine 이름 "htp" 유지 |

접근 2로 확장 시 유지되는 것: FastRPC skel/stub, rpcmem 할당자, DMA/VTCM/워커 풀 인프라, HVX 커널 전부, op-list 순회 실행기 골격. 교체되는 것: 텐서 디스크립터 일반화, op 세트 확장, 자동 lowering 패스.

## 전체 아키텍처

```
[호스트 (CPU, arm64)]                            [DSP (Hexagon cDSP, v75/v79)]
CausalLM 앱 (tokenizer, 샘플링)
  └─ nntrainer 모델 (engine="htp")
      └─ HexagonRunner (신규, htp_backend 대체)
          │
          ├─ 초기화 (1회):
          │   ① rpcmem으로 가중치·KV cache·         ┐
          │      입출력 버퍼 할당, 가중치 복사        │
          │   ② 그래프 → op-list 직렬화 ──────RPC──→ 실행기: op-list 수신·검증,
          │                                              버퍼 오프셋 테이블 구성
          │
          └─ forward(token_ids[], n_tokens, pos) ─RPC──→ op-list 순회 실행 (HVX 커널)
              logits 수신 ←────────────────────────────  마지막 토큰 logits 반환
```

**Prefill과 decode는 같은 경로.** `forward`가 토큰 배열을 받는다:
- Prefill: 프롬프트를 최대 청크(예: 128토큰)로 잘라 청크당 RPC 1회. op-list를 M=n_tokens로 실행, causal mask, KV 일괄 채움
- Decode: n_tokens=1로 반복. 토큰당 RPC 1회
- 커널 차이는 M뿐. decode는 대역폭 병목, prefill은 연산 병목(HMX 확장 시 최대 수혜 지점)

**디렉토리** (기존 `nntrainer/tensor/htp_backend/` 대체):

```
nntrainer/tensor/hexagon/
├── host/
│   ├── hexagon_backend.{h,cpp}   # 세션 lifecycle
│   ├── rpcmem_allocator.{h,cpp}  # rpcmem(dma-buf) 할당 래퍼
│   ├── graph_lowering.{h,cpp}    # nntrainer 그래프 → op-list 직렬화
│   └── hexagon_runner.{h,cpp}    # init / forward API
├── htp/                          # hexagon-clang 별도 빌드 → libnntr_skel.so
│   ├── nntr_htp.idl              # FastRPC 인터페이스 (QAIC 컴파일)
│   ├── executor.c                # op-list 해석 실행기
│   ├── ops/                      # op별 진입점 (9종)
│   └── hvx/                      # HVX 프리미티브 (ggml-hexagon 차용분 포함)
└── meson.build
```

**통합 지점:** engine="htp"의 동작을 op 단위 ComputeOps 디스패치에서 **그래프 레벨 오프로드**로 교체. 모델 초기화 시 컴파일된 레이어 그래프를 lowering하고 forward 전체를 HexagonRunner가 담당.

**부분 오프로드는 하지 않는다.** 그래프 분할 경계마다 토큰당 RPC·동기화·fp↔int 변환이 발생 — 기존 병목의 재현이다. 지원 불가 op를 만나면 대응은 그래프 분할이 아니라 DSP 커널 추가다. CPU fallback은 실행 전략이 아니라 안전망(DSP 초기화 실패, 예상 밖 그래프)이며, 발동 시 전체 그래프가 CPU로 간다.

## 마일스톤

1. **RPC 골격** ([01](01-rpc-skeleton.md)): idl+stub/skel 빌드(hexagon-clang 크로스 빌드 + meson 연동), 더미 왕복 성공
2. **커널 세트** ([02](02-hvx-kernels.md)): 9종 커널 시뮬레이터 통과
3. **호스트 파이프라인** ([03](03-host-pipeline.md)): W8_CX 도구 + lowering + 가중치 배치 + fake-quant 기준선
4. **e2e 통합** ([04](04-e2e-verification.md)): 디바이스 1레이어 → 전체 모델, 3-way 정확도 통과
5. **성능** ([04](04-e2e-verification.md)): DMA 더블 버퍼링·HAP_power 튜닝, TPS 측정

## 범위 외 (후속 작업)

- HMX matmul (prefill 최대 수혜), Q4 가중치(w4a8, decode 상한 2배), int8 KV cache(긴 컨텍스트), q/k/v 입력 양자화 공유, op 파이프라이닝, 부분 오프로드(접근 2 일반화 시), speculative decoding 등 M>1 decode 기법
