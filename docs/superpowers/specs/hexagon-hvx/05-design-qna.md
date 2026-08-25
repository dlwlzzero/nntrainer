# 부록: 설계 과정 Q&A

[← 개요](00-overview.md)

설계 논의 중 나온 질문과 결론 요약.

**Q. uint8 같은 타입으로 양자화하고 싶다.**
그냥 uint8 캐스팅은 양자화가 아니다 — 항상 `실수값 ≈ scale × (정수 − zero_point)` 구조이며 scale을 블록/채널 단위로 관리해야 한다. 가중치는 0 중심 분포라 대칭(int8, zero_point 없음)이 표준이고, uint8(비대칭)은 zero_point 보정항만큼 커널이 복잡해지는데 가중치에서는 정확도 이득이 없다. → int8 대칭 채택.

**Q. Q8_0(블록 32)보다 channelwise를 쓰고 싶다.**
channelwise가 HVX 커널 관점에서 오히려 단순하다: 행 전체를 순수 int8 dot으로 int32 누산까지 끝내고 마지막에 scale을 1회만 곱한다. 8-bit에서 per-channel은 정확도도 표준적으로 충분(QNN/ONNX 생태계 기본). 블록 양자화가 정확도상 중요해지는 건 4-bit부터. → per-channel 채택.

**Q. 누산을 int32 말고 int8로 하면 quant/dequant 시간이 줄지 않나?**
불가능하고 이득도 없다. int8×int8 곱 하나가 이미 ±16k로 int8 범위를 초과하고, K=1024개 합산은 24bit 이상 필요 — int8 누산이면 즉시 오버플로우. int32 누산은 별도 단계가 아니라 `vrmpy` 명령 안에서 공짜로 일어난다. quant/dequant 비용은 O(M×K)로 행렬곱 O(M×N×K)의 ~0.1% 수준이라 최적화 대상이 아니다.

**Q. ION/rpcmem이 뭔가?**
cDSP는 호스트 malloc 메모리를 볼 수 없다(주소 공간 분리). ION/dma-buf는 여러 하드웨어 블록이 공유 가능한 버퍼를 만드는 커널 기능, rpcmem은 이를 감싸 CPU·DSP 양쪽에 zero-copy로 매핑해주는 Hexagon SDK API. 가중치를 초기화 때 1회 복사하면 이후 재전송이 없는 근거.

**Q. "디코드"라고 썼는데 prefill도 포함인가?**
포함. `forward(token_ids[], n_tokens, pos)`로 토큰 배열을 받아 prefill은 청크당 RPC 1회(M=청크), decode는 M=1. 커널 차이는 M뿐.

**Q. 버퍼 오프셋 테이블이 뭔가?**
DSP는 호스트 포인터를 못 쓰므로 텐서 참조를 (버퍼 id, 오프셋)으로 표현한다. init 때 DSP가 매핑된 버퍼들의 자기 쪽 주소로 `버퍼 id → DSP 주소` 표를 만들고, 실행 시 `테이블[buf] + offset`으로 해석. 오프셋은 전부 호스트 lowering이 정적으로 계산.

**Q. 부분 오프로드를 해야 하는 것 아닌가?**
아니다. 분할 경계마다 매 토큰 RPC·동기화·fp↔int 변환이 발생 — 기존 0.16 TPS 병목의 재현. qwen3 그래프는 op 9종으로 전부 덮이므로 정상 경로에 미지원 op가 없고, CPU fallback은 안전망일 뿐이다. 미지원 op의 올바른 대응은 커널 추가.

**Q. .idl 확장자가 뭔가?**
Interface Definition Language. QAIC 컴파일러가 .idl에서 호스트 stub / DSP skel 마샬링 코드를 자동 생성한다. protobuf의 .proto와 같은 역할.

**Q. IO 버퍼, dma-buf 등 버퍼 용어 정리.**
dma-buf/ION(커널 공유 버퍼 기술), fd(버퍼 식별자), rpcmem(SDK 래퍼)은 기술 용어. WEIGHTS/KV/ACT/IO/OPLIST는 전부 같은 rpcmem 버퍼에 용도에 따라 붙인 역할 이름일 뿐.

**Q. 캐시 일관성이란?**
CPU/DSP가 DRAM(공유 문서)을 각자 캐시(자기 책상 복사본)로 작업하므로, 쓴 쪽은 flush(DRAM 반영), 읽는 쪽은 invalidate(복사본 폐기 후 재읽기)가 필요. FastRPC 드라이버가 호출 인자로 선언된 버퍼는 자동 처리 → IO 버퍼를 forward의 정식 인자로 선언하는 이유.

**Q. DSP 워커 4개가 맞나? 근거는?**
하드코딩하면 안 되는 값. HVX 유닛 수는 칩마다 다르다(4~6). ggml-hexagon처럼 `qurt_hvx_get_units()`로 런타임 조회해 워커 수 = HVX 유닛 수로 설정. 유닛보다 많은 워커는 벡터 유닛 대기만 늘린다.

**Q. 활성화를 fp16으로 한 이유?**
fp32는 대역폭·버퍼 2배, int8 상주는 residual stream에 양자화 오차가 레이어마다 누적(norm/RoPE/residual은 정밀도 민감). fp16은 v68+ HVX 네이티브 지원 + 민감한 누산만 커널 내부 fp32로 처리하는 표준 절충.

**Q. 레이어당 op 수는? "앞뒤"란?**
정확히 세면 레이어당 16개(norm, q/k/v proj, q/k QK-Norm, rope, attn, o proj, add, norm, gate, up, silu_mul, down, add). 앞뒤 = 레이어 루프 밖의 EMBED(앞), 최종 RMSNORM+MATMUL_LOGITS(뒤). 총 ~451 op.

**Q. KV cache를 int8로 하면? proj도 바뀌나?**
proj는 불변 — 양자화는 ATTN op의 append 시점에 국한. 이득은 KV 메모리·읽기 대역폭 절반(긴 컨텍스트일수록 큼), 비용은 SDPA 복잡화와 K의 민감도(RoPE 이후 채널 outlier). qwen3-0.6b는 KV가 작아 마일스톤 1은 fp16, 포맷 필드로 후속을 열어둠.

**Q. 호스트 lowering이 뭔가?**
컴파일러 용어로 상위 표현→하위 표현 변환. 여기서는 nntrainer 레이어 그래프를 op-list + 정적 버퍼 오프셋 + 타일 파라미터로 바꾸는 초기화 1회 호스트 단계. 결정은 호스트가 미리, DSP는 실행만 하는 원칙의 구현.

**Q. vrmpy가 "int8 4쌍"인데 왜 벡터는 8개인가?**
층위가 다르다. 벡터(128B)는 int32 lane 32개, lane마다 int8 4쌍 곱-누산 → 명령당 128 MAC. "8개"는 K=1024 행이 128B 벡터 8개라는 뜻 — 출력 1개 = vrmpy 8회 + lane reduce 1회.

**Q. decode에서도 가중치를 재사용할 수 없나?**
수학적으로 불가. matvec은 가중치 원소당 정확히 1회 사용(토큰 내 재사용 없음), 토큰 간 재사용은 VTCM 8MB ≪ 0.6GB이고 dense 모델이라 매 토큰 전 가중치를 읽는다. 우회는 재사용이 아니라 읽기량 축소(Q4) 또는 M>1 기법(speculative decoding 등).

**Q. "활성화는 calibration 불필요"의 의미?**
활성화는 런타임에만 존재하므로 scale 결정이 문제인데, static 방식은 대표 입력(calibration 데이터셋)을 미리 흘려 범위를 고정하고, dynamic 방식은 실제 행의 absmax로 그때그때 산출한다. 우리는 dynamic이라 오프라인 준비물이 없고, 정확도도 항상 실값 범위에 맞아 유리. 비용(max 찾기)은 ~0.1%로 무시 가능.

**Q. 정확도 검증 장치 상세?**
[04-e2e-verification](04-e2e-verification.md)의 3-way 기준선 참고 — ①원본 fp32, ②fake-quant fp32(양자화 손실만), ③DSP. ①vs②로 양자화 설계를, ②vs③으로 커널 구현을 각각 판정하고, 디버그 모드 + 이진 탐색으로 첫 발산 op를 ~9회 비교로 특정.
