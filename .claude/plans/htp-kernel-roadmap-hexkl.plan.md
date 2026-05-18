# nntrainer HTP 커널 구현 로드맵 (hexKL 기반)

**작성일**: 2026-05-18
**브랜치**: hexkl_integration
**상태**: pending (사용자 승인 대기)
**관계**: `HMX_OPS_PLAN.md` rev.2의 후속(rev.3) — matmul 한정으로 rev.2의
"raw intrinsic 채택" 결정을 번복한다.

## 0. 제약의 정확한 해석과 영향

**제약**: HMX로 수행 가능한 모든 matmul 연산은 hexKL API를 통해 구현한다.
raw HMX intrinsic(`hmx_utils.h` inline asm, llama.cpp `Q6_mxmem_*`)은
matmul에 **사용 불가**.

### hexKL의 3계층 API (현재 브랜치 조사 기준)

| 계층 | 헤더 / 라이브러리 | 실행 위치 | 현재 사용 | matmul 능력 |
|---|---|---|---|---|
| CPU Macro (SDKL) | `sdkl.h` / `libsdkl.so` (dlopen) | 호스트, FastRPC 내장 | 사용 중 | `sdkl_npu_mm_f32f16_f32` 1종뿐 |
| NPU Macro | `hexkl_macro.h` | DSP (자체 skel 안) | 미사용 | matmul 매크로 + `rm_to_wh` |
| Micro | `hexkl_micro.h` / `libhexkl_micro.a` | DSP (자체 skel 안) | 미사용 | 32x32 f16 타일 HMX 프리미티브 (`hexkl_micro_hmx_mm_f16`) |

### 영향 범위

- matmul(FC), attention의 QK^T / softmax x V 행렬곱 → hexKL 경유 필수
- flash attention 내부 행렬곱도 hexKL micro 타일로 분해
- 양자화 weight matmul → HVX로 f16 dequant 후 hexKL micro에 투입
  (matmul 자체는 hexKL 유지)
- 비-matmul HVX op(rms_norm, rope, softmax, activation, elementwise)는
  제약 밖 — 자체 HVX 커널 또는 llama.cpp 포팅 자유

## 1. 핵심 아키텍처 결정 — Micro 기반 자체 skel

**CPU Macro만으로 불충분한 이유**: 현재 경로(`sdkl_npu_mm_f32f16_f32`)는
단일 변형, FP32 act 강제, 커널 fusion 불가, op 추가가 QC 의존.
CausalLM에 필요한 f16xf16 / batched / GQA / flash-attn 융합을 못 한다.

**결정**: hexKL Micro API 위에 자체 DSP skel(`libnntr_htp_skel.so`)을 구축.

- HMX 명령어 발행은 전적으로 `hexkl_micro`/`hexkl_macro` 안에서 일어남 →
  제약 충족
- 타일링/비용모델/디스패치/op 융합은 자체 skel 코드 (HMX intrinsic 아님 →
  제약 무관)
- 기존 CPU Macro 경로(`sdkl_interface.h`)는 안정 fallback으로 보존
  (빌드 옵션 `htp-backend=sdkl|micro`)

```
   호스트 (ARM)                          Hexagon DSP - libnntr_htp_skel.so
 +--------------------+                +----------------------------------+
 | nntrainer          |  FastRPC/      | op dispatcher                    |
 |  float_tensor.cpp  |  dspqueue      |  - matmul   -> hexkl_micro_hmx_mm | HMX = hexKL
 |  HtpBackend 추상화 | <----------->  |  - flash_attn -> micro mm + HVX sm| 행렬곱만 hexKL
 |  weight cache      |  op batch      |  - rms_norm/rope/act -> HVX 커널  | 자유
 |  htp-backend=      |                |  - worker pool (n_hvx)           |
 |   sdkl | micro     |                | links: libhexkl_micro.a          |
 +--------------------+                +----------------------------------+
        +- fallback: libsdkl.so (CPU Macro, 현행)
```

## 2. 단계별 로드맵

### Phase 0 — 환경/조사 (차단 해소)

- `HEXAGON_SDK_HOME/addons/hexkl_addon`에서 `hexkl_micro.h`/`hexkl_macro.h`
  API 전수 조사: 지원 dtype, 타일 단위, accumulator/bias 시그니처,
  batched 여부, skel 빌드 방법(`libhexkl_micro.a` 링크).
- hexKL micro가 f16xf16, f32 act, accumulation 어디까지 지원하는지 확정 →
  미지원 변형은 HVX 전후처리로 보강 설계.
- DSP arch(v75 등) 및 skel 빌드 툴체인 확정.
- 산출물: hexKL Micro/Macro API 능력 매트릭스.

### Phase 1 — hexKL Micro 단독 검증 (skel 없이)

- 시뮬레이터에서 `hexkl_micro_hmx_mm_f16` + `hexkl_micro_hmx_rm_to_wh_f16`을
  단독 호출 (SDK의 `test_hexkl_micro_hmx_mm_f16.c` 재현).
- 32x32 타일 결과를 CPU 레퍼런스와 0.1% 오차 내 일치 확인.
- WH 레이아웃 규칙 파악 (→ `sdkl-offline-wh-conversion.plan.md`와 연계).

### Phase 2 — matmul 커널 + 자체 skel 최소 골격

- `libnntr_htp_skel.so` 골격: FastRPC IDL(`start/stop` + op batch),
  op dispatcher, `worker_pool`.
- matmul 커널: 큰 `[M,K]x[K,N]`을 32x32 타일로 분해 → 타일마다
  `hexkl_micro_hmx_mm_f16` 호출, accumulation. 타일링/비용모델은 자체 코드
  (llama.cpp `hmx_compute_chunks` 방식 차용 가능 — HMX intrinsic 아님).
- 호스트 `HtpBackend` 추상 인터페이스 도입, `float_tensor.cpp`의 sdkl
  분기를 `htp-backend` 옵션으로 분기.
- 검증: FC matmul 결과가 CPU Macro 경로와 동일.

### Phase 3 — matmul 변형 확대

- f16xf16, FP32 act 입력(전처리 변환), batched matmul, `MUL_MAT_ID`(MoE).
- 양자화 weight: Q4_0/Q8_0를 HVX `vlut16`로 f16 dequant → hexKL micro 투입
  (matmul은 hexKL 유지).
- 호스트 weight cache를 pre-WH `.bin`(별도 plan)과 연동.

### Phase 4 — flash attention (행렬곱만 hexKL)

- QK^T, softmax(P) x V 행렬곱 → 32x32 타일로 분해,
  `hexkl_micro_hmx_mm_f16` 호출.
- softmax/scale/mask는 HVX 커널 (제약 밖).
- GQA(그룹 KV) 처리, 온라인 softmax. 정확도는 llama.cpp
  `flash-attn-ops.c`와 대조.

### Phase 5 — HVX 비-matmul 커널

- `rms_norm`, `rope`, `softmax`, `silu/gelu/swiglu`, elementwise.
- llama.cpp `ggml-hexagon/htp/*-ops.c`에서 포팅 (HMX 아님 → 제약 무관,
  라이선스만 확인).
- CausalLM 디코더 레이어 전체를 DSP에 상주.

### Phase 6 — 통신/배칭 최적화

- 1단계는 안정적 FastRPC direct, 2단계로 op 배칭(`dspqueue` 또는 message
  channel) 도입 — 디코드 루프 저지연화.
- llama.cpp `dspqueue` 방식이 nntrainer spin-poll(`FIXME` 다수)보다 성숙
  → 통신 계층은 llama.cpp 참조.

### Phase 7 — 빌드/테스트/문서

- meson `enable-htp`에 `htp-backend=sdkl|micro` 옵션, skel 빌드
  (android+hexagon) 통합, skel 서명.
- `unittest_htp_kernels.cpp` 확장: matmul/flash-attn/HVX op를
  sim+실기기에서.
- `README.md`/`docs/how-to-use-htp-backend.md`, `HMX_OPS_PLAN.md` rev.3 갱신.

## 3. HMX_OPS_PLAN rev.2 대비 변경점

| 항목 | rev.2 | 본 로드맵 (rev.3) |
|---|---|---|
| HMX matmul 구현 | raw intrinsic (Option 나) | hexKL micro 경유 (제약) |
| 기준 코드베이스 | `htp_libs_integration`(B) 복원 | B의 skel 골격/HVX op만 차용, matmul 커널은 hexKL micro로 재작성 |
| llama.cpp(C) 역할 | HMX 커널 보강 | HVX op/타일링 비용모델/dspqueue만 보강 (HMX intrinsic 차용 불가) |
| CPU Macro 경로 | 제거 검토 | fallback으로 보존 (`htp-backend=sdkl`) |

## 4. 리스크 / 미결정

| 항목 | 내용 |
|---|---|
| hexKL micro 능력 한계 | f16xf16/accumulation/batched 미지원 시 Phase 3/4 설계 변경 — Phase 0에서 최우선 확정 |
| micro 성능 | 타일별 호출 오버헤드가 raw intrinsic 대비 클 수 있음 — Phase 2에서 벤치마크, macro 레벨 활용 검토 |
| skel 빌드 | `libhexkl_micro.a` DSP 링크/서명 절차 미검증 |
| 라이선스 | llama.cpp HVX op 포팅 시 라이선스 표기 |
| WH 레이아웃 | matmul 입력 weight 레이아웃 — `sdkl-offline-wh-conversion.plan.md`와 정합 필요 |

## 5. 다음 단계

1. Phase 0 착수 — hexKL Micro/Macro API 능력 조사 (`HEXAGON_SDK_HOME`
   환경 접근 가능 여부 포함).
2. `htp-backend=sdkl|micro` 빌드 옵션 분기 설계 확정.
3. Phase 0 결과에 따라 Phase 2 matmul 커널 착수.
