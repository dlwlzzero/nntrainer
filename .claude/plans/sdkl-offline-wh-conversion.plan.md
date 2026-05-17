# Plan: SDKL Matmul용 오프라인 WH-레이아웃 Weight 변환

**작성일**: 2026-05-17
**브랜치**: hexkl_integration
**입력 모드**: Conversational (free-form 요청)
**복잡도**: Medium–High
**상태**: pending (사용자 승인 대기)

## 요약

현재 SDKL 행렬곱 경로는 런타임 첫 matmul 호출 시 ARM CPU에서
`sdkl_cpu_rm_to_wh_f16_inplace()`로 weight를 row-major `[N,K]` → HMX **WH 레이아웃**
으로 변환한다. 이 변환을 런타임에서 분리하여 `weight_converter_hmx.py`에서
오프라인으로 미리 처리하고, `.bin` 자체가 WH 레이아웃을 담도록 한다. 추가로 동일한
pre-converted WH weight를 NPU macro / micro 레벨에서도 주입 가능한지 검증한다.

선택된 접근: **WH 레이아웃 numpy 재구현** (proprietary 함수 ctypes 호출이 아님).
범위: **CPU Macro 경로 구현 + macro/micro 레벨 검증까지**.

## 현재 동작 (기준선)

1. **오프라인**: `weight_converter_hmx.py` → FP16 row-major `[N,K]` weight를 `.bin`에 저장
2. **런타임 (첫 matmul, ARM CPU)**: `sdkl_cpu_rm_to_wh_f16_inplace()`로 WH 변환 → 캐시 저장
3. **이후 호출**: 캐시된 WH weight 재사용

→ 목표: 2번을 떼어내어 오프라인(1번)으로 이동.

## 코드베이스 조사 결과

| 사실 | 출처 | 의미 |
|---|---|---|
| 변환 함수 `sdkl_cpu_rm_to_wh_f16_inplace(n_row, n_col, W)` | `sdkl.h:812`, `float_tensor.cpp:1016` | 분리 대상인 "CPU 변환" |
| **in-place**, 버퍼 크기 `n_row*n_col` 고정 | `sdkl.h:792-812` | WH 레이아웃은 N·K개 16비트 워드의 **순수 순열** — 패딩 없음. numpy 재구현 가능 |
| HMX f16 블록 = **32×32** | `hexkl_micro.h:60-86` | 타일 구조 → 타일별 역공학 가능 (타일당 ≤1024 고유값) |
| `libsdkl.so` = **ARM aarch64 전용** | `lib/{armv8,armv9}_*` | 역공학/검증에 ARM 디바이스 또는 QEMU-user 필요. 최종 변환기는 x86 순수 numpy |
| WH 변환은 **비공개** (SDK에 레퍼런스 구현 없음) | examples는 호출만 함 | 순열 규칙을 실험으로 역공학해야 함 |
| 런타임 캐시는 weight 포인터 키, `npu_alloc`+`memcpy`+`rm_to_wh`로 채움 | `float_tensor.cpp:1004-1026` | pre-WH weight면 `rm_to_wh` 호출만 게이팅으로 생략 |
| `.bin`은 헤더 없는 raw 텐서 연속체 | `weight_converter_hmx.py` | 레이아웃 marker는 파일 밖(env/파일명)으로 처리 |
| `hexkl_macro_rm_to_wh_f16_inplace` 시그니처 동일 | `hexkl_macro.h:348` | macro WH 레이아웃이 CPU-macro와 동일할 가능성 높음 — 검증 필요 |

## 따라야 할 패턴

| 분류 | 출처 | 패턴 |
|---|---|---|
| 변환기 스타일 | `weight_converter_hmx.py:31-40` | `save_linear_fp16()`이 `np.float16 [N,K]` 기록 — `.tofile()` 직전에 WH permute 삽입 |
| 런타임 게이팅 | `float_tensor.cpp:34-36` (`CDSP_DOMAIN_ID` `#define`) | 동일하게 가벼운 플래그 방식 사용 |
| SDKL 경로 | `float_tensor.cpp:999-1041` | 캐시 유지, `cpu_rm_to_wh_f16_inplace` 호출만 조건부 생략 |
| 테스트 하니스 | `test/unittest/unittest_htp_kernels.cpp` | 레이아웃 동치 비교 케이스 추가 |

## 변경 대상 파일

| 파일 | 작업 | 이유 |
|---|---|---|
| `tools/sdkl_wh_probe/` (ARM 도구) | CREATE | Phase 0 — WH 레이아웃 역공학 하니스 (1회성 R&D) |
| `Applications/CausalLM/res/common/wh_layout.py` | CREATE | numpy WH 변환 구현 |
| `Applications/CausalLM/res/qwen3/qwen3-4b/weight_converter_hmx.py` | UPDATE | `--prewh` 플래그 추가 |
| `nntrainer/tensor/float_tensor.cpp` | UPDATE | pre-WH weight면 `rm_to_wh` 생략 (env 플래그 게이팅) |
| `test/unittest/unittest_htp_kernels.cpp` | UPDATE | pre-WH 경로 테스트 케이스 |
| `nntrainer/tensor/htp_backend/README.md` | UPDATE | Runtime Data Flow 갱신 |
| `docs/how-to-use-htp-backend.md` | UPDATE | pre-WH 사용법 |
| `nntrainer/tensor/htp_backend/HMX_OPS_PLAN.md` | UPDATE | macro/micro 호환성 결론 추가 |

## 단계별 계획

### Phase 0 — WH 레이아웃 역공학 (핵심 마일스톤, 1회성 R&D, ARM 필요)
- ARM 도구 `tools/sdkl_wh_probe` 작성: `libsdkl.so` dlopen → `sdkl_cpu_rm_to_wh_f16_inplace` 호출
- 역공학 전략 (fp16 버퍼를 raw `uint16`로 취급 — 변환은 값과 무관):
  1. **타일 내부 순열**: 32×32 타일 1개에 1024개 고유값 → 32×32→1024 인덱스 맵 추출
  2. **타일 순서**: 각 32×32 타일이 상수 `= tile_index`인 `[N×K]` 행렬 → 타일 선형 배열 순서 추출
  3. **엣지 케이스**: N 또는 K가 32의 배수가 아닐 때 동작 (in-place ⇒ 32 배수 요구 가능성 높음)
- **최우선 검증**: 변환이 순수 전단사(bijection)인지 확인. 아니면 numpy 재구현 불가 → 재논의
- 산출물: `WH(N,K)` 순열 규칙 명세 문서

### Phase 1 — numpy WH 구현 + 골든 검증
- `wh_layout.py`에 `rm_to_wh_f16(w: np.ndarray[N,K] float16) -> np.ndarray` 구현
- numpy 출력이 `libsdkl` 출력과 **바이트 단위 동일**한지 다수 shape로 검증
  (정사각/세로/가로/실제 Qwen3 차원)

### Phase 2 — `weight_converter_hmx.py` 통합
- `--prewh` 플래그 추가: 설정 시 `save_linear_fp16()`이 `.tofile()` 전에 `rm_to_wh_f16()` 적용
- 출력 파일명에 레이아웃 표시 (예: `nntr_qwen3_..._w16a32_whpre.bin`)
- 비-linear weight (norm/embedding)는 영향 없음 — FP32 RM 유지

### Phase 3 — 런타임: pre-WH weight 변환 생략
- 게이팅 플래그 `NNTRAINER_SDKL_WEIGHTS_PREWH` 추가 (env var, `CDSP_DOMAIN_ID` `#define` 스타일)
- `float_tensor.cpp:1011-1021` 캐시 채우는 부분: 플래그 설정 시 `cpu_rm_to_wh_f16_inplace`
  호출 없이 `npu_alloc` 버퍼로 `memcpy`만 수행
- `.bin` 헤더/로더 변경 없음 (안전). 시작 시 어느 레이아웃을 기대하는지 로그 출력

### Phase 4 — macro / micro 동치 검증
- **Macro**: `hexkl_macro_rm_to_wh_f16_inplace`가 CPU macro와 동일한 WH 레이아웃을
  생성하는지 확인 (on-DSP 출력 vs 골든). 동일하면 pre-WH `.bin`을 NPU macro 경로에 그대로 사용
- **Micro**: `hexkl_micro_hmx_rm_to_wh_f16`는 DSP에서 `(row_tile, col_tile)` 주소로 32×32
  타일 단위 변환 (`test_hexkl_micro_hmx_mm_f16.c:164`). micro 타일 내부 WH 레이아웃이
  CPU-macro 타일 내부 레이아웃과 같은지 판정. 같으면 pre-WH 타일을 직접 복사하고
  per-tile `rm_to_wh` 생략 가능
- 산출물: macro 가능 여부 / micro 가능 여부 + 필요한 shim 문서화 → `HMX_OPS_PLAN.md`에 반영

### Phase 5 — 테스트 & 문서
- Python: `wh_layout` numpy vs libsdkl 골든 테스트
- C++: `unittest_htp_kernels.cpp`에 pre-WH 경로 케이스 추가
  (pre-WH bin + 플래그 결과 == 런타임 변환 결과)
- 문서: `htp_backend/README.md` Runtime Data Flow §1, `docs/how-to-use-htp-backend.md`,
  `HMX_OPS_PLAN.md` macro/micro 호환성 노트

## 의존성

- **ARM 디바이스 또는 QEMU-user aarch64** — Phase 0 역공학, Phase 1/4 골든 캡처에 필수
  (배포되는 변환기 자체에는 불필요)
- Hexagon SDK `/local/mnt/.../hexkl_addon` 존재 확인됨
- DSP 빌드 툴체인 — Phase 4 micro/macro on-DSP 검증에만 필요

## 검증

```bash
# Phase 1: numpy WH vs libsdkl 골든 비교
python3 Applications/CausalLM/res/common/test_wh_layout.py

# Phase 2: pre-WH bin 생성
python3 Applications/CausalLM/res/qwen3/qwen3-4b/weight_converter_hmx.py \
  --model_path ./Qwen3-0.6B --output_name ./nntr_qwen3_0_6b_w16a32_whpre.bin --prewh

# Phase 5: C++ 단위 테스트
meson test -C build htp_kernels
```

## 리스크

| 리스크 | 가능성 | 완화책 |
|---|---|---|
| WH 변환이 순수 순열이 아님 (값 재인코딩) | 낮음 | in-place + 동일 크기 시그니처가 순열을 강하게 시사. Phase 0에서 최우선 검증 — 거짓이면 온디바이스 변환으로 폴백 |
| WH 레이아웃이 N/K 32 배수 요구 | 중간 | Phase 0 엣지 케이스 검사. Qwen3 차원은 대부분 128 배수 — 위반 weight는 변환 시 경고 |
| `libsdkl.so` 변경 시 numpy 레이아웃 드리프트 | 중간 | 변환기에 SDK 버전 고정. C++ 런타임 변환 경로를 기본 폴백으로 유지 |
| macro/micro WH 레이아웃이 CPU-macro와 상이 | 중간 | Phase 4는 검증 전용 — 부정 결과도 유효한 결론, 차단 요인 아님 |
| 플래그+bin 조합 오류 시 조용히 결과 깨짐 | 영향 큼 | 파일명 규칙(`_whpre.bin`) + 시작 로그 + 불일치 방어 테스트 |

## 승인 기준

- [ ] WH 변환이 순수 순열임을 Phase 0에서 입증
- [ ] numpy `rm_to_wh_f16`이 libsdkl 출력과 바이트 일치
- [ ] `weight_converter_hmx.py --prewh`가 WH 레이아웃 `.bin` 생성
- [ ] 런타임이 `NNTRAINER_SDKL_WEIGHTS_PREWH` 시 변환 생략, 결과 동일
- [ ] macro/micro 주입 가능 여부 문서화
- [ ] 테스트 통과, 문서 갱신

## 미결 사항 (진행 전 확인 필요)

선택된 "numpy 재구현" 방식은 Phase 0에 ARM 환경(실기기 adb 또는 QEMU-user
aarch64)이 필수다. 둘 다 없으면 numpy 재구현 검증이 불가능하므로 변환 실행 환경
선택지를 재논의해야 한다.
