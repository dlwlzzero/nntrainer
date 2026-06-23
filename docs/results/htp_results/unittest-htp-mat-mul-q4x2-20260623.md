# HTP Q4_0 x4x2 HVX dequant 정확성 검증 — Galaxy S25 Ultra

- **일시:** 2026-06-23 KST
- **단말:** Galaxy S25 Ultra (SM-S938N), Snapdragon 8 Elite (SM8750), Hexagon V79, ADB `R3CY205ZMND`
- **DSP_ARCH:** v75 (v75 skel이 V79 HW에서 하위호환 동작)
- **브랜치 / 커밋:** htp_libs_integration @ 02a94cca (Task1 HVX dequant `510bed98`, Task2 nibble-latency gtest `02a94cca` 포함)
- **목적:** Task 1 (`510bed98`) HVX vectorized `dequantize_x4x2_group_q4_0` 정확성 확정
- **실행자:** j2z0.lee

## 배경

S26 Ultra (SM8850)에서 동일 테스트를 시도했으나 nibble MSE=29.46으로 실패. 체계적 디버깅 결과 HVX 코드 버그가 아닌 **v75 skel ↔ SM8850 DSP 세션 불안정** 문제로 판정. S25 Ultra(V79)로 재검증.

## 실행 명령

```bash
adb -s R3CY205ZMND shell "cd /data/local/tmp/htp_test && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./unittest_htp_mat_mul \
  --gtest_filter='nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency*:nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32*'"
```

## 결과 요약

| 테스트 그룹 | 케이스 수 | PASS | MSE 범위 |
|-------------|-----------|------|----------|
| `mat_mul_af32_pwqk0_of32_*` (Q4_0 x4x2 GEMM) | 7 | 7 | 1.1e-9 ~ 8.0e-9 |
| `mat_mul_q4x2_nibble_latency_*` (HVX vector dequant) | 7 | 7 | 9.7e-6 ~ 3.2e-5 |
| **합계** | **14** | **14** | |

**전체 PASS. HVX vectorized dequant 정확성 확정.**

## 원본 로그

```
[==========] Running 14 tests from 1 test suite.
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_32_256_256
Q4_0 x4x2 GEMM: 32 x 256 x 256
 - MSE (vs Q4_0 dequant ref): 1.44977e-09
[       OK ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_32_256_256 (4 ms)
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_32_512_512
Q4_0 x4x2 GEMM: 32 x 512 x 512
 - MSE (vs Q4_0 dequant ref): 2.86642e-09
[       OK ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_32_512_512 (15 ms)
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_32_1024_1024
Q4_0 x4x2 GEMM: 32 x 1024 x 1024
 - MSE (vs Q4_0 dequant ref): 5.09825e-09
[       OK ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_32_1024_1024 (38 ms)
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_1_1024_1024
Q4_0 x4x2 GEMM: 1 x 1024 x 1024
 - MSE (vs Q4_0 dequant ref): 3.83127e-09
[       OK ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_1_1024_1024 (9 ms)
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_32_1024_256
Q4_0 x4x2 GEMM: 32 x 1024 x 256
 - MSE (vs Q4_0 dequant ref): 7.99609e-09
[       OK ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_32_1024_256 (10 ms)
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_32_256_1024
Q4_0 x4x2 GEMM: 32 x 256 x 1024
 - MSE (vs Q4_0 dequant ref): 1.15589e-09
[       OK ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_32_256_1024 (9 ms)
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_28_256_256
Q4_0 x4x2 GEMM: 28 x 256 x 256
 - MSE (vs Q4_0 dequant ref): 1.43191e-09
[       OK ] nntrainer_htp_mat_mul.mat_mul_af32_pwqk0_of32_28_256_256 (3 ms)
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_1_1024_1024
Q4_0 x4x2 vector 1x1024x1024  MSE=1.02171e-05  avg_latency=322us
[       OK ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_1_1024_1024 (14 ms)
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_1_1024_3072
Q4_0 x4x2 vector 1x1024x3072  MSE=9.69221e-06  avg_latency=758us
[       OK ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_1_1024_3072 (36 ms)
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_1_3072_1024
Q4_0 x4x2 vector 1x3072x1024  MSE=3.20609e-05  avg_latency=767us
[       OK ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_1_3072_1024 (36 ms)
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_32_1024_1024
Q4_0 x4x2 vector 32x1024x1024  MSE=1.0003e-05  avg_latency=331us
[       OK ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_32_1024_1024 (41 ms)
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_32_1024_3072
Q4_0 x4x2 vector 32x1024x3072  MSE=1.0013e-05  avg_latency=817us
[       OK ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_32_1024_3072 (105 ms)
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_32_3072_1024
Q4_0 x4x2 vector 32x3072x1024  MSE=3.07933e-05  avg_latency=829us
[       OK ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_32_3072_1024 (106 ms)
[ RUN      ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_256_1024_3072
Q4_0 x4x2 vector 256x1024x3072  MSE=9.96716e-06  avg_latency=1005us
[       OK ] nntrainer_htp_mat_mul.mat_mul_q4x2_nibble_latency_256_1024_3072 (525 ms)
[----------] 14 tests from nntrainer_htp_mat_mul (960 ms total)

[  PASSED  ] 14 tests.
```

## 레이턴시 요약 (nibble vector path, S25 Ultra V79)

| shape (B×K×N) | avg_latency |
|----------------|-------------|
| 1×1024×1024 | 322 µs |
| 1×1024×3072 | 758 µs |
| 1×3072×1024 | 767 µs |
| 32×1024×1024 | 331 µs |
| 32×1024×3072 | 817 µs |
| 32×3072×1024 | 829 µs |
| 256×1024×3072 | 1005 µs |

## 결론

- **HVX vectorized `dequantize_x4x2_group_q4_0` 정확성 확정** (S25 Ultra V79 기준)
- pwqk0 GEMM MSE ~1e-9: Q4_0 dequant ref 대비 수치 오차 없음
- nibble vector MSE ~1e-5: FP16 누적 오차 범위 내 정상
- S26 (SM8850)에서의 MSE=29.46은 HVX 코드 버그가 아닌 v75 skel ↔ SM8850 DSP 환경 불안정으로 판정됨
