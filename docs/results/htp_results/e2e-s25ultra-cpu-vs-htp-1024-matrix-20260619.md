# HTP vs CPU 4-way Matrix 결과 — S25 Ultra (V79), Qwen3-0.6B, 1024-token prefill

- **일시(date/time):** 2026-06-19 (오후), KST
- **단말:** Galaxy S25 Ultra (SM-S938N), Snapdragon 8 Elite (SM8750), Hexagon **V79**, arm64-v8a, Android 15, ADB `R3CY205ZMND`
- **브랜치:** htp_libs_integration
- **모델:** Qwen3-0.6B (28 layers, hidden=1024, intermediate=3072, heads=16, KV-heads=8, head_dim=128, vocab=151936, tied embeddings)
- **입력 프롬프트:** 1024 tokens (고정; Qwen3-0.6B HF 토크나이저 검증 완료)
- **num_to_generate:** 512 (R1·R3·R4), 32 (R2 — Q4_0_X4X2 decode 속도 제약으로 단축)
- **실행자:** j2z0.lee

---

## 결과 요약

| Run | 백엔드 | 가중치 형식 | prefill (ms) | prefill TPS | gen (ms) | gen TPS | peak RSS (KB) | e2e (ms) |
|-----|--------|------------|--------------|-------------|----------|---------|---------------|----------|
| **R1** | **HTP (DSP)** | **pwf16** (HMX 타일 FP16) | 2,081 | **492.1** | 51,122 | **10.02** | 1,932,396 | 54,494 |
| **R2** | **HTP (DSP)** | **Q4_0_X4X2** (scalar dequant) | 12,151 | 84.3 | 108,702 | 0.29 ⚠️ | 1,383,412 | 121,738 |
| **R3** | **CPU** | **plain FP16** | 10,014 | 102.3 | 121,607 | 4.21 | 1,938,780 | 132,832 |
| **R4** | **CPU** | **plain Q4_0** | 2,497 | **410.1** | 15,381 | **33.29** | 1,364,288 | 18,740 |
| **R5** | **HTP (DSP)** | **Q4_0_X4X2** (HVX vector dequant) | 2,264 | **452.3** | 5,302 | **6.04** | 1,383,556 | 7,614 |

> R2 gen ⚠️: `fastrpc_mmap failed err=1` 반복 발생으로 decode throughput 오염; 32 토큰만 생성(~108초). R2 prefill도 mmap 오류로 오염 추정.
> R5 gen: 32 토큰, mmap 오류 없음. run1(452.3 TPS / 6.04 TPS) vs run2(439.7 TPS / 5.77 TPS), 변동 ~3%.
> R5 커밋: `510bed98` (HVX vector `dequantize_x4x2_group_q4_0`, 2026-06-22), 측정일: 2026-06-23.

---

## 상세 관측 및 해석

### R1 — HTP pwf16: 가장 빠른 prefill (HTP 기준)

- DSP 세션 정상 개방, `hmx_mat_mul_af32_pwf16_of32` 커널 실행
- prefill 492 TPS: 1024-token 배치에서 HMX 행렬곱이 RPC 오버헤드를 압도
- generation 10 TPS: decode(M=1)에서도 pwf16 커널이 per-op RPC 비용을 상쇄, 실용 가능 수준
- 가중치 파일 1.44 GB (embed FP32 + FC FP16 타일 permute, no tied lm_head)
- peak RSS ~1.88 GB (FP16 가중치 + KV cache + 활성화 버퍼 합산)

### R2 — HTP Q4_0_X4X2: prefill 느림, decode 비실용

- prefill 84 TPS: Q4_0_X4X2 SOA 레이아웃은 `htp_ops_mat_mul_af32_pwqk0_of32` 커널 의존; 1024-row 배치에서도 pwf16(R1) 대비 ~5.8× 느림
- `fastrpc_mmap failed err=1` 반복 발생: decode 중 per-op 공유메모리 할당 실패; DSP 세션은 유지되나 decode throughput 0.29 TPS
- 32 토큰으로 단축 후 완주(108초)
- peak RSS ~1.35 GB — 가중치 870 MB(Q4_0_X4X2, 4-bit + 2-byte scale/block)
- **결론:** Q4_0_X4X2 경로는 prefill-전용(대형 배치) 시나리오에서만 유효; decode 실용 불가

### R3 — CPU plain FP16: 기준선 (CPU 대비 비교용)

- `HtpInterface: failed to load libhtp_ops.so` 예상 메시지 출력(CPU 빌드에는 libhtp_ops.so 없음) — 정상
- prefill 102 TPS: 1024-token CPU GEMM (FP16 weights, FP32 activations, NNTR_NUM_THREADS=4)
- generation 4.21 TPS: decode M=1 CPU FP16 matmul; 메모리 대역폭 의존적
- peak RSS ~1.89 GB: FP16 FC 가중치 ~1.43 GB + embed FP32 + KV cache
- **NOTE:** config.json 초기 부재로 한 번 실패 후 재실행; 이후 정상 완주

### R4 — CPU plain Q4_0: CPU 최고 효율

- prefill 410 TPS: Q4_0 dequant-on-the-fly + CPU GEMM — 가중치 BW 절감으로 R3 대비 4× 빠름
- generation 33.3 TPS: decode M=1에서도 Q4_0 dequant 이점 유지; 4개 스레드 충분히 활용
- peak RSS ~1.33 GB: 가중치 870 MB(4-bit) + embed FP32 + KV cache
- e2e 18.7초: 4-way 중 **가장 짧은 e2e**(512 gen 기준)
- **결론:** 오늘 측정 범위에서 CPU Q4_0이 throughput·memory 모두 우세한 단일 최적점

---

## 비교 분석

### prefill (1024 tokens)

```
R1  HTP pwf16            492 TPS  ████████████████████████████████████  (1×)
R5  HTP Q4_0_X4X2 HVX   452 TPS  █████████████████████████████████     (0.92×)
R4  CPU Q4_0             410 TPS  ████████████████████████████████      (0.83×)
R3  CPU FP16             102 TPS  ████████                               (0.21×)
R2  HTP Q4_0_X4X2 scal   84 TPS  ██████ ⚠️mmap오류                    (0.17×)
```

- R5(HVX dequant) 452 TPS: R2(scalar dequant) 대비 ~5.4× 향상, R1(pwf16) 대비 0.92×
- R5는 CPU Q4_0(R4, 410 TPS)도 앞섬: HTP Q4_0_X4X2 경로가 1024-tok에서 CPU를 첫 역전
- R2 prefill은 fastrpc_mmap 오류로 오염; 실제 scalar 기준선은 longprompt 측정값(~115 TPS @ 1024 tok) 참조

### generation (decode, M=1)

```
R4  CPU Q4_0             33.3 TPS  ███████████████████████████████████  (1×)
R1  HTP pwf16            10.0 TPS  ██████████                           (0.30×)
R5  HTP Q4_0_X4X2 HVX    6.0 TPS  ██████                               (0.18×)
R3  CPU FP16              4.2 TPS  ████                                 (0.13×)
R2  HTP Q4_0_X4X2 scal   0.29 TPS █ ⚠️mmap오류                       (0.009×)
```

- R5 gen 6.0 TPS: R2(0.29 TPS, mmap 오류) 대비 ~20× 향상, scalar 비오염 기준(longprompt ~0.29 TPS) 대비도 ~20×
- R4 CPU Q4_0(33 TPS)가 여전히 decode 1위; R1 HTP pwf16(10 TPS)보다 R5가 느림
- R5 decode 6 TPS: 실용 가능 수준으로 개선 (R2의 "decode 비실용" 판정 번복)

### 메모리 (peak RSS)

| Run | peak RSS |
|-----|----------|
| R1 HTP pwf16 | **1,932 MB** |
| R3 CPU FP16  | **1,939 MB** |
| R2 HTP Q4_0_X4X2 | **1,350 MB** |
| R4 CPU Q4_0  | **1,332 MB** |

- FP16(R1·R3) ~1.9 GB vs Q4_0(R2·R4) ~1.35 GB: 가중치 BW 절감이 RSS에 직결
- mmap-read=false(R1·R2) vs mmap-read=true(R3·R4) 차이도 있으나, 가중치 dtype 영향이 지배적

---

## 종합 결론

| 목적 | 권장 경로 | 이유 |
|------|-----------|------|
| 최고 prefill 속도(1024 tok) | **HTP pwf16 (R1)** | 492 TPS, HMX 배치 이점 (R5와 격차 8%) |
| 최저 메모리 + prefill HTP급 | **HTP Q4_0_X4X2 HVX (R5)** | 452 TPS, peak RSS 1.35 GB (vs pwf16 1.88 GB) |
| 최고 decode throughput | **CPU Q4_0 (R4)** | 33 TPS, 메모리 절감 + 스레드 활용 |
| HTP 단독 배포(decode 포함) | **HTP pwf16 (R1)** | 10 TPS; R5도 6 TPS로 실용 가능해짐 |
| 최저 메모리 + 빠른 decode | **CPU Q4_0 (R4)** | 1.33 GB peak, e2e 18.7s |

> **R5 추가 후 핵심 시사점 업데이트:**
> HVX vector dequant(`510bed98`)로 Q4_0_X4X2 경로의 prefill이 **CPU Q4_0을 역전(452 vs 410 TPS)**했다.
> decode도 0.29→6.0 TPS로 20× 개선되어 실용 가능 수준이 됐다(CPU 대비 ~0.18×, HTP pwf16 대비 ~0.6×).
> 메모리(1.35 GB)를 제약으로 하는 배포에서는 **HTP Q4_0_X4X2 HVX**가 pwf16과 동등한 prefill을 훨씬 적은 메모리로 제공하는 유효한 선택지가 됐다.

---

## 빌드·배포 정보

### HTP 빌드 (R1·R2)
- Meson 옵션: `-Dmmap-read=false -Denable-htp=true`
- ndk-build: `APP_MODULES="nntrainer_causallm" ENABLE_HTP=1`
- 배포 경로: `/data/local/tmp/htp_e2e_v79/`
- ADSP_LIBRARY_PATH: `/vendor/lib/rfsa/adsp;/data/local/tmp/htp_e2e_v79/lib` (세미콜론 구분자)

### CPU 빌드 (R3·R4)
- Meson 옵션: `-Dmmap-read=true` (libhtp_ops.so 없음)
- ndk-build: `APP_MODULES="nntrainer_causallm"` (ENABLE_HTP 미설정)
- 배포 경로: `/data/local/tmp/cpu_e2e/`
- libc++_shared.so: NDK r25c에서 직접 복사

### 가중치 변환
| Run | 변환 스크립트 | 파일 크기 |
|-----|-------------|----------|
| R1 | weight_converter_hmx.py | 1,503,395,840 bytes (FP16 타일+embed FP32) |
| R2 | weight_converter_htp.py | 870,318,080 bytes (Q4_0_X4X2 SOA+embed FP32) |
| R3 | weight_converter_cpu_fp16.py | 1,503,395,840 bytes (plain FP16+embed FP32) |
| R4 | weight_converter_cpu_q4_0.py | 870,318,080 bytes (plain Q4_0 block+embed FP32) |

---

## 비고

- **R2 gen 토큰 수 단축:** Q4_0_X4X2 decode 0.29 TPS → 512 tok = ~29분. 실측 32 tok으로 단축. prefill 지표는 완전 1024-token 측정값.
- **R3 config.json 부재 초기 실패:** CPU 모델 디렉터리에 config.json이 없어 첫 실행 실패. HTP 모델 디렉터리에서 복사 후 정상 완주.
- **CPU mmap-read=true:** R3·R4는 eager load 없이 mmap 지연 로딩. peak RSS가 점진적으로 증가하며 최종값이 실제 RSS.
- **NNTR_NUM_THREADS=4:** 모든 런 동일 설정.
