# HTP 긴 프롬프트 prefill 재측정 — S25 Ultra (V79)

> Plan Task 1: [2026-06-19-htp-qwen3-0.6b-perf-regression-validate-and-prefill-gate.md](../../superpowers/plans/2026-06-19-htp-qwen3-0.6b-perf-regression-validate-and-prefill-gate.md)
> 목적: e2e-s25ultra-20260619-112938의 회귀 원인이 **per-op FastRPC 고정비**이며 prefill 회귀가 **짧은 프롬프트(18tok)의 산물**인지를, 프롬프트 길이를 18/206/450 토큰으로 늘려 실측 검증.

- **일시:** 2026-06-19 14:08 KST
- **단말:** Galaxy S25 Ultra (SM-S938N), Snapdragon 8 Elite (SM8750), Hexagon **V79**, ADB `R3CY205ZMND`
- **측정 빌드:** 단말 배치 V79 e2e 빌드(`/data/local/tmp/htp_e2e_v79/`, app/libs 2026-06-19 빌드 = `685646fc` 계열, **prefill-gate 미적용 = no-gate baseline**). pwqk0 x4x2 경로. *재빌드/재배치 없이 기존 배치 측정.*
- **모델:** Qwen3-0.6B, FC=Q4_0 x4x2, embed/norm/lm_head=FP32, 28 layers. weight bin 870,318,080 bytes(e2e 게이트와 동일 바이너리).
- **고정 조건:** `num_to_generate=32`, `NNTR_NUM_THREADS=4`, 동일 단말·동일 모델 바이너리. 변수는 프롬프트 길이와 HTP on/off 뿐.
- **실행:** `Applications/CausalLM/res/qwen3/qwen3-0.6b/_htp_work/run_longprompt_bench.sh` (config 3종 swap → 6회 자동 측정). 원본 로그: 같은 디렉토리 `longprompt-bench-R3CY205ZMND.log`.

## 결과

| 프롬프트 | prefill 토큰 | HTP prefill | CPU prefill | HTP gen | CPU gen |
|---|---|---|---|---|---|
| P1 (짧음) | 18  | 3616 ms / **4.98 TPS**  | 122 ms / 147.5 TPS | 0.296 TPS | 12.42 TPS |
| P2 (중간) | 206 | 4610 ms / **44.69 TPS** | 395 ms / 521.5 TPS | 0.295 TPS | 11.59 TPS |
| P3 (김)   | 450 | 5913 ms / **76.10 TPS** | 902 ms / 498.9 TPS | 0.293 TPS | 10.98 TPS |

> P1 재현성: HTP 4.98 / CPU 147.5 TPS는 베이스라인(5.21 / 144)과 일치 — 측정 환경 정상.

## 판정 1: prefill RPC 고정비 floor — **확정 (직접 관측)**

HTP prefill **시간(ms)** 은 토큰이 18→450(25×)로 늘어도 3616→5913ms로 거의 증가하지 않는다. 선형 적합:

```
HTP prefill_ms(N) ≈ 3515 + 5.3·N      (intercept = per-op RPC 고정 floor)
CPU prefill_ms(N) ≈ 2.1·N             (fixed floor ≈ 0, 순수 연산)
```

(검산: HTP P1→P2 기울기 5.29, intercept 3521 / P2→P3 기울기 5.34, intercept 3510 — 매우 일관.)

prefill 시간의 ~3.5초가 **토큰 수와 무관한 고정비**이고, 이는 토큰당 196회 RPC(28 layer × 7 proj)의 왕복 합이다. 분석(`병목 원인 분석` 절)이 주장한 "per-op FastRPC 고정비가 prefill을 지배"가 **수치로 직접 확인**됐다.

## 판정 2: prefill 회귀는 길이에 따라 크게 줄지만 **CPU를 역전하지는 못함** (가설 정정)

HTP prefill TPS는 길이에 따라 급상승(4.98→44.69→76.10, ~15×)하여 **회귀가 크게 완화**된다:

| 프롬프트 | HTP prefill이 CPU보다 느린 배율 |
|---|---|
| P1 (18tok)  | 29.6× slower (3616/122) |
| P2 (206tok) | 11.7× slower (4610/395) |
| P3 (450tok) |  6.6× slower (5913/902) |

그러나 **역전은 일어나지 않는다.** HTP는 (a) ~3.5s 고정 floor + (b) 토큰당 기울기(5.3ms)가 CPU(2.1ms)보다 **커서**, 두 선이 만나지 않고 절대 격차는 오히려 벌어진다.

> **이전 추정(N≳500 손익분기) 정정:** 그 추정은 CPU prefill을 144 TPS로 가정했으나, 그 값은 18-토큰 런의 **모델 로드 노이즈**가 섞인 수치였다. 실제 CPU prefill은 ~500 TPS(2.1ms/token)로 훨씬 빠르다. 0.6B에서 prefill HTP는 측정 범위(≤450tok)에서 CPU를 이기지 못하며, 외삽으로도 기울기 자체가 더 가팔라 역전 불가.

## 판정 3: decode 회귀는 프롬프트 길이와 무관 — **구조적 확정**

HTP generation TPS는 프롬프트 길이와 무관하게 **0.296 / 0.295 / 0.293으로 완전히 평탄**하다. decode는 항상 M=1, 토큰당 196회 RPC이므로 프롬프트 길이로 상각할 여지가 없다(CPU는 12.42→10.98로 KV 컨텍스트 증가에 따라 소폭 감소). HTP decode는 전 길이에서 CPU보다 ~40× 느리다.

→ **decode 회귀는 prefill과 달리 길이로 못 고친다.** M=1을 키울 방법이 없으므로 방식 A(decode를 CPU로 폴백하는 게이트)가 유효함이 재확인된다.

## 결론 (Task 2 입력)

1. **per-op RPC 고정비(~3.5s/forward)가 지배 요인** — 직접 관측됨. 회귀 원인이 weight 포맷·HMX가 아니라 오프로드 입도임이 다시 확정.
2. **prefill `M >= 128` 게이트의 의미 재평가:** 게이트로 큰 M의 prefill을 HTP에 보내도 0.6B에서는 CPU가 여전히 빠르다(P3 6.6×). 즉 0.6B에서는 prefill조차 HTP 이득이 없으므로, **게이트는 0.6B를 사실상 전부 CPU로 보낸다**(= 회귀 제거). 큰-M HTP 이득은 0.6B보다 큰 모델(4B/30B)에서 검증해야 한다.
3. **decode 게이트(방식 A)는 정당** — 길이 무관 ~40× 회귀, M=1은 상각 불가.

## 원본 로그 (발췌)

```
RUN p1 / HTP   prefill: 18 tokens, 3616 ms, 4.97788 TPS    generation: 32 tokens, 108148 ms, 0.295891 TPS
RUN p2 / HTP   prefill: 206 tokens, 4610 ms, 44.6855 TPS   generation: 32 tokens, 108345 ms, 0.295353 TPS
RUN p3 / HTP   prefill: 450 tokens, 5913 ms, 76.1035 TPS   generation: 32 tokens, 109107 ms, 0.29329 TPS
RUN p1 / CPU   prefill: 18 tokens, 122 ms, 147.541 TPS     generation: 32 tokens, 2576 ms, 12.4224 TPS
RUN p2 / CPU   prefill: 206 tokens, 395 ms, 521.519 TPS    generation: 32 tokens, 2760 ms, 11.5942 TPS
RUN p3 / CPU   prefill: 450 tokens, 902 ms, 498.891 TPS    generation: 32 tokens, 2914 ms, 10.9815 TPS
```
(전체 로그: `Applications/CausalLM/res/qwen3/qwen3-0.6b/_htp_work/longprompt-bench-R3CY205ZMND.log`)
