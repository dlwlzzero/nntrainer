# HTP 긴 프롬프트 prefill 재측정 (mmap-read=false 정정 빌드) — S25 Ultra (V79)

> 정정 대상: [`e2e-s25ultra-longprompt-20260619-140828.md`](e2e-s25ultra-longprompt-20260619-140828.md) (Task 1).
> 사유: Task 1 측정에 쓴 디바이스 배치 `libnntrainer.so`가 **`-Dmmap-read=false`를 빠뜨린 빌드**(= `MMAP_READ=1`, weight를 mmap으로 lazy 로드)였다. how-to-use-htp-backend.md가 HTP 디바이스 빌드에 명시한 플래그를 적용한 정정 빌드로 동일 측정을 재수행한다.

- **일시:** 2026-06-19 14:48 KST
- **단말:** Galaxy S25 Ultra (SM-S938N), Snapdragon 8 Elite (SM8750), Hexagon **V79**, ADB `R3CY205ZMND`
- **정정 빌드:** `libnntrainer.so`를 **`-Dmmap-read=false -Denable-htp=true -Dwerror=false`** 로 재빌드(`MMAP_READ=0`, weight를 시작 시 eager read)하여 V79 e2e 배치 경로(`/data/local/tmp/htp_e2e_v79/lib/`)의 **`libnntrainer.so`만 교체**. 나머지(app `nntrainer_causallm`, `libcausallm_core.so`, `libhtp_ops*.so`)는 Task 1과 동일 바이너리 — `mmap-read` 외 변수를 고정. (이전 `libnntrainer.so`는 `libnntrainer.so.mmapread_true.bak`로 보존.)
  - 빌드 커밋: 현재 HEAD(`5b226f58` 계열). 685646fc(Task 1 배치)와의 차이는 `float_tensor.cpp` 한 파일뿐이고, 그 변경(`17356cbb`)은 **Q4_0→x4x2 per-call repack 제거**로서 본 측정이 쓰는 사전-repack된 **Q4_0_X4X2** 경로에는 영향이 없다 → q4_0_x4x2 런타임 거동은 Task 1과 동일, 유일한 변수는 `mmap-read`.
- **모델:** Qwen3-0.6B, FC=Q4_0_X4X2 x4x2, embed/norm/lm_head=FP32, 28 layers. weight bin `nntr_qwen3_0.6b_q4_0_x4x2.bin` (870,318,080 bytes, Task 1과 동일).
- **고정 조건:** `num_to_generate=32`, `NNTR_NUM_THREADS=4`, `fsu=false`, 동일 단말·동일 모델 바이너리. 변수는 프롬프트 길이(18/206/450)와 HTP on/off 뿐.
- **실행:** `_htp_work/run_longprompt_bench.sh` (config 3종 swap → 6회 자동 측정). 원본 로그: `_htp_work/longprompt-bench-R3CY205ZMND.mmapread_false.log` (Task 1 로그는 `…mmapread_true.log`로 보존).

## 결과 (mmap-read=false)

| 프롬프트 | prefill 토큰 | HTP prefill | CPU prefill | HTP gen | CPU gen |
|---|---|---|---|---|---|
| P1 (짧음) | 18  | 3479 ms / **5.17 TPS**  | 123 ms / 146.3 TPS | 0.296 TPS | 12.20 TPS |
| P2 (중간) | 206 | 4613 ms / **44.66 TPS** | 404 ms / 509.9 TPS | 0.295 TPS | 11.42 TPS |
| P3 (김)   | 450 | 6282 ms / **71.63 TPS** | 922 ms / 488.1 TPS | 0.293 TPS | 10.76 TPS |

## Task 1(mmap-read=true) 대비 비교

| 측정 | mmap-read=**true** (Task 1) | mmap-read=**false** (정정) | Δ |
|---|---|---|---|
| P1 HTP prefill | 3616 ms / 4.98 TPS | 3479 ms / 5.17 TPS | −137 ms (−3.8%) |
| P2 HTP prefill | 4610 ms / 44.69 TPS | 4613 ms / 44.66 TPS | +3 ms (+0.1%) |
| P3 HTP prefill | 5913 ms / 76.10 TPS | 6282 ms / 71.63 TPS | +369 ms (+6.2%) |
| P1/P2/P3 HTP gen | 0.296 / 0.295 / 0.293 | 0.296 / 0.295 / 0.293 | **동일** |
| P1 CPU prefill | 122 ms / 147.5 TPS | 123 ms / 146.3 TPS | +1 ms |
| P2 CPU prefill | 395 ms / 521.5 TPS | 404 ms / 509.9 TPS | +9 ms |
| P3 CPU prefill | 902 ms / 498.9 TPS | 922 ms / 488.1 TPS | +20 ms |
| P1/P2/P3 CPU gen | 12.42 / 11.59 / 10.98 | 12.20 / 11.42 / 10.76 | ~−0.2 TPS (노이즈) |

## 판정: `mmap-read`는 prefill floor·decode 회귀에 **영향 없음** — Task 1 결론 재확인

1. **HTP prefill 고정 floor는 그대로다.** `mmap-read=false`(weight eager read)에서도 HTP prefill ms는 P1 −3.8% / P2 +0.1% / P3 +6.2%로 **방향이 일정치 않은 run-to-run 노이즈** 수준이며, 체계적 감소가 없다. 선형 적합:

   ```
   mmap=false:  HTP prefill_ms(N) ≈ 3360 + 6.5·N   (intercept ≈ 고정 floor 3.2–3.4s)
   mmap=true (Task 1): HTP prefill_ms(N) ≈ 3515 + 5.3·N
   ```

   intercept(고정 floor)는 두 빌드 모두 **~3.2–3.5s**로 일치. 즉 ~3.5초 floor는 **weight mmap lazy page-fault의 산물이 아니라** 토큰당 196회 RPC(28 layer × 7 proj) 왕복의 합 = **per-op FastRPC 고정비**임이 정정 빌드에서 다시 확인된다.

2. **HTP decode 회귀는 완전히 동일.** HTP gen TPS는 0.296 / 0.295 / 0.293으로 Task 1과 **소수점까지 동일**. decode는 항상 M=1·토큰당 196 RPC라 weight 로드 방식과 무관하다. 구조적 회귀임이 재확인.

3. **CPU 경로도 변화 없음(노이즈 내).** CPU prefill/gen 모두 Task 1 대비 ±수% 이내. 특히 Task 1이 "model load noise"로 의심했던 **P1 CPU prefill 147.5→146.3 TPS**도 eager read에서 동일하다 → 그 18-토큰 prefill의 낮은 상각률은 mmap lazy fault 때문이 아니라 **토큰 수가 적어 forward 고정 setup이 상각되지 않은 것**. Task 1의 "N≳500 손익분기 추정 정정"(CPU prefill ~500 TPS, HTP 역전 불가) 결론은 그대로 유효.

## 결론

- **정정 효과: 측정값 변화 없음(노이즈 내).** how-to 문서가 요구한 `-Dmmap-read=false`를 적용해도 prefill 고정 floor(~3.5s), decode 회귀(~0.29 TPS, CPU 대비 ~40×), prefill HTP가 CPU를 못 이김(P3 ~6.8×)이 모두 그대로 재현된다.
- **의의:** Task 1 결론(회귀 원인 = per-op FastRPC 고정비, weight 포맷/HMX/로드방식 아님)이 **빌드 플래그 의혹을 제거한 채로 검증**됐다. 후속 Task 2(배치 RPC op으로 호출 수 196→~112 절감)의 전제—floor가 호출 수에 비례—는 유지된다.
- 디바이스의 `libnntrainer.so`는 이후 측정 일관성을 위해 **`mmap-read=false` 빌드로 교체된 상태로 유지**(이전 빌드 `.mmapread_true.bak` 보존).

## 원본 로그 (발췌, mmap-read=false)

```
RUN p1 / HTP   prefill: 18 tokens, 3479 ms, 5.1739 TPS     generation: 32 tokens, 108138 ms, 0.295918 TPS
RUN p2 / HTP   prefill: 206 tokens, 4613 ms, 44.6564 TPS   generation: 32 tokens, 108391 ms, 0.295227 TPS
RUN p3 / HTP   prefill: 450 tokens, 6282 ms, 71.6332 TPS   generation: 32 tokens, 109089 ms, 0.293338 TPS
RUN p1 / CPU   prefill: 18 tokens, 123 ms, 146.341 TPS     generation: 32 tokens, 2624 ms, 12.1951 TPS
RUN p2 / CPU   prefill: 206 tokens, 404 ms, 509.901 TPS    generation: 32 tokens, 2803 ms, 11.4163 TPS
RUN p3 / CPU   prefill: 450 tokens, 922 ms, 488.069 TPS    generation: 32 tokens, 2973 ms, 10.7635 TPS
```
(전체 로그: `Applications/CausalLM/res/qwen3/qwen3-0.6b/_htp_work/longprompt-bench-R3CY205ZMND.mmapread_false.log`)
