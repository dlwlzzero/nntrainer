# HMX Ops 통합 계획 (hmx_ops) — rev.2

> 목표: nntrainer가 Hexagon DSP의 HMX(행렬 가속기) intrinsic 연산을
> `nntrainer/tensor/htp_backend/htp/hmx_ops/`에서 호출할 수 있게 한다.
> 타깃: **Hexagon 시뮬레이터 + Android 실기기** 양쪽.

> **rev.2 변경 사유**: 조사 결과 `hmx_ops`는 `htp_libs_integration` 브랜치에
> 이미 구현되어 있고, llama.cpp `ggml-hexagon`이 더 성숙한 참조 구현임이 확인됨.
> "hexkl_micro.h 위에 처음부터 구축"하던 rev.1 계획을 폐기하고,
> **기존 커스텀 백엔드 복원 + llama.cpp 기준 업그레이드**로 전환.

---

## 1. 현황 — 3개의 코드베이스

### A. 현재 브랜치 `hexkl_integration` (HEAD, 07a3292f)
- `htp_backend/include/sdkl_interface.h` → Qualcomm 공식 prebuilt `libsdkl.so`
  (hexkl_addon **CPU Macro API**) 를 dlopen.
- 지원 op: `sdkl_npu_mm_f32f16_f32` (FP32 act × FP16 weight) **1개뿐**.
- HMX intrinsic을 직접 다루지 않음 — QC 라이브러리에 위임.
- `htp_libs_integration` 대비 58 commits 앞, 7 commits 뒤.

### B. 브랜치 `htp_libs_integration` (~10,000줄 커스텀 백엔드)
- `htp/hmx_ops/` — `mat_mul.c` (1698줄), `flash_attn.c` (1590줄),
  `flash_attn_sp_hdim.c` (832줄)
- `htp/hvx_ops/` — `rms_norm.c`, `precompute_table.c`, `mm_benchmark.c`
- `include/htp/hmx_utils.h` — **raw HMX intrinsic** (inline asm):
  `activation.hf = mxmem`, `weight.hf = mxmem`, `bias = mxmem2`, `cvt.hf = acc`
- `include/htp/` — `hvx_internal.h`, `hvx_math.h`, `hvx_convert.h`,
  `quants.h`, `dma_utils.h`, `vtcm_mgr.h`, `worker_pool.h`, `hmx_mgr.h`
- IDL `htp_ops.idl`, `host/session.c`·`op_export.c`, `htp/commu.c`,
  `op_executor.cc`, `worker_pool.c`, `vtcm_mgr.cc`, `hmx_mgr.c`, `power.c`
- `htp_interface.h` — 호스트 dlopen 로더 (`libhtp_ops.so`)
- CMake (`build_htp.sh`: android + hexagon DSP_ARCH=v75) → `libhtp_ops.so` +
  `libhtp_ops_skel.so`
- 메시지 채널(`OpComputeRequest` via rpcmem) + FastRPC 두 경로 지원
- 유닛테스트 `test/unittest/htp/` (mat_mul 28 / rms_norm 5 / quantizer 2)
- **hexkl_addon 미사용** — 전적으로 raw intrinsic.

### C. `~/Project/llama.cpp` — `ggml/src/ggml-hexagon/` (참조 구현)
- `htp/hmx-matmul-ops.c` (1705줄), `htp/matmul-ops.c` (145KB) 등
- `htp/hmx-utils.h` — B와 동일 계보. **개선점**:
  - 통합 명령어 `mxmem(%0,%1):after.hf = acc` (B는 구식 2-instruction)
  - `hmx_load_tile_pair_fp16` — VTCM 4MB bank-boundary 정밀 버스에러(0x2601) 회피
- HMX 커널: 양자화 weight x4x2 (Q4_0/Q8_0/MXFP4/IQ4_NL), batched matmul,
  `hmx_compute_chunks` 비용모델 기반 타일링
- HVX op 대량 커버리지: act/binary/softmax/rope/flash-attn/ssm-conv/argsort/
  set-rows/get-rows/cpy/repeat/sum-rows/unary
- 백엔드 선택: `htp_iface.idl` → `htp_iface_start(..., use_hmx)`,
  환경변수 `GGML_HEXAGON_USE_HMX`로 HMX/HVX 토글, skel `libggml-htp-v{arch}.so`

### 공통 계보
B와 C 모두 오픈소스 **htp-ops-lib** (github.com/haozixu/htp-ops-lib) 포팅.
→ raw intrinsic 경로가 사실상 표준. **어느 참조 구현도 hexkl_addon을 안 씀.**

---

## 2. 아키텍처 의사결정

| 경로 | 내용 | 장점 | 단점 |
|---|---|---|---|
| **(가) hexkl_addon** | 현재 HEAD. QC 공식 `libsdkl.so`/`libhexkl_micro.a` | 공식 지원, 유지보수 QC | matmul 1종, HMX 직접제어 불가, fused 커널 불가 |
| **(나) raw intrinsic** | B+C 경로. `hmx_utils.h` inline asm | 완전한 제어, flash-attn·양자화 가능, **이미 구현·테스트됨**, 참조 2종 | unofficial, asm 유지보수 |

→ **결정: (나) 채택.** `hmx_ops`의 핵심인 "intrinsic 명령어"는 (나)에만 존재.
(가)는 안정적 fallback으로 빌드 옵션 보존 가능(선택).

→ **rev.1 폐기**: hexkl_micro.h 위 신규 구축은 B를 중복 재구현하는 셈이라 폐기.

---

## 3. 발전된 계획 — 3개 후보

### Option 1 — 커스텀 백엔드 전체 복원 (권장 베이스)
`htp_libs_integration`의 `htp_backend/`(B)를 현재 브랜치로 포팅.
- `hmx_ops`가 즉시 존재 (이미 작성·테스트됨).
- `sdkl_interface.h`(가)는 빌드 옵션으로 병존 or 제거.
- 작업: 선택적 포팅(아래 4절). 클린 머지는 불가 — 두 브랜치에서
  `htp_backend/`가 사실상 각각 재작성되어 충돌.

### Option 2 — 하이브리드
(가) hexkl_addon은 안정 matmul 경로로 유지 + `htp/hmx_ops/`는 (가)가
못 하는 op(flash attention, 양자화 weight)만 선택 포팅.
- 리스크 분산. 단 두 백엔드 동시 유지 비용.

### Option 3 — llama.cpp 기준 신규 이식
B를 건너뛰고 C(`ggml-hexagon`)의 더 성숙한 HMX 커널을 직접 이식.
- 최신 버그픽스·양자화·batched 확보. 단 ggml 의존(`ggml-common.h`,
  `htp_tensor`, `htp-msg.h`)을 nntrainer 타입으로 걷어내는 작업 큼.

→ **권장: Option 1을 베이스로, HMX 커널만 Option 3 방식으로 C에서 보강.**
즉 "B 복원 → `hmx_ops/mat_mul.c`·`hmx_utils.h`를 C 기준 업그레이드".

---

## 4. 권장 경로 실행 계획 (Option 1 + C 보강)

### Phase 0 — 결정·환경
- Option 1/2/3 최종 확정 (§3).
- `hexkl_addon`(가) 경로 유지 여부 확정.
- `source $HEXAGON_SDK_HOME/setup_sdk_env.source`, 툴체인 `HEXAGON_Tools/8.8.06`,
  DSP arch (B는 v75 / C는 v73·v75·v79) 확정.

### Phase 1 — 커스텀 백엔드 포팅 (B → 현재 브랜치)
- `htp_libs_integration:nntrainer/tensor/htp_backend/`의 파일 일습 이식:
  `htp/`, `host/`, `include/`, `htp_ops.idl`, `htp_interface.h`,
  `CMakeLists.txt`, `build_htp.sh`, `run.sh`, `meson.build`.
- 현재 브랜치의 `sdkl_interface.h`/`meson.build`(가)와 충돌 정리:
  - 병존: `enable-htp` 하위에 `htp-backend=sdkl|custom` 옵션 분기.
  - 단일화: (가) 제거 시 `float_tensor.cpp`의 sdkl 분기도 교체.
- 58 commits 분량의 upstream functional API 변경과의 정합성 확인
  (`float_tensor.cpp`, `qnn_context.cpp` 등).

### Phase 2 — 빌드 통합 (sim + android)
- `build_htp.sh`/CMake: `build_cmake android` + `build_cmake hexagon`.
  arch를 v73/v75/v79 인자화 (현재 v75 고정).
- meson `enable-htp`에서 host stub + DSP skel 빌드·설치 연결.
- Android: skel 서명(`elfsigner` testsig), 배치 경로(`/vendor/lib/rfsa/adsp/`).
- 시뮬레이터: `hexagon-sim` 실행 타깃 + skel 단독 검증.

### Phase 3 — HMX 커널 업그레이드 (C 기준)
- `include/htp/hmx_utils.h`: `hmx_consume_accumulator_fp16`을 C의 통합
  명령어 `mxmem():after.hf = acc`로 교체, `hmx_load_tile_pair_fp16` +
  4MB bank-boundary 가드 도입.
- `htp/hmx_ops/mat_mul.c`: C `hmx-matmul-ops.c`의 `hmx_compute_chunks`
  비용모델 타일링, 양자화 weight(x4x2) 경로 반영.
- `hmx_ops/flash_attn*.c`: C `flash-attn-ops.c`와 대조해 GQA·정확도 보강.

### Phase 4 — nntrainer 통합
- `float_tensor.cpp` matmul 경로: `HtpInterface`(커스텀) 우선 →
  (가) 유지 시 sdkl fallback → CPU fallback.
- 빌드 옵션·런타임 플래그(예: `NNTRAINER_HTP_BACKEND`)로 경로 선택.

### Phase 5 — 테스트·검증
- 시뮬레이터: `hexagon-sim`으로 skel 단독, HMX 결과 vs C 레퍼런스
  (0.1% 허용오차).
- Android: `run.sh`로 `adb push` 후 `test/unittest/htp/` 실행
  (mat_mul/rms_norm/quantizer).
- chan(메시지 채널) vs FastRPC 두 경로 MSE 일치 + 지연 비교.

### Phase 6 — 문서·정리
- `htp_backend/README.md`(B버전, 디렉터리맵 포함) 채택.
- `docs/how-to-use-htp-backend.md` 갱신.
- 미추적 `build_htp/` 잔여 산출물 정리.
- 이후 확장: C의 HVX op군(softmax/rope/flash-attn)을 `hvx_ops/`에 점진 이식.

---

## 5. 리스크 / 미결정

| 항목 | 내용 |
|---|---|
| 브랜치 충돌 | `htp_backend/`가 두 브랜치에서 재작성 → 클린 머지 불가, 선택 포팅 필요 |
| (가)vs(나) 병존 | 두 백엔드 유지 비용 vs 단일화 시 회귀 위험 |
| ggml 의존 제거 | Option 3 보강 시 C 커널의 `ggml-common.h`·`htp_tensor` 의존 분리 필요 |
| DSP arch | B는 v75 고정, C는 v73/v75/v79 — 타깃 기기에 맞춰 인자화 |
| Android skel 서명 | 실기기 로드 시 `elfsigner` testsig 필요 (sim 불필요) |
| upstream API 정합 | HEAD의 58 commits(functional API 재이식)와 B 코드의 호환 검증 |
| htp-ops-lib 라이선스 | B·C 모두 htp-ops-lib 포팅 — 라이선스/저작권 표기 확인 |

---

## 7. Message Channel 전달 방식 분석 (htp-ops-lib 계보)

`htp_libs_integration` 백엔드는 호스트↔DSP 전달을 **2경로** 제공:
**(1) FastRPC direct call** — IDL 생성 stub/skel, op마다 RPC 왕복.
**(2) Message Channel** — 공유메모리(rpcmem) + DSP 상주 poller 스레드.

### 공유 버퍼 구조 (`include/message.h`)
```
MessageHeader
 ├ MessageState state (8B union v[8]/d): v[0]=host ready, v[1]=DSP done
 ├ checksum, n_reqs
 └ req_offsets[n_reqs+1]   각 요청 바이트 오프셋 (배칭 지원)
RequestHeader  ─ state(반환코드), type(OP_COMPUTE / RPCMEM_MAP)
 └ OpComputeRequest ─ op(HtpOpsIndex), payload[]
      └ MatMulParams / RmsNormF32Params / FlashAttnParams
        (버퍼는 RpcmemBufAddr{fd, offset}로 지정)
```

### 수명주기
- 호스트 `alloc_shared_mem_buf`(`rpcmem_alloc` UNCACHED + `fastrpc_mmap`)
  → `create_htp_message_channel(fd, size)` (FastRPC)로 fd 전달.
- DSP `htp_ops_create_channel` → `HAP_mmap_get(fd)` 동일 물리메모리 매핑
  → poller 스레드 `msg_receiver_loop` 기동 (qurt, prio 64, 8KB stack).
- **전역 채널 1개**(`global_msg_chan`) — DSP 동시 1개만 허용.

### 1회 왕복 프로토콜 (2-flag 핸드셰이크)
- 호스트: `state.d=0` → 요청 작성 → **`v[0]=1`**(ready) →
  **`while(v[1]!=1) busy-poll`** → `RequestHeader.state`에서 반환코드.
- DSP `msg_receiver_loop` (1μs 폴링): `cache INVALIDATE` →
  **`memd_aq`**(acquire load) `state.d` → `v[0]==0||v[1]!=0`면 sleep →
  `n_reqs` 순회 `execute_op_simple()` → `cache FLUSH` → `barrier` →
  **`memd_rl`**(release store) `v[1]=1`(ack).
- 동기화: `v[0]`/`v[1]` 2-flag + `memd_aq`/`memd_rl` acquire-release +
  수동 캐시 INVALIDATE/FLUSH.
- ⚠️ 코드에 `// FIXME: memory order may not be working` 등 — 메모리 순서
  미검증 **실험 단계**.

### DSP 디스패치 (`op_executor.cc execute_op_simple`)
`req->op`(HtpOpsIndex) switch → `hvx_rms_norm_f32` /
`hmx_mat_mul_af32_pwf16_of32` / `hmx_mat_mul_af32_pwqk0_of32`(양자화) /
`simple_flash_attn`. `mmap_manager`가 fd→VA 캐싱.

### FastRPC vs Message Channel

| 항목 | FastRPC direct | Message Channel |
|---|---|---|
| 호출당 오버헤드 | FastRPC 마샬링 (~수십μs) | spin-poll, RPC 우회 → 낮음 |
| fd→VA 변환 | 매 호출 `HAP_mmap_get/put` | `mmap_manager` 캐싱 1회 |
| 배칭 | 불가 (1호출 1op) | `n_reqs`로 다중 요청 |
| DSP 자원 | 호출 시 PD 스케줄 | poller 스레드 상시 점유 |
| 성숙도 | 안정 | 실험적 (FIXME 다수) |

### llama.cpp와의 차이
ggml-hexagon은 `htp_iface.idl`에 `start/stop`만 두고 op는 DSP queue
(`dsp_queue_id`) 기반 비동기 큐(htp-drv)로 처리 — nntrainer의 단순
spin-poll message channel과 **다른 메커니즘**.

### 계획 반영 포인트
- Phase 1 포팅 시 message channel(`commu.c`/`message.h`/`mmap_mgr`)을
  포함할지 결정. 초기엔 **안정적인 FastRPC direct 경로만** 가져오고,
  message channel은 성능 최적화 단계(후속)로 분리 권장.
- message channel 채택 시 메모리 순서(`FIXME` 구간) 검증을 별도 작업으로.
- 배칭/저지연이 필요한 디코드 루프(CausalLM 토큰 생성)에서 message
  channel 이득이 크므로, 2단계 도입 가치 있음.

---

## 8. 다음 단계

1. §3 Option 확정 (권장: Option 1 베이스 + C 보강).
2. (가) hexkl_addon 경로 유지 여부 확정.
3. §7 — message channel 포함 여부 / 도입 시점 확정
   (권장: 1단계 FastRPC only → 2단계 message channel).
4. 확정 시 Phase 1 착수 — `htp_libs_integration`의 `htp_backend/` 선택 포팅.
