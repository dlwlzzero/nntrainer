# M3: 호스트 파이프라인 — W8_CX 양자화·lowering·가중치 배치·fake-quant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** W8_CX(int8 대칭 per-output-channel) 양자화 도구, qwen3 graph lowering(op-list + WEIGHTS 이미지), fake-quant fp32 기준선을 만들고 전부 x86에서 검증한다 (디바이스 불필요).

**Architecture:** ① cpu_backend에 대칭 per-channel int8 양자화 프리미티브 추가 → ② `DataType::W8_CX` + 저장 브랜치(QS4CX 전례 복제) → ③ `nntr_quantize`에 W8_CX 노출 → ④ safetensors 단계에서 quant→dequant만 적용하는 `nntr_fakequant` 도구 + CausalLM perplexity 평가 모드로 양자화 손실 게이트(<1%) → ⑤ 순수 C++ lowering 모듈(`graph_lowering.{h,cpp}`)이 M2 와이어 포맷 v2로 op-list와 WEIGHTS 배치 계획을 생성하고 가중치를 패킹한다. DSP·엔진 연결은 M4.

**Tech Stack:** x86 gcc/g++ (meson 기본 빌드: enable-app/test/ccapi 전부 기본 true), gtest(test/unittest 표), M2 와이어 포맷 `nntr_htp_common.h`(ABI v2), nlohmann json(CausalLM 기존 의존), `_Float16`(gcc 13/clang 지원)

**Spec:** `docs/superpowers/specs/hexagon-hvx/03-host-pipeline.md` (배경: `00-overview.md`, 와이어 포맷·op 시맨틱스: `02-hvx-kernels.md` + M2 결과물 `nntrainer/tensor/hexagon/htp/nntr_htp_common.h`)

## Global Constraints

- **작업 브랜치:** `hvx_impl`에서 분기한 **`hvx_m3`** 브랜치에서 전 Task를 수행한다 (`git checkout -b hvx_m3 hvx_impl`). `hvx_impl`에는 직접 커밋하지 않는다.
- **ponytail (사용자 계약, M2 계승):** 모든 구현 서브에이전트에 `ponytail:ponytail` (full) 적용 — 가장 단순한 동작하는 해법, 표준 라이브러리·기존 코드 우선, 투기적 추상화 금지. 리뷰 서브에이전트는 스펙 준수+품질 관점.
- **승인 게이트 (사용자 계약, 최우선):** 에이전트는 빌드/테스트를 **직접 실행하지 않는다**. 각 Task의 검증 단계는 "명령어 제시 → 정지 → 사용자가 실행한 출력 수신 → 판정"이다. 출력 수신 전 통과/실패 판정·완료 보고 금지. 사용자 승인 후에만 커밋 (커밋 메시지 초안을 먼저 보여주고 승인받는다).
- **커밋 형식:** `[태그] 제목` + 바디 + `Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>` + `Co-authored-by: Claude Fable 5 <noreply@anthropic.com>`
- **CI 규칙 (신규 파일마다):** `// SPDX-License-Identifier: Apache-2.0` + doxygen 블록(`@file @date @brief @see @author dlwlzzero <dlwlzzero@gmail.com> @bug`), repo `.clang-format` 준수, 영문 주석만.
- **plan/spec/handoff 문서는 커밋하지 않는다** (`docs/superpowers/`는 gitignore 대상, .gitignore 한 줄 워킹트리 변경은 커밋 제외 유지).
- **x86 빌드:** `meson setup build_x86 && ninja -C build_x86` (enable-app/enable-test/enable-ccapi 기본 true — 옵션 불필요). Hexagon SDK 환경 불필요 (M3 전체가 디바이스·시뮬레이터 없는 x86 작업).
- **정렬 규칙 (M2 계승):** lowering이 만드는 모든 텐서 오프셋·행 stride는 128B 정렬. 검증은 `nntr_htp_oplist_validate`(init 시점)로, forward 경로 검사 없음.
- **양자화 산식 (전 Task 공통):** 행(출력 채널)별 `scale = absmax/127`, `q = clamp(round(x·127/absmax), −127, 127)`, dequant `x̂ = q·scale`. fake-quant와 W8_CX .bin이 **같은 함수**(`quant_w8cx_f32`)를 쓰므로 ①vs② 비교의 전제(동일 양자화 값)가 자동 성립한다.
- **정확도 보고 형식 (사용자 메모리):** 정확도 수치(PPL 등)는 항상 수행 시간(wall time)과 짝지어 표로 보고한다.
- **qwen3-0.6b 모델은 이 머신에 아직 없다 — HF에서 다운로드 필요** (Task 3 검증 첫 단계). 리포: `Qwen/Qwen3-0.6B` (bf16 단일 safetensors, ~1.5GB). 받은 디렉토리를 아래에서 `$QWEN3`로 표기. 다운로드도 승인 게이트 대상(사용자 실행).

---

## 공통 설계 (모든 Task의 단일 기준)

### W8_CX 파일 포맷

2D 가중치 `[N][K]`(N=출력 채널) 하나당: **int8 blob `N*K` 바이트(N-major, 행 stride=K) 바로 뒤에 fp32 scale `N*4` 바이트.** QS4CX와 동일 구조(단지 4bit→8bit). 저장 시 nntrainer 텐서(K×N)를 `transpose("0:2:1")`로 N×K로 만들어 양자화한다 — QS4CX 브랜치(`layer_devel.h:423-445`)와 동일.

- 스펙의 "PER_CHANNEL_AFFINE quantizer 재사용"은 **불가능** — `PerChannelAffineQuantizer`는 NYI 스텁(`quantizer.cpp:184-209`, 입력을 그대로 반환). 실제 재사용 가능한 전례는 QS4CX free-function 경로이며 이 계획은 그것을 따른다.
- 기존 `quant_qs8cx_f32`도 재사용 불가 — min/max 범위 기반(비대칭 분포에서 클리핑 발생)이라 스펙의 대칭 absmax 방식과 다르다. dequant(`int8·scale`)만 동일하므로 dequant는 기존 internal을 위임 재사용한다.

### qwen3-0.6b 치수 (lowering 상수 검증용)

hidden 1024, n_layers 28, n_heads 16, n_kv_heads 8, head_dim 128, ffn 3072, vocab 151936, rope_theta 1e6, rms_eps 1e-6. matmul K값 전부 128 배수: q/k/v/gate/up K=1024, o K=2048, down K=3072, logits K=1024 → int8 행이 자연 128B 정렬.

### lowering API (`nntrainer/tensor/hexagon/host/graph_lowering.h`)

```cpp
namespace nntrainer::hexagon {

struct HexModelConfig { // nntr_htp_oplist_header 필드 + rope/eps
  uint32_t n_layers, n_heads, n_kv_heads, head_dim;
  uint32_t hidden, ffn, vocab, max_seq, max_chunk;
  float rms_eps, rope_theta;
};

struct HexLayerWeights { // 전부 non-owning 포인터, int8은 N-major [N][K]
  const int8_t *wq, *wk, *wv, *wo, *w_gate, *w_up, *w_down;
  const float *wq_s, *wk_s, *wv_s, *wo_s, *w_gate_s, *w_up_s, *w_down_s;
  const float *attn_norm, *ffn_norm, *q_norm, *k_norm; // fp32 gamma
};

struct HexModelWeights {
  const int8_t *embed;   // [vocab][hidden], tied → lm_head 겸용
  const float *embed_s;  // [vocab]
  const float *final_norm;
  std::vector<HexLayerWeights> layers; // size == n_layers
};

struct HexWeightOffsets { // WEIGHTS 버퍼 내 바이트 오프셋 (전부 128B 정렬)
  uint32_t embed, embed_scale, rope_table, final_norm;
  struct PerLayer {
    uint32_t wq, wq_s, wk, wk_s, wv, wv_s, wo, wo_s;
    uint32_t gate, gate_s, up, up_s, down, down_s;
    uint32_t attn_norm, ffn_norm, q_norm, k_norm; // fp16으로 변환 저장
  };
  std::vector<PerLayer> layers;
};

struct HexLoweredGraph {
  std::vector<uint8_t> oplist;      // header(64B) + n_ops*64B
  HexWeightOffsets woff;
  uint64_t weights_size, kv_size, act_size;
};

HexLoweredGraph lower_qwen3(const HexModelConfig &cfg);          // Task 7
void pack_weights(const HexLoweredGraph &g, const HexModelConfig &cfg,
                  const HexModelWeights &w, uint8_t *dst);       // Task 8
} // namespace nntrainer::hexagon
```

### WEIGHTS 배치 순서 (pack과 test의 단일 기준)

`align128(x) = (x + 127) & ~127u`. 커서 0에서 시작해 순서대로: `embed`(vocab·hidden int8) → `embed_scale`(vocab·4) → `rope_table`(max_seq·128·2, fp16 `[max_seq][cos64‖sin64]`) → `final_norm`(hidden·2 fp16) → 레이어 0..L−1 각각 `wq, wq_s, wk, wk_s, wv, wv_s, wo, wo_s, gate, gate_s, up, up_s, down, down_s, attn_norm, ffn_norm, q_norm, k_norm` — 각 텐서 시작마다 align128. norm gamma는 fp32→fp16 변환(`_Float16` 캐스팅), q_norm/k_norm은 `[head_dim]` 공유 벡터. RoPE 행 p: `cos[i]=cosf(p·θ^(−2i/128))`, `sin[i]` 동일 각도 (i=0..63, θ=rope_theta) — M2 `ref_rope_table_fill`과 동일 산식.

### ACT 배치 (레이어 간 재사용, per-slot 128B 정렬)

| slot | 크기(바이트) | 용도 |
|---|---|---|
| `x` | max_chunk·hidden·2 | residual stream (EMBED out, ADD out) |
| `t` | max_chunk·hidden·2 | RMSNORM out (matmul 입력) |
| `q` | max_chunk·n_heads·128·2 | q proj out (per-head norm·rope 제자리) |
| `kb` | max_chunk·n_kv_heads·128·2 | k proj out |
| `vb` | max_chunk·n_kv_heads·128·2 | v proj out |
| `ao` | max_chunk·n_heads·128·2 | ATTN out |
| `h2` | max_chunk·hidden·2 | o proj out / down proj out (교대 재사용) |
| `g` | max_chunk·ffn·2 | gate out, SILU_MUL out (제자리) |
| `u` | max_chunk·ffn·2 | up out |

`act_size` = 마지막 slot 끝. elementwise op(ADD/SILU_MUL/RMSNORM per-head)의 out==in0 별칭은 안전(M2 커널이 strip 단위 read-then-write).

### op 시퀀스 (총 `1 + 16·n_layers + 2` op, qwen3-0.6b=451)

| # | kind | in0 | in1 | in2 | out | m,k,n / flags·param |
|---|---|---|---|---|---|---|
| 0 | EMBED | TOKENS@0 | W.embed | W.embed_scale | x | m=0, k=hidden |
| L.1 | RMSNORM | x | W.attn_norm | – | t | n=hidden, param0=eps비트 |
| L.2 | MATMUL_W8A8 | t | W.wq | W.wq_s | q | k=hidden, n=n_heads·128 |
| L.3 | MATMUL_W8A8 | t | W.wk | W.wk_s | kb | n=n_kv_heads·128 |
| L.4 | MATMUL_W8A8 | t | W.wv | W.wv_s | vb | n=n_kv_heads·128 |
| L.5 | RMSNORM | q | W.q_norm | – | q | PER_HEAD, n=n_heads·128 |
| L.6 | RMSNORM | kb | W.k_norm | – | kb | PER_HEAD, n=n_kv_heads·128 |
| L.7 | ROPE | q | kb | W.rope_table | =in0 | |
| L.8 | ATTN | q | kb | vb | ao | layer=L, param0=1/√128비트 |
| L.9 | MATMUL_W8A8 | ao | W.wo | W.wo_s | h2 | k=n_heads·128, n=hidden |
| L.10 | ADD | x | h2 | – | x | n=hidden |
| L.11 | RMSNORM | x | W.ffn_norm | – | t | n=hidden |
| L.12 | MATMUL_W8A8 | t | W.gate | W.gate_s | g | k=hidden, n=ffn |
| L.13 | MATMUL_W8A8 | t | W.up | W.up_s | u | n=ffn |
| L.14 | SILU_MUL | g | u | – | g | n=ffn |
| L.15 | MATMUL_W8A8 | g | W.down | W.down_s | h2 | k=ffn, n=hidden |
| L.16 | ADD | x | h2 | – | x | n=hidden |
| 끝-2 | RMSNORM | x | W.final_norm | – | t | n=hidden |
| 끝-1 | MATMUL_LOGITS | t | W.embed (tied) | W.embed_scale | LOGITS@0 | m=1, k=hidden, n=vocab |

`kv_size = 2·n_layers·n_kv_heads·max_seq·head_dim·2`. 미지정 필드(m 등)는 0 (m=0 → 실행 시 n_tokens).

### 검증용 tiny 설정 (Task 7·8 테스트)

n_layers=2, hidden=256, n_heads=4, n_kv_heads=2, head_dim=128, ffn=512, vocab=512, max_seq=64, max_chunk=8, eps=1e-6, theta=1e6 → n_ops=35. (validate 제약 충족: hidden%64, ffn%64, matmul k%128.)

---

### Task 1: W8_CX 대칭 per-channel 양자화 프리미티브

**Files:**
- Modify: `nntrainer/tensor/cpu_backend/fallback/fallback_internal.h`, `fallback_internal.cpp`
- Modify: `nntrainer/tensor/cpu_backend/fallback/fallback.h`, `fallback.cpp`
- Modify: `nntrainer/tensor/cpu_backend/x86/x86_compute_backend.h`, `.cpp` / `arm/arm_compute_backend.h`, `.cpp` (qs8cx와 동일 위임 패턴 — `grep -n qs8cx`로 4파일의 선언·위임 지점을 찾아 그대로 미러)
- Modify: `test/unittest/unittest_nntrainer_quantizer.cpp`

**Interfaces:**
- Produces (모든 백엔드 헤더 동일 시그니처, `nntrainer::` 네임스페이스):

```cpp
/* 대칭 per-output-channel int8: scale[n_idx] = absmax/127 (dequant 곱셈자) */
void quant_w8cx_f32(size_t n, size_t k, void *rhs_native_mtx_f32,
                    void *rhs_native_mtx_w8cx, void *rhs_scales_f32);
void dequant_w8cx_f32(size_t n, size_t k, void *rhs_native_mtx_w8cx,
                      void *rhs_scales_f32, void *rhs_native_mtx_f32);
```

- Task 2(저장 브랜치)·4(fake-quant)·7/8 테스트가 이 두 함수를 소비한다.

- [ ] **Step 1: internal 구현** — `fallback_internal.{h,cpp}`에 추가 (qs8cx 함수들 옆):

```cpp
void __fallback_quant_nxk_w8cx_f32(size_t n, size_t k, const float *rhs_f32,
                                   int8_t *rhs_w8cx, float *rhs_scales_f32) {
  for (size_t n_idx = 0; n_idx < n; ++n_idx) {
    const float *src_ptr = rhs_f32 + n_idx * k;
    int8_t *dst_ptr = rhs_w8cx + n_idx * k;

    float amax = 0.0f;
    for (size_t k_idx = 0; k_idx < k; ++k_idx)
      amax = std::max(amax, std::fabs(src_ptr[k_idx]));

    const float scale = amax / 127.0f;
    const float inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;

    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      int32_t v = (int32_t)std::lround(src_ptr[k_idx] * inv_scale);
      v = std::max(v, (int32_t)-127);
      v = std::min(v, (int32_t)127);
      dst_ptr[k_idx] = (int8_t)v;
    }
    rhs_scales_f32[n_idx] = scale;
  }
}
```

dequant internal은 신설하지 않는다 — 산식이 동일하므로 public `dequant_w8cx_f32`가 `__fallback_dequant_nxk_qs8cx_f32`를 그대로 호출 (위임 지점에 주석으로 명기).

- [ ] **Step 2: public 위임 4벌** — `fallback.{h,cpp}` + x86/arm backend에 qs8cx와 같은 형태로 선언·위임 추가. doxygen 주석에 "symmetric absmax/127 per-output-channel; scale is the dequant multiplier" 명기.

- [ ] **Step 3: gtest 작성** — `unittest_nntrainer_quantizer.cpp`에 추가:

```cpp
TEST(nntrainer_Quantizer, quant_w8cx_roundtrip_p) {
  const size_t N = 4, K = 64;
  std::vector<float> w(N * K);
  for (size_t i = 0; i < N * K; ++i)
    w[i] = 0.01f * (float)((int)(i * 2654435761u % 200) - 100); // 결정적, [-1,1)
  std::vector<int8_t> q(N * K);
  std::vector<float> s(N), d(N * K);

  nntrainer::quant_w8cx_f32(N, K, w.data(), q.data(), s.data());
  nntrainer::dequant_w8cx_f32(N, K, q.data(), s.data(), d.data());

  for (size_t n = 0; n < N; ++n) {
    float amax = 0.0f;
    for (size_t k = 0; k < K; ++k)
      amax = std::max(amax, std::fabs(w[n * K + k]));
    EXPECT_FLOAT_EQ(s[n], amax / 127.0f);
    for (size_t k = 0; k < K; ++k) {
      EXPECT_LE(std::abs((int)q[n * K + k]), 127);
      EXPECT_NEAR(w[n * K + k], d[n * K + k], s[n] * 0.5f + 1e-8f);
    }
  }
}

TEST(nntrainer_Quantizer, quant_w8cx_zero_row_p) {
  const size_t N = 2, K = 32;
  std::vector<float> w(N * K, 0.0f);
  for (size_t k = 0; k < K; ++k)
    w[K + k] = 0.5f; // 두 번째 행만 비제로
  std::vector<int8_t> q(N * K, 42);
  std::vector<float> s(N, -1.0f);
  nntrainer::quant_w8cx_f32(N, K, w.data(), q.data(), s.data());
  for (size_t k = 0; k < K; ++k)
    EXPECT_EQ(q[k], 0); // 0-행: q 전부 0
  EXPECT_FLOAT_EQ(s[0], 0.0f);
  EXPECT_EQ(q[K], 127);
}
```

- [ ] **Step 4: 검증 (사용자 실행) — 명령어 제시 후 정지**

```bash
meson setup build_x86 2>/dev/null || true
ninja -C build_x86 test/unittest/unittest_nntrainer_quantizer
./build_x86/test/unittest/unittest_nntrainer_quantizer --gtest_filter='*w8cx*'
```

Expected: 2개 테스트 PASS, exit 0. (ninja 타깃명이 다르면 `ninja -C build_x86` 전체 빌드 후 실행.)

- [ ] **Step 5: 승인 후 커밋** — `[tensor] Add the symmetric per-channel W8_CX int8 quantization primitives`

---

### Task 2: DataType::W8_CX + 저장 브랜치

**Files:**
- Modify: `api/ccapi/include/tensor_dim.h:55-71` (enum), `nntrainer/utils/base_properties.h:663-671` (EnumList/EnumStr), `nntrainer/models/model_common_properties.h:190-224` (모델 텐서 타입), `nntrainer/tensor/tensor_dim.cpp:170-184, 415-425` (크기·문자열), `nntrainer/models/neuralnet.cpp:971-977` (qparam 트레일러 제외 목록), `nntrainer/layers/layer_devel.h:366-453` (저장 브랜치)
- Modify: `test/unittest/unittest_nntrainer_save_with_dtype.cpp`

**Interfaces:**
- Consumes: Task 1 `quant_w8cx_f32`.
- Produces: `ml::train::TensorDim::DataType::W8_CX` (문자열 `"W8_CX"`, 크기 1바이트), 모델 텐서 타입 enum `W8CXA32`/`W8CXA16` (문자열 `"W8_CX-FP32"`/`"W8_CX-FP16"`), 그리고 `NeuralNetwork::save(..., layer_dtype_map={name→W8_CX})` 시 공통 설계의 W8_CX 파일 포맷으로 기록되는 저장 경로. Task 3(도구)·4가 이 dtype을 소비한다.

- [ ] **Step 1: enum·문자열 플러밍** — 위 6개 파일에 W8_CX 추가. 주의: `base_properties.h`의 EnumList와 EnumStr은 인덱스 정렬 필수(둘 다 끝에 추가). `neuralnet.cpp:971-977`의 조건에 `!= W8_CX` 추가(QS4CX처럼 별도 qparam 트레일러 없음 — scale이 blob에 내장). `tensor.cpp` 생성자 디스패치는 **건드리지 않는다** (M3는 저장 전용, W8_CX Tensor 객체를 만들지 않음 — 로드는 M4).

- [ ] **Step 2: 저장 브랜치** — `layer_devel.h`의 QS4CX 브랜치(`:423-445`) 바로 뒤에 추가:

```cpp
} else if (dtype == TensorDim::DataType::W8_CX) {
  NNTR_THROW_IF(weight.getDataType() != TensorDim::DataType::FP32,
                std::runtime_error)
    << "Save with quantization only supports for FP32 weight.";
  TensorDim dim = weight.getDim();
  size_t K = dim.height();
  size_t N = dim.width();
  if (K == 1) { /* bias-like 1D tensors stay FP32 */
    weight.save(file);
  } else {
    Tensor weight_t = weight.transpose("0:2:1");
    size_t q_size = N * K;
    size_t scale_size = N * sizeof(float);
    std::vector<uint8_t> rhs_q(q_size + scale_size);
    int8_t *data = reinterpret_cast<int8_t *>(rhs_q.data());
    float *scale = reinterpret_cast<float *>(rhs_q.data() + q_size);
    nntrainer::quant_w8cx_f32(N, K, weight_t.getData(), data, scale);
    file.write(reinterpret_cast<const char *>(rhs_q.data()),
               q_size + scale_size);
  }
}
```

- [ ] **Step 3: gtest 작성** — `unittest_nntrainer_save_with_dtype.cpp`에 새 그룹. 기존 헬퍼 `createInitializedNN(input_width, units)` 사용, 파일을 다시 읽어 Task 1 함수 결과와 바이트 비교:

```cpp
TEST(SaveWithDtypeW8CX, save_w8cx_bytes_match_p) {
  auto nn = createInitializedNN(64, 32); // weight (1,1,64,32): K=64, N=32
  std::string file_path = "test_w8cx_64_32.bin";
  EXPECT_NO_THROW(
    nn->save(file_path, ModelFormat::MODEL_FORMAT_BIN, DataType::W8_CX));

  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(file.is_open());
  size_t fsize = file.tellg();
  // fc weight: N*K int8 + N*4 scale, 이어서 bias(units=32) fp32 그대로
  const size_t N = 32, K = 64;
  ASSERT_GE(fsize, N * K + N * 4);
  file.seekg(0);
  std::vector<uint8_t> blob(N * K + N * 4);
  file.read(reinterpret_cast<char *>(blob.data()), blob.size());
  file.close();

  // scale 영역이 전부 유한 fp32이고 음수가 아님을 확인
  const float *scales = reinterpret_cast<const float *>(blob.data() + N * K);
  for (size_t n = 0; n < N; ++n) {
    EXPECT_TRUE(std::isfinite(scales[n]));
    EXPECT_GE(scales[n], 0.0f);
  }
  // int8 값 범위 확인
  const int8_t *q = reinterpret_cast<const int8_t *>(blob.data());
  for (size_t i = 0; i < N * K; ++i)
    EXPECT_LE(std::abs((int)q[i]), 127);
  remove(file_path.c_str());
}

TEST(SaveWithDtypeW8CX, save_w8cx_non_fp32_source_n) {
  // 기존 SaveWithDtype 그룹의 non-FP32 소스 throw 테스트와 동일 구조로
  // W8_CX 지정 시 std::runtime_error를 기대
}
```

(정확한 save 호출 시그니처·비FP32 케이스 구성은 같은 파일의 `SaveWithDtypeQ4` 그룹을 그대로 따른다. 가능하면 dequant 후 원본 가중치와 `scale/2` 이내 일치까지 추가 — 원본 가중치는 `nn` 저장 전 `run_context` 접근이 없으므로 FP32로 한 번 더 저장해 파일끼리 비교하는 방식 허용.)

- [ ] **Step 4: 검증 (사용자 실행) — 정지**

```bash
ninja -C build_x86
./build_x86/test/unittest/unittest_nntrainer_save_with_dtype --gtest_filter='SaveWithDtypeW8CX*'
./build_x86/test/unittest/unittest_nntrainer_save_with_dtype   # 기존 36개 회귀
./build_x86/test/unittest/unittest_nntrainer_quantizer --gtest_filter='*w8cx*'
```

Expected: 신규·기존 전부 PASS, exit 0.

- [ ] **Step 5: 승인 후 커밋** — `[tensor] Add the W8_CX data type with its save-with-quantization branch`

---

### Task 3: nntr_quantize W8_CX 지원 + 실모델 양자화

**Files:**
- Modify: `Applications/CausalLM/quantize.cpp` — `dtype_str_map`(`:103-107`)에 `{"W8_CX", DataType::W8_CX}`, help 문자열(`:41, :154, :426`)에 W8_CX 추가, `generateOutputBinName` suffix 목록(`:204-206`)에 `"_w8_cx"` 추가

**Interfaces:**
- Consumes: Task 2의 W8_CX 저장 경로.
- Produces: `nntr_quantize $QWEN3 --fc_dtype W8_CX --embd_dtype W8_CX -o <dir>` → W8_CX .bin + `nntr_config.json`(`fc_layer_dtype: "W8_CX"`, `model_tensor_type: "W8_CX-FP32"`). M4의 디바이스 e2e가 이 .bin을 소비한다.

- [ ] **Step 1: 구현** — 위 4개 지점 수정. `--output_format safetensors`와 W8_CX 조합은 명시적 에러로 거부(`safetensors output is not supported for W8_CX yet`) — safetensors 저장 경로(`safetensors_util.cpp isQuantized`)는 M3 범위 외.

- [ ] **Step 2: 모델 다운로드 (사용자 실행, 최초 1회) — 정지**

```bash
pip install -U "huggingface_hub[cli]" 2>/dev/null || true
hf download Qwen/Qwen3-0.6B --local-dir /local/mnt/workspace/models/qwen3-0.6b
export QWEN3=/local/mnt/workspace/models/qwen3-0.6b
ls -l $QWEN3   # model.safetensors(~1.5GB, bf16), config.json, tokenizer.json 확인
```

(`hf` 명령이 없으면 `huggingface-cli download` 동일 인자. 저장 위치는 사용자 재량 — 이후 명령은 `$QWEN3` 기준.)

- [ ] **Step 3: 검증 (사용자 실행) — 정지**

```bash
ninja -C build_x86
./build_x86/Applications/CausalLM/nntr_quantize $QWEN3 \
    --fc_dtype W8_CX --embd_dtype W8_CX -o /tmp/qwen3_w8cx
ls -l /tmp/qwen3_w8cx/ && cat /tmp/qwen3_w8cx/nntr_config.json
```

Expected: 에러 없이 완료. .bin 크기 ≈ **0.6~0.7GB** (전 파라미터 int8 + 채널당 fp32 scale + norm fp32; fp32 원본 ~2.4GB의 약 ¼). `nntr_config.json`에 `"fc_layer_dtype": "W8_CX"`, `"embedding_dtype": "W8_CX"`, `"model_tensor_type": "W8_CX-FP32"`. (실행 파일 경로가 다르면 `find build_x86 -name nntr_quantize`. HF 리포에 `nntr_config.json`이 없으므로 로더가 요구하면 기존 모델의 것을 참조해 qwen3용으로 작성 — `--config` 옵션 활용.)

- [ ] **Step 4: 승인 후 커밋** — `[causallm] Support the W8_CX weight type in nntr_quantize`

---

### Task 4: nntr_fakequant 도구 (safetensors quant→dequant)

**Files:**
- Create: `Applications/CausalLM/fake_quantize.cpp`
- Modify: `Applications/CausalLM/meson.build` (`e_quantize` 블록 옆에 `nntr_fakequant` executable 추가, 동일 의존성)

**Interfaces:**
- Consumes: Task 1 `quant_w8cx_f32`/`dequant_w8cx_f32`.
- Produces: `nntr_fakequant <in.safetensors> <out.safetensors>` — 2D 텐서(F32/BF16)에만 W8_CX 왕복을 적용하고 전 텐서를 F32로 기록. Task 6의 기준선 파이프라인이 소비.

설계 근거: 모델 로드 후 가중치를 제자리 수정할 공개 API가 없고(`NeuralNetwork`에 getWeights 부재), save 체인에 플래그를 관통시키면 ccapi 공개 시그니처가 바뀐다. safetensors는 자기술적(이름·shape·dtype)이므로 파일→파일 변환이 코어 무변경 최소 구현이다. HF 선형 가중치는 `[out,in]` row-major = N×K라 `quant_w8cx_f32`를 그대로 적용 가능하고, `nntr_quantize`의 W8_CX 경로(transpose 후 N×K)와 **같은 값·같은 함수**로 양자화되므로 3-way ①vs②의 전제가 성립한다.

- [ ] **Step 1: 구현** — 핵심 구조 (nlohmann `ordered_json` 사용 — data_offsets 순서 보존):

```cpp
// safetensors: [u64 LE header_len][header JSON][data]
// 1) 헤더 파싱: 각 텐서 {dtype, shape, data_offsets}
// 2) 선택 규칙: shape.size()==2 && dtype in {F32, BF16} → W8_CX 왕복 대상
//    (qwen3-0.6b: 7*28 proj + embed = 197개. 1D norm 등은 비대상)
// 3) BF16→F32: uint32_t f = ((uint32_t)h) << 16; memcpy to float
// 4) 대상: quant_w8cx_f32(N, K, f32, q, s) 후 dequant_w8cx_f32(N, K, q, s, f32)
// 5) 출력: 모든 텐서를 F32로 기록 (dtype/오프셋 재계산, __metadata__ 유지)
// 6) 텐서별 리포트 출력: name shape fq|copy, 마지막에 "fq_tensors=<count>"
```

에러 처리: 다중 샤드(index.json) 입력은 명시적 에러(qwen3-0.6b는 단일 파일), F32/BF16 외 dtype이 2D면 에러(조용한 미적용 방지).

- [ ] **Step 2: 검증 (사용자 실행) — 정지**

```bash
ninja -C build_x86
./build_x86/Applications/CausalLM/nntr_fakequant \
    $QWEN3/model.safetensors /tmp/qwen3_fq/model.safetensors
```

Expected: `fq_tensors=197` (qwen3-0.6b: 28레이어×7 proj + embedding), 출력 파일 크기 ≈ 전 텐서 F32 합(~2.4GB), 에러 없음.

- [ ] **Step 3: 승인 후 커밋** — `[causallm] Add the nntr_fakequant tool producing the W8_CX QDQ baseline`

---

### Task 5: perplexity 평가 모드

**Files:**
- Modify: `Applications/CausalLM/models/transformer.h` (기본 throw 가상 함수 — 기존 멀티모달 패턴과 동일)
- Modify: `Applications/CausalLM/models/causal_lm.h`, `causal_lm.cpp`
- Modify: `Applications/CausalLM/main.cpp`

**Interfaces:**
- Produces: `virtual double Transformer::evaluatePerplexity(const std::string &text)` (기본: `throw std::runtime_error("evaluatePerplexity not supported by this model")`), CausalLM 오버라이드, 그리고 `nntrainer_causallm <model_path> --eval <textfile>` CLI 모드(`PPL <값>` 출력). Task 6이 소비.

- [ ] **Step 1: CausalLM::evaluatePerplexity 구현** — teacher-forcing: 위치 p의 입력은 항상 **참조 토큰**(생성 토큰 아님), 위치 p 로짓으로 토큰 p+1의 NLL 누적. `run()`의 디코드 루프(`causal_lm.cpp:620-646`)와 같은 호출 규약(`allocateAndBindKVCache()` 매 스텝, `build_inference_inputs` 패턴, `incremental_inference` 인자, output `delete[]`)을 그대로 따른다 — `build_inference_inputs` 람다는 private 멤버 함수로 추출해 run()과 공유:

```cpp
double CausalLM::evaluatePerplexity(const std::string &text) {
  if (!is_initialized)
    throw std::runtime_error("CausalLM model is not initialized.");
  NNTR_THROW_IF(BATCH_SIZE != 1, std::invalid_argument)
    << "evaluatePerplexity supports batch_size=1 only";

  auto ids = tokenizer->Encode(text);
  unsigned int n = static_cast<unsigned int>(ids.size());
  NNTR_THROW_IF(n < 2, std::invalid_argument) << "eval text too short";
  if (n > MAX_SEQ_LEN) {
    std::cerr << "[eval] truncating " << (n - MAX_SEQ_LEN) << " tokens\n";
    n = MAX_SEQ_LEN;
  }

  allocateAndBindKVCache();
  setKVCachePosition(0);

  float *input_sample = (float *)malloc(sizeof(float) * MAX_SEQ_LEN);
  std::vector<float *> label;
  std::vector<float *> input = buildInferenceInputs(input_sample); // run()과 공유

  double nll = 0.0;
  for (unsigned int p = 0; p + 1 < n; ++p) {
    input_sample[0] = static_cast<float>(ids[p]);   // 참조 토큰 강제
    allocateAndBindKVCache();
    auto out = model->incremental_inference(BATCH_SIZE, input, label,
                                            /*input_len=*/1, p, p + 1, false);
    const float *logits = out[0];
    float maxv = *std::max_element(logits, logits + NUM_VOCAB);
    double sum = 0.0;
    for (unsigned int v = 0; v < NUM_VOCAB; ++v)
      sum += std::exp((double)logits[v] - maxv);
    nll += -((double)logits[ids[p + 1]] - (double)maxv - std::log(sum));
    for (auto o : out)
      delete[] o;
  }
  free(input_sample);
  return std::exp(nll / (double)(n - 1));
}
```

(구현 시 `incremental_inference`의 4번째 인자(input_len)와 from/to 규약을 디코드 루프 실사용과 대조해 조정 — 판단 기준은 "매 스텝 KV 위치 p에 1토큰 기록, 로짓은 위치 p의 것"이며, 검증 Step의 fp32 PPL 수치가 상식 범위인지로 확인한다.)

- [ ] **Step 2: main.cpp 모드 추가** — `model->repack_weight()` 직후(현행 `argc>=3 → argv[2]=prompt` 분기보다 먼저):

```cpp
if (argc >= 4 && std::string(argv[2]) == "--eval") {
  std::ifstream ef(argv[3]);
  if (!ef.is_open()) {
    std::cerr << "cannot open eval file: " << argv[3] << std::endl;
    return EXIT_FAILURE;
  }
  std::stringstream ss;
  ss << ef.rdbuf();
  auto t0 = std::chrono::steady_clock::now();
  double ppl = model->evaluatePerplexity(ss.str());
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - t0).count();
  std::cout << "PPL " << ppl << " wall_ms " << ms << std::endl;
  return EXIT_SUCCESS;
}
```

- [ ] **Step 3: 검증 (사용자 실행) — 정지** — 먼저 fp32 기준 .bin 준비 후 평가. 평가 텍스트는 200~500토큰 분량 영어 산문 파일(예: 위키 문단)을 `/tmp/eval.txt`로 준비:

```bash
ninja -C build_x86
./build_x86/Applications/CausalLM/nntr_quantize $QWEN3 \
    --fc_dtype FP32 --embd_dtype FP32 -o /tmp/qwen3_fp32
./build_x86/Applications/CausalLM/nntrainer_causallm /tmp/qwen3_fp32 --eval /tmp/eval.txt
```

Expected: `PPL <5~50 범위의 유한값> wall_ms <값>`, exit 0. PPL이 수백~수천이면 위치/로짓 정렬 버그(off-by-one)로 판정하고 Step 1의 from/to 규약을 재검토한다.

- [ ] **Step 4: 승인 후 커밋** — `[causallm] Add the teacher-forced perplexity eval mode`

---

### Task 6: fake-quant 기준선 측정 (M3 정확도 게이트)

**Files:** 신규 코드 없음 (Task 3·4·5 산출물 사용). 결과는 Task 9에서 문서에 기록.

**Interfaces:**
- Consumes: `nntr_fakequant`(Task 4), FP32 변환·평가 경로(Task 5).
- Produces: ① 원본 fp32 vs ② fake-quant fp32 PPL 표 — [04](../specs/hexagon-hvx/04-e2e-verification.md) 3-way 검증의 ② 확정.

- [ ] **Step 1: 파이프라인 실행 (사용자 실행) — 정지**

```bash
# fq safetensors를 모델 디렉토리 구조로 구성 (config류는 원본 링크)
mkdir -p /tmp/qwen3_fq && cp $QWEN3/*.json /tmp/qwen3_fq/
./build_x86/Applications/CausalLM/nntr_fakequant \
    $QWEN3/model.safetensors /tmp/qwen3_fq/model.safetensors
./build_x86/Applications/CausalLM/nntr_quantize /tmp/qwen3_fq \
    --fc_dtype FP32 --embd_dtype FP32 -o /tmp/qwen3_fq_bin
# ② fake-quant 평가 (①은 Task 5에서 측정 완료)
./build_x86/Applications/CausalLM/nntrainer_causallm /tmp/qwen3_fq_bin --eval /tmp/eval.txt
```

- [ ] **Step 2: 게이트 판정** — 표로 보고 (정확도+시간 병기 규칙):

| 실행 | PPL | wall(ms) |
|---|---|---|
| ① 원본 fp32 | | |
| ② fake-quant fp32 | | |

**게이트: `(PPL② − PPL①)/PPL① < 1%`.** 초과 시 이 마일스톤을 멈추고 양자화 설계 재검토(스펙 명시) — 후보: outlier 채널 확인, embedding만 fp16 유지 실험. 통과 수치는 Task 9 문서화에 인용.

- [ ] **Step 3:** 커밋 없음 (측정 Task). 결과 수치를 진행 원장에 기록.

---

### Task 7: graph lowering — op-list 생성

**Files:**
- Create: `nntrainer/tensor/hexagon/host/graph_lowering.h`, `graph_lowering.cpp` (공통 설계 API — `lower_qwen3`까지, `pack_weights`는 선언만)
- Modify: `nntrainer/tensor/hexagon/meson.build` (host 소스·헤더 목록에 추가 — android 라이브러리용, x86 테스트는 직접 컴파일)
- Create: `test/hexagon/test_lowering.cpp`

**Interfaces:**
- Consumes: `nntr_htp_common.h`의 v2 구조체·`nntr_htp_oplist_validate` (SDK 의존성 없음 — 순수 C/C++).
- Produces: `HexLoweredGraph lower_qwen3(const HexModelConfig &cfg)` — 공통 설계의 op 시퀀스 표·WEIGHTS 배치 순서·ACT 배치대로 oplist 바이트, `HexWeightOffsets`, `weights_size/kv_size/act_size`를 채운다. Task 8과 M4(HexagonRunner::init 인자)가 소비.

- [ ] **Step 1: 테스트 작성** — `test_lowering.cpp` (gtest 아님, M2 `test_oplist_header.c` 패턴의 self-contained main; PASS 시 `LOWER_TEST PASS` 출력 + exit 0):
  - tiny 설정(공통 설계)으로 `lower_qwen3` 호출.
  - **validate:** `buf_size = {weights_size, kv_size, act_size, max_chunk*4, vocab*4}`로 `nntr_htp_oplist_validate(oplist.data(), oplist.size(), buf_size) == 0`.
  - **op 수·시퀀스:** `n_ops == 1 + 16*n_layers + 2`; kind 시퀀스가 표와 일치(EMBED, 레이어당 [RMSNORM, MM, MM, MM, RMSNORM, RMSNORM, ROPE, ATTN, MM, ADD, RMSNORM, MM, MM, SILU_MUL, MM, ADD], RMSNORM, MATMUL_LOGITS).
  - **tied 공유:** 마지막 op(MATMUL_LOGITS)의 `in1.offset == ops[0](EMBED).in1.offset`, `in2` 동일. MATMUL_LOGITS `m==1`, out buf==LOGITS.
  - **헤더 필드:** magic/version/n_layers/…/max_chunk 전부 cfg와 일치.
  - **레이어 매개변수:** 각 MATMUL의 (k,n)이 표와 일치, ATTN의 layer 필드 = 레이어 인덱스, RMSNORM param0 == eps의 fp32 비트, per-head 플래그 위치 정확.
  - **ACT 겹침 금지:** x/t/q/kb/vb/ao/h2/g/u 슬롯 구간이 서로 소(disjoint)이고 `act_size` 안.
  - **실치수 스모크:** qwen3-0.6b 치수로 `n_ops==451`, `kv_size == 2*28*8*max_seq*128*2`, validate 통과, `weights_size`가 655~700MB 구간(embed 155.6MB + proj int8 합 + scale + norm fp16 + rope 테이블).

- [ ] **Step 2: 구현** — `lower_qwen3`: WEIGHTS 커서(배치 순서 절), ACT 커서(슬롯 표)로 오프셋 계산 → `nntr_htp_oplist_header` 채움 → 레이어 루프에서 표의 16 op을 `nntr_htp_op_desc`로 방출. 모든 곱은 `uint64_t`로 계산(오버플로 방지 — embed는 1.5억 원소). `param0`은 `memcpy`로 fp32 비트패턴 기록.

- [ ] **Step 3: 검증 (사용자 실행) — 정지**

```bash
g++ -std=c++17 -Wall -Werror \
    -I nntrainer/tensor/hexagon/htp -I nntrainer/tensor/hexagon/host \
    -o /tmp/test_lowering test/hexagon/test_lowering.cpp \
    nntrainer/tensor/hexagon/host/graph_lowering.cpp && /tmp/test_lowering
gcc -Wall -Werror -o /tmp/test_oplist test/hexagon/test_oplist_header.c && /tmp/test_oplist  # 회귀
```

Expected: `LOWER_TEST PASS` + exit 0, 기존 헤더 테스트 PASS 유지.

- [ ] **Step 4: 승인 후 커밋** — `[htp] Add the qwen3 graph lowering emitting the v2 op-list`

---

### Task 8: 가중치 패킹 + RoPE 테이블

**Files:**
- Modify: `nntrainer/tensor/hexagon/host/graph_lowering.cpp` (`pack_weights` 구현)
- Modify: `test/hexagon/test_lowering.cpp` (패킹 테스트 추가)

**Interfaces:**
- Consumes: Task 7 `HexLoweredGraph`/`HexWeightOffsets`.
- Produces: `void pack_weights(const HexLoweredGraph &g, const HexModelConfig &cfg, const HexModelWeights &w, uint8_t *dst)` — `dst`(크기 `g.weights_size`)에 int8 blob memcpy, scale memcpy, norm fp32→fp16 변환(`_Float16` 캐스팅), RoPE 테이블 생성 기록. M4에서 `dst`가 WEIGHTS rpcmem이 된다.

- [ ] **Step 1: 테스트 작성** — tiny 설정, 결정적 합성 가중치(M2 `frand` LCG 재사용: `seed=12345`, proj는 int8 `(int8_t)(i*31+n*7)` 패턴, scale은 `0.001f+0.0001f*n`, norm은 `[0.5,1.5)` fp32):
  - `pack_weights` 후 `HexWeightOffsets`의 각 오프셋에서 memcmp: int8 blob·scale 원본과 바이트 일치.
  - norm: `dst+woff.attn_norm`을 `_Float16`으로 읽어 원본 fp32와 `|d| ≤ 1e-3 + 1e-3·|ref|` 이내.
  - RoPE: 행 p∈{0, 1, max_seq−1}, i∈{0, 1, 63}에 대해 `cosf(p·θ^(−2i/128))`/`sinf(...)` 대비 fp16 오차 `|d| ≤ 2e-3 + 5e-3·|ref|`.
  - tied: embed 영역이 한 번만 존재(embed 오프셋 외에 lm_head용 별도 사본 없음 — `weights_size`가 embed 1벌 기준으로 계산됐는지 오프셋 합산으로 확인).
  - 전 구간 초기화 확인: `dst`를 0xA5로 프리필 후 pack → 패딩 갭 외 모든 텐서 구간이 기록됨.

- [ ] **Step 2: 구현** — 배치 순서 절 그대로. fp16 변환:

```cpp
static inline uint16_t f32_to_f16_bits(float v) {
  _Float16 h = (_Float16)v;
  uint16_t b;
  memcpy(&b, &h, 2);
  return b;
}
```

RoPE 생성은 위 산식 그대로 (`powf(cfg.rope_theta, -2.0f * i / 128.0f)`).

- [ ] **Step 3: 검증 (사용자 실행) — 정지**

```bash
g++ -std=c++17 -Wall -Werror \
    -I nntrainer/tensor/hexagon/htp -I nntrainer/tensor/hexagon/host \
    -o /tmp/test_lowering test/hexagon/test_lowering.cpp \
    nntrainer/tensor/hexagon/host/graph_lowering.cpp && /tmp/test_lowering
```

Expected: `LOWER_TEST PASS`(패킹 검사 포함) + exit 0.

- [ ] **Step 4: 승인 후 커밋** — `[htp] Pack the DSP weight image with the precomputed RoPE table`

---

### Task 9: 문서 갱신 + 전체 회귀

**Files:**
- Modify: `docs/backend_guide/HEXAGON_HVX.md`

**Interfaces:**
- Consumes: Task 1–8 전부.

- [ ] **Step 1: 문서 갱신** — HEXAGON_HVX.md에 M3 반영:
  - 신규 절 "Host pipeline (M3)": W8_CX 포맷(int8 N×K + fp32 scale, 산식), `nntr_quantize`/`nntr_fakequant`/`--eval` 사용법, lowering API와 WEIGHTS/ACT 배치 요약, Task 6의 PPL 결과 표(①/② + wall time).
  - Current status: M3 완료, M4(디바이스 e2e — engine="htp" Context 등록, W8_CX .bin 로드→`HexModelWeights` 어댑터, HexagonRunner 연결) 예고.

- [ ] **Step 2: 전체 회귀 (사용자 실행) — 정지**

```bash
ninja -C build_x86
./build_x86/test/unittest/unittest_nntrainer_quantizer
./build_x86/test/unittest/unittest_nntrainer_save_with_dtype
g++ -std=c++17 -Wall -Werror -I nntrainer/tensor/hexagon/htp -I nntrainer/tensor/hexagon/host \
    -o /tmp/test_lowering test/hexagon/test_lowering.cpp \
    nntrainer/tensor/hexagon/host/graph_lowering.cpp && /tmp/test_lowering
gcc -Wall -Werror -o /tmp/test_oplist test/hexagon/test_oplist_header.c && /tmp/test_oplist
# hexagon sim 회귀 (M2 산출물 무영향 확인, 선택 — SDK 환경 필요)
```

Expected: 전부 PASS/exit 0.

- [ ] **Step 3: 승인 후 커밋** — `[docs] Update the backend guide for the M3 host pipeline`

---

## Self-Review 결과

- **스펙 커버리지 (03-host-pipeline.md):** W8_CX 도구(int8 대칭 per-output-channel, scale fp32) Task 1–3 ✓ / embedding int8 + tied 공유(EMBED·MATMUL_LOGITS 동일 오프셋) Task 3·7 ✓ / 활성화 오프라인 작업 없음(계획에 활성화 관련 작업 부재로 자동 충족) ✓ / graph lowering(정적 오프셋·128B 정렬·타일…) Task 7 ✓ / 가중치 배치(행 정렬·scale·norm fp16·RoPE 테이블) Task 8 ✓ / fake-quant 기준선 Task 4–6 ✓ / x86 테스트(lowering 정렬·경계·op 수, W8_CX 왕복) Task 1·2·7·8 ✓ / PPL <1% 게이트 Task 6 ✓
- **의도적 범위 결정 (스펙과의 차이, 구현자 주의):**
  1. "PER_CHANNEL_AFFINE quantizer 재사용" → **QS4CX free-function 전례 복제**로 대체. 근거: `PerChannelAffineQuantizer::quantize/dequantize`는 NYI 스텁(`quantizer.cpp:184-209`)이고, 실제 저장 경로(`layer_devel.h`)는 quantizer 클래스를 쓰지 않는다. 스텁 구현 부활은 소비자 없는 코드.
  2. fake-quant는 **safetensors 파일 단계**에서 적용(신규 `nntr_fakequant`). 근거: 로드된 모델 가중치를 제자리 수정할 공개 API 부재, save 체인 플래그 관통은 ccapi 공개 시그니처 변경. 같은 `quant_w8cx_f32`를 쓰므로 ②와 실제 .bin의 양자화 값 동일성은 유지된다.
  3. 스펙 "런타임 초기화" 중 **W8_CX .bin 로드(CharTensor 배선)와 rpcmem 배치·init RPC 글루는 M4로 이연**. 근거: 소비자(engine="htp" Context — 현재 미등록, backend guide §6)가 M4에서 생기고, M3의 스펙 명시 테스트 범위(x86: lowering·왕복·fake-quant)에 포함되지 않는다. M3는 그 글루가 호출할 `lower_qwen3`/`pack_weights`까지 완성한다.
  4. lowering의 "타일 파라미터 역산"은 **미포함** — M2 실행기가 VTCM 타일 크기를 DSP에서 자체 산정(`htp_graph_init`)하므로 와이어에 타일 필드가 없다. 스펙 문구는 M2 설계 확정 이전의 것.
  5. W8_CX의 safetensors 출력 포맷은 미지원(명시적 에러). 필요 시 후속.
  6. tiny 설정의 hidden(256) ≠ n_heads·head_dim(512) — qwen3 실치수(1024 vs 2048)와 같은 비율 구조로 q proj의 N≠hidden 경로를 테스트가 강제한다.
- **타입 일관성:** `quant_w8cx_f32(n,k,src,dst,scales)`/`dequant_w8cx_f32` 명칭·인자 순서가 Task 1(정의)·2(저장)·4(fq)·테스트에서 동일. `HexModelConfig`/`HexLayerWeights`/`HexModelWeights`/`HexWeightOffsets`/`HexLoweredGraph`/`lower_qwen3`/`pack_weights`가 Task 7·8·테스트에서 동일. `evaluatePerplexity(const std::string&)`가 transformer.h(가상)·causal_lm(오버라이드)·main.cpp(호출)에서 동일.
- **알려진 리스크:** ① Task 5의 `incremental_inference` from/to·input_len 규약은 실사용 대조로 확정 필요 — PPL 상식 범위 검사(5~50)가 off-by-one을 잡는 안전망. ② x86 meson 전체 빌드가 이 머신(재프로비저닝, Ubuntu 24.04)에서 처음일 수 있음 — Task 1 검증에서 의존성 문제가 나오면 출력 기반으로 해결 후 진행. ③ safetensors 헤더의 JSON 키 순서/메타데이터 보존은 `ordered_json`으로 처리하나 HF 파서 호환은 Task 4 검증(로드 성공)이 최종 판정. ④ CPU fp32 0.6B의 토큰당 추론 시간에 따라 Task 5·6 평가(수백 토큰×2회)가 수십 분일 수 있음 — 평가 텍스트를 200~500토큰으로 제한한 이유.
