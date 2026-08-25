# M4–5: 디바이스 e2e·정확도 검증·성능 튜닝·engine="htp" 통합 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M3까지 만든 W8_CX .bin·lowering·packing을 실제 디바이스(8 Elite)에서 qwen3-0.6b e2e로 돌리고, 4-way 기준선으로 "양자화 손실 vs 커널 버그"를 분리해 정확도 게이트를 통과시킨 뒤, TPS를 측정·튜닝하고 마지막으로 nntrainer `engine="htp"` 경로에 배선한다.

**Architecture:** ① hvx_m3에서 임베딩 W8_CX 양자화 축 결함을 고치고 W8_CX .bin을 산출(브랜치 간 유일한 전달물) → ② hvx_impl에서 .bin을 읽어 `HexModelWeights`로 노출하는 리더 + `nntr_hexpack` 도구가 packed WEIGHTS 이미지(`.hexw`)와 형상 파일(`.hexcfg`)을 생성 → ③ M2의 `ref_graph_forward`를 x86으로 끌어와 같은 이미지를 해석 실행하는 참조 실행기(②')로 lowering·packing·op 시맨틱스를 디바이스 없이 검증 → ④ 같은 이미지를 rpcmem에 올려 돌리는 독립 크로스빌드 디바이스 하네스 + op 인덱스까지만 실행해 중간 활성화를 반환하는 디버그 RPC로 발산 op를 이진 탐색 → ⑤ TPS 측정·HAP_power·청크 스윕 → ⑥ 검증이 끝난 러너를 `engine="htp"`에 배선.

**Tech Stack:** Hexagon SDK 6.3.0.0(QAIC/hexagon-clang v75·v79, `-mhvx-ieee-fp`), Android NDK(aarch64 크로스빌드), x86 gcc 13(`_Float16`), FastRPC/rpcmem, adb(S25 Ultra), python3+transformers(토크나이즈·비교 스크립트)

**Spec:** `docs/superpowers/specs/hexagon-hvx/04-e2e-verification.md` (배경 `00-overview.md`, 호스트 파이프라인 `03-host-pipeline.md`, Q&A `05-design-qna.md`). **`01-rpc-skeleton.md`·`02-hvx-kernels.md`는 스펙 디렉토리에 없다**(완료 후 `docs/superpowers/dones/`로 이동) — 와이어 포맷·op 시맨틱스의 권위는 M2 산출물 `nntrainer/tensor/hexagon/htp/nntr_htp_common.h`와 `dones/02-hvx-kernels.md`다. M3 원장의 동일 Note를 계승한다.

## Global Constraints

- **승인 게이트 (사용자 계약, 최우선) — M3 Ruling 반영:** 검증(빌드/테스트/adb)은 **컨트롤러가 직접 실행하고 출력을 보고**한다 (M3 원장 Ruling: "It would be better test done by yourself" — 상시 위임). **커밋은 여전히 사용자 승인 후에만** 하고, 커밋 메시지 초안을 먼저 보여준다. 구현 서브에이전트는 코드·테스트 작성까지만 하고 빌드/테스트 실행·커밋을 하지 않는다. 실제 출력을 받기 전에 통과/실패 판정이나 완료 보고를 하지 않는다.
- **ponytail:** 모든 구현 서브에이전트에 `ponytail:ponytail`(full) 적용 — 가장 단순한 동작하는 해법, 기존 코드·표준 라이브러리 우선, 투기적 추상화 금지. 리뷰 서브에이전트는 스펙 준수 + 품질 관점.
- **브랜치:** **Task 1만 `hvx_m3`**(임베딩 축 수정 + .bin 재생성), **Task 2 이후 전부 `hvx_impl`**. 두 브랜치는 머지하지 않는다. 브랜치 간 전달물은 파일 3종뿐: W8_CX `.bin`, `nntr_config.json`, M3에서 측정된 ①/② PPL 수치.
- **커밋 형식:** `[태그] 제목` + 바디 + `Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>` + `Co-authored-by: Claude Opus 5 <noreply@anthropic.com>`. 태그는 `[htp]`(hexagon 백엔드), `[causallm]`(앱 트리), `[docs]`.
- **CI 규칙 (신규 파일마다):** `// SPDX-License-Identifier: Apache-2.0` + doxygen 블록(`@file @date @brief @see @author dlwlzzero <dlwlzzero@gmail.com> @bug`), repo `.clang-format` 준수, **영문 주석만**. 셸/파이썬 스크립트는 SPDX 주석 + 실행 비트, **하드코딩 경로 금지**(체커 `static.check.yml` — 모델 경로는 전부 인자/환경변수).
- **plan/spec/handoff 문서는 커밋하지 않는다** (`docs/superpowers/`는 gitignore 대상).
- **디바이스:** S25 Ultra(8 Elite), adb serial `R3CY205ZMND`. S26은 불안정(M2 기록) — 사용 금지. skel 아키텍처는 `HEX_ARCH=v79`.
- **SDK 환경:** `source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.3.0.0/setup_sdk_env.source` 후에만 hexagon/NDK 빌드 스크립트를 실행한다. x86 작업(Task 2·3·4·5)은 SDK 불필요.
- **x86 meson 구성 (M3 Ruling 계승):** 이 머신에 `flatc`가 없어 기본 셋업이 실패한다. `build_x86`는 `meson setup build_x86 -Denable-tflite-interpreter=false -Denable-tflite-backbone=false`로 구성한다. 이미 구성돼 있으면 그대로 쓰고 `ninja -C build_x86`만 돌린다.
- **모델 경로:** `$QWEN3 = /local/mnt/workspace/models/qwen3-0.6b`(사용자 환경). 명령에서만 쓰고 **코드·스크립트·테스트에는 절대 하드코딩하지 않는다**(전부 인자로). 이 디렉토리에는 `model.safetensors`(HF 원본), `nntr_qwen3_0.6b_fp32.bin`(변환된 fp32), `nntr_config.json`(현재 FP32 설정)이 있다.
- **정확도 보고 형식 (사용자 메모리):** 정확도 수치(PPL·로짓 오차·top-1 일치율)는 항상 수행 시간과 짝지어 표로 보고한다 — 디바이스 측정은 `Pcycles`, `wall_ms`, `forward_us`를 병기.
- **정렬 규칙 (M2·M3 계승):** WEIGHTS/ACT의 모든 텐서 오프셋·행 stride는 128B 정렬. 검증은 init 시점 `nntr_htp_oplist_validate`가 담당하고 forward 경로에는 검사가 없다.
- **런타임 인자 방어 (M2 계승):** `htp_graph_forward`는 `n_tokens==0 || n_tokens>max_chunk || pos+n_tokens>max_seq`를 거부한다. 이번에 추가하는 디버그 경로도 같은 검사를 통과한 뒤에만 실행한다.

---

## 공통 설계 (모든 Task의 단일 기준)

### 4-way 기준선 (스펙 3-way의 정제)

스펙은 ①원본 fp32 / ②fake-quant fp32 / ③DSP의 3-way다. 실제로는 ②와 ③ 사이에 **활성화 양자화(w8a8 동적 per-token int8) + fp16 활성화**라는 두 번째 변화가 있어, 그대로 두면 "②vs③ 초과 = 커널 버그"라는 판정이 성립하지 않는다. 그래서 ②'를 하나 끼워 넣는다:

| 기준선 | 실행 주체 | 가중치 | 활성화 | 역할 |
|---|---|---|---|---|
| ① 원본 fp32 | `nntr_causallm --eval` (CPU) | fp32 | fp32 | 진실 기준. M3 측정 완료 |
| ② fake-quant fp32 | `nntr_causallm --eval` (CPU, `nntr_fakequant` 산출물) | W8_CX QDQ | fp32 | 순수 **가중치** 양자화 손실. M3 측정 완료 |
| ②' 참조 실행기 | `hexagon_ref_run` (x86, Task 4) | 실제 packed 이미지 | fp16 + per-token int8 | **DSP와 동일한 수학**의 스칼라 구현. ②vs②' = 활성화 양자화 손실, lowering·packing 결함 |
| ③ DSP | `hexagon_e2e_test` (디바이스, Task 7) | 동일 이미지 | 동일 | ②'vs③ = **순수 커널/하드웨어 구현 오차** |

- **①vs②**: M3에서 통과 — `PPL 19.7010 → 19.7821 (+0.412%) < 1%` (eval_long.txt 1061토큰, wall 497,342ms / 516,685ms). 재측정하지 않고 인용한다.
- **②vs②' 게이트 (Task 5):** eval_short(128토큰)에서 `PPL 델타 ≤ 3%`, `top-1 일치율 ≥ 97%`. 초과 시 원인은 커널이 아니라 lowering/packing/리더 또는 활성화 양자화 설계 — **디바이스로 넘어가지 않는다.**
- **③vs②' 게이트 (Task 8):** 같은 이미지·같은 토큰에서 스텝별 로짓 벡터 비교 `max |Δ| ≤ 2e-2 + 2e-2·|ref|`, `top-1 일치율 ≥ 99.5%`, `PPL 델타 ≤ 0.5%`. 초과 시 커널 버그 → Task 9 이진 탐색.
- **③ 모델 수준 확인 (Task 8):** eval_long(1061토큰) 디바이스 PPL을 ①/② 표에 병기. `PPL③ − PPL②)/PPL② ≤ 3%`.

허용 오차 근거: ②'와 ③은 동일한 fp16 입력·동일 양자화 산식을 쓰므로 차이는 누산 순서(HVX 벡터 reduce vs 스칼라 순차)와 fp16 반올림뿐이다. M2 커널 테스트가 같은 조건에서 `atol 1e-2` 수준을 통과했으므로 451 op 누적을 감안해 2e-2로 잡는다.

### 산출물 파일 포맷 (x86 → 디바이스 전달물)

`nntr_hexpack`(Task 3)이 세 파일을 만든다. **op-list는 파일로 옮기지 않는다** — 소비자(참조 실행기·디바이스 하네스)가 `.hexcfg`를 읽어 `lower_qwen3()`를 직접 호출해 동일 바이트를 재생성하므로, 이미지와 op-list가 어긋날 여지가 없다.

| 파일 | 내용 |
|---|---|
| `<out>.hexw` | packed WEIGHTS 이미지, 크기 = `HexLoweredGraph::weights_size` |
| `<out>.hexcfg` | `HexModelConfig` 11개 필드, `key=value` 한 줄씩 (아래) |
| `<prefix>.tokens.i32` | int32 LE 토큰 ID 배열 (Task 4의 `tools/hexagon/make_tokens.py` 생성, 이미지와 무관) |

`.hexcfg` 정확한 형식 (파서·라이터가 이 순서·표기를 그대로 쓴다):

```
n_layers=28
n_heads=16
n_kv_heads=8
head_dim=128
hidden=1024
ffn=3072
vocab=151936
max_seq=2048
max_chunk=128
rms_eps=1e-06
rope_theta=1e+06
```

부동소수 두 필드는 `printf("%g")`로 쓰고 `strtof`로 읽는다.

### W8_CX .bin 바이트 레이아웃 (Task 2 리더의 단일 기준)

`nntr_quantize --fc_dtype W8_CX --embd_dtype W8_CX` 출력은 **헤더 없는 순차 스트림**이다(`TensorBase::save`는 raw 바이트만 쓴다). 텐서 순서는 그래프 생성 순서이며, W8_CX 2D 가중치는 `int8 blob N*K`(N-major, 행 stride=K) 뒤에 `fp32 scale N개`, 1D(K==1) 텐서는 fp32 원본 그대로다.

qwen3-0.6b (hidden=1024, QDIM=n_heads·head_dim=2048, KVDIM=n_kv_heads·head_dim=1024, ffn=3072, vocab=151936, tied embedding):

| # | 텐서 | 형태 | 바이트 |
|---|---|---|---|
| 0 | `embedding0` (tie_word_embeddings) | int8 [vocab][hidden] + f32[vocab] | 155,582,464 + 607,744 |
| 레이어 l = 0..27 (아래 11개가 이 순서로 반복) | | | |
| l.1 | `attention_norm` | f32[hidden] | 4,096 |
| l.2 | `wq` | int8 [QDIM][hidden] + f32[QDIM] | 2,097,152 + 8,192 |
| l.3 | `q_norm` | f32[head_dim] | 512 |
| l.4 | `wk` | int8 [KVDIM][hidden] + f32[KVDIM] | 1,048,576 + 4,096 |
| l.5 | `k_norm` | f32[head_dim] | 512 |
| l.6 | `wv` | int8 [KVDIM][hidden] + f32[KVDIM] | 1,048,576 + 4,096 |
| l.7 | `attention_out` (wo) | int8 [hidden][QDIM] + f32[hidden] | 2,097,152 + 4,096 |
| l.8 | `ffn_norm` | f32[hidden] | 4,096 |
| l.9 | `ffn_up` | int8 [ffn][hidden] + f32[ffn] | 3,145,728 + 12,288 |
| l.10 | `ffn_gate` | int8 [ffn][hidden] + f32[ffn] | 3,145,728 + 12,288 |
| l.11 | `ffn_down` | int8 [hidden][ffn] + f32[hidden] | 3,145,728 + 4,096 |
| 끝 | `output_norm` | f32[hidden] | 4,096 |

- **up이 gate보다 먼저다** (`transformer.cpp:530-547`의 주석: "nntrainer binary stores mlp weights in up, gate order"). 헷갈리기 쉬운 지점.
- 레이어당 = 15,787,008 B, 28레이어 = 442,036,224 B. **전체 기대 크기 = 156,190,208 + 442,036,224 + 4,096 = 598,230,528 B** (Task 1 수정 후 값).
- tied 모델이므로 lm_head 사본은 파일에 없다.
- `mha_core`, `addition`, `swiglu`, KV cache placeholder(`input`) 레이어는 가중치가 없어 스트림에 기여하지 않는다.

### Task 1이 고치는 결함 (M3에서 리스크로 기록된 항목의 확정)

`tie_word_embedding.cpp`(및 같은 패턴의 `embedding_layer.cpp`)의 W8_CX 저장 분기가 `K = dim.height()`, `N = dim.width()`를 그대로 써서 `quant_w8cx_f32(N, K, ...)`를 호출한다. 임베딩 가중치의 `finalize()`는 `dim.height(in_dim=vocab); dim.width(out_dim=hidden)`이므로(`tie_word_embedding.cpp:100-101`) **N=1024(hidden), K=151936(vocab)** 이 되어, 실제 메모리 배치([vocab][hidden] row-major)와 어긋난 축으로 양자화된다. 결과는 scale 1024개(vocab별 아님) + 잘못된 그룹핑이며, `nntr_fakequant`(②)는 HF 텐서 [vocab][hidden]을 N=vocab로 양자화하므로 **②와 ③의 임베딩 값이 서로 다르다** — 3-way 판정의 전제가 깨진다.

산술적 확정: M3에서 실측된 .bin 크기 **597,626,880 B** = 위 표의 598,230,528 B − (607,744 − 4,096). 즉 임베딩 scale이 vocab개(607,744 B)가 아니라 hidden개(4,096 B)로 저장됐음이 파일 크기로 증명된다. 수정 후 크기가 598,230,528 B가 되는 것이 Task 1의 판정 기준이다.

### 디버그 RPC 규약 (Task 6)

ABI를 **v3으로 올린다**. op-list 와이어 포맷 자체는 변하지 않지만 IDL 인터페이스가 바뀌므로, 디바이스에 오래된 skel이 남아 있는 실패 모드를 init 핸드셰이크에서 잡기 위함이다.

```
AEEResult forward(in sequence<int32> token_ids, in uint32 pos,
                  rout sequence<float> logits, rout uint64 dsp_pcycles);
AEEResult forward_debug(in sequence<int32> token_ids, in uint32 pos,
                        in uint32 n_ops_limit,
                        in uint32 dump_buf, in uint32 dump_offset,
                        rout sequence<uint8> dump, rout uint64 dsp_pcycles);
```

- `forward_debug`는 op `[0, n_ops_limit)`만 실행한 뒤 `bufs[dump_buf] + dump_offset`에서 `dump` 길이만큼 복사해 반환한다. `n_ops_limit > n_ops`, `dump_buf >= NNTR_HTP_BUF_COUNT`, `dump_offset + dumpLen > buf_size[dump_buf]`는 거부(`AEE_EBADPARM`).
- KV cache는 부분 실행에서도 갱신되므로, 디버그 세션은 매 비교마다 `pos=0`부터 다시 시작한다(하네스가 보장).
- `dsp_pcycles`는 `HAP_perf_get_pcycles()` 차분. forward 경로 오버헤드는 스칼라 2회 읽기뿐이다.

### 디렉토리·명명 (M2·M3 규칙 계승)

- 모델 지식(qwen3 형상·.bin 순서)은 앱 트리: `Applications/CausalLM/hexagon/`.
- 백엔드 코어(모델 무관)는 `nntrainer/tensor/hexagon/host/`.
- 테스트/하네스 실행 파일은 `test/hexagon/`(평탄), 시뮬레이터 전용은 `test/hexagon/sim/`.
- 스크립트는 `tools/hexagon/`.
- op 구현 함수는 `hvx_op_*`, 계약명은 `htp_*` 유지.

---

### Task 1: 임베딩 W8_CX 양자화 축 수정 + W8_CX .bin 재생성 (브랜치: `hvx_m3`)

**Files:**
- Modify: `Applications/CausalLM/layers/tie_word_embedding.cpp` (W8_CX 분기, 약 532-545행)
- Modify: `Applications/CausalLM/layers/embedding_layer.cpp` (W8_CX 분기 — 같은 결함, untied 모델용)

**Interfaces:**
- Consumes: `nntrainer::quant_w8cx_f32(size_t n, size_t k, void *src_f32, void *dst_i8, void *scales_f32)` (M3 Task 1).
- Produces: `$QWEN3/nntr_qwen3_0.6b_w8cx.bin` — 공통 설계 §"W8_CX .bin 바이트 레이아웃"을 정확히 따르는 598,230,528 B 파일. Task 2 이후의 유일한 입력.

- [ ] **Step 1: 수정 전 상태 고정 (컨트롤러 실행)**

`nntr_quantize`는 HF safetensors를 직접 읽지 못한다(M3 Ruling) — 모델 디렉토리의 `nntr_config.json`이 가리키는 **fp32 .bin이 입력**이고, `--fc_dtype/--embd_dtype/--lmhead_dtype`가 출력 dtype이다. 현재 `$QWEN3/nntr_config.json`은 FP32 + `model_file_name: nntr_qwen3_0.6b_fp32.bin`이라 그대로 입력으로 쓰면 된다(별도 config 사본 불필요).

```bash
git checkout hvx_m3
ninja -C build_x86
grep -E '"(model_file_name|fc_layer_dtype|embedding_dtype)"' "$QWEN3/nntr_config.json"
ls -l "$QWEN3/nntr_qwen3_0.6b_fp32.bin"
```

Expected: 빌드 성공, config가 FP32 + fp32 .bin(2,384,199,680 B)을 가리킴. 다르면 M3 Task 5의 FP32 변환 결과가 바뀐 것이므로 원장과 대조한 뒤 진행한다.

- [ ] **Step 2: 두 파일의 W8_CX 분기 수정**

`tie_word_embedding.cpp`의 W8_CX 분기를 아래로 교체한다. `K = dim.height()`(=vocab), `N = dim.width()`(=hidden)이므로 **인자 순서를 뒤집어** 메모리 배치([vocab][hidden] row-major)와 양자화 축을 맞춘다:

```cpp
          } else if (dtype == nntrainer::TensorDim::DataType::W8_CX) {
            if (K == 1) {
              weight.save(file);
            } else {
              // The embedding weight is laid out [in_dim][out_dim] row major
              // (finalize sets height=in_dim, width=out_dim), so one row is
              // one vocabulary entry: quantize with n=K rows of k=N values.
              // Passing (N, K) here would slice across vocabulary rows and
              // emit N scales instead of one per row.
              const size_t data_size = (size_t)N * K;
              const size_t scale_size = (size_t)K * sizeof(float);

              std::vector<uint8_t> rhs_q(data_size + scale_size);
              nntrainer::quant_w8cx_f32(K, N, weight.getData(), rhs_q.data(),
                                        rhs_q.data() + data_size);
              file.write((const char *)rhs_q.data(), data_size + scale_size);
            }
          } else {
```

`embedding_layer.cpp`의 W8_CX 분기도 동일하게 바꾼다(그쪽은 `K == 1` 가드가 없으므로 위와 같은 형태로 가드를 포함해 맞춘다):

```cpp
        } else if (dtype == nntrainer::TensorDim::DataType::W8_CX) {
          // See tie_word_embedding.cpp: rows are vocabulary entries, so the
          // row count is K (height) and the row length is N (width).
          const size_t data_size = (size_t)N * K;
          const size_t scale_size = (size_t)K * sizeof(float);

          std::vector<uint8_t> rhs_q(data_size + scale_size);
          nntrainer::quant_w8cx_f32(K, N, weight.getData(), rhs_q.data(),
                                    rhs_q.data() + data_size);
          file.write((const char *)rhs_q.data(), data_size + scale_size);
        } else {
```

- [ ] **Step 3: 재생성 + 크기 판정 (컨트롤러 실행)**

```bash
ninja -C build_x86
mkdir -p "$QWEN3/w8cx"
./build_x86/Applications/CausalLM/nntr_quantize "$QWEN3" \
    --fc_dtype W8_CX --embd_dtype W8_CX --lmhead_dtype W8_CX \
    -o "$QWEN3/w8cx"
ls -l "$QWEN3/w8cx/"*.bin "$QWEN3/w8cx/nntr_config.json"
```

Expected: exit 0(M3 실측 wall 2.9s), 출력 `nntr_qwen3_0.6b_w8cx_DEFAULT.bin`(파일명 suffix는 `_w8cx` — 도구가 dtype 문자열에서 밑줄을 제거한다, M3 Ruling), 크기가 **정확히 598,230,528 B**, 생성된 `nntr_config.json`의 fc/embd/lmhead가 모두 `W8_CX`이고 `model_tensor_type`이 `W8_CX-FP32`.

수정 전 값 **597,626,880 B**가 나오면 두 분기 중 어느 쪽도 실제 경로가 아니었다는 뜻이다 — qwen3는 `tie_word_embeddings: true`이므로 `tie_word_embedding.cpp`가 실경로이고 `embedding_layer.cpp`는 untied 모델용 예방 수정이다.

- [ ] **Step 4: 값 검증 (컨트롤러 실행)**

임베딩 첫 행이 vocab 행 단위로 양자화됐는지 safetensors 원본과 대조한다:

```bash
python3 - "$QWEN3/w8cx"/*.bin "$QWEN3/model.safetensors" <<'PY'
import sys, json, struct
import numpy as np
binp, stp = sys.argv[1], sys.argv[2]
VOCAB, HID = 151936, 1024
q = np.fromfile(binp, dtype=np.int8, count=VOCAB*HID).reshape(VOCAB, HID)
s = np.fromfile(binp, dtype=np.float32, count=VOCAB, offset=VOCAB*HID)
assert s.shape == (VOCAB,) and np.all(np.isfinite(s)) and np.all(s > 0), "scale block broken"
with open(stp, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n)); base = 8 + n
    e = hdr["model.embed_tokens.weight"]
    off, end = e["data_offsets"]; dt = e["dtype"]
    f.seek(base + off); raw = f.read(end - off)
ref = (np.frombuffer(raw, dtype=np.float32) if dt == "F32"
       else (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
       ).reshape(VOCAB, HID)
deq = q.astype(np.float32) * s[:, None]
err = np.linalg.norm(deq[:64] - ref[:64]) / np.linalg.norm(ref[:64])
print(f"scale[0..3]={s[:4]} rel_rms_err={err:.4%}")
assert err < 0.05, "embedding axis still wrong"
print("EMBED_AXIS OK")
PY
```

Expected: `EMBED_AXIS OK`, `rel_rms_err` 1% 내외(int8 전형). 실패하면 축이 여전히 어긋난 것이다.

- [ ] **Step 5: 승인 후 커밋 (hvx_m3)** — `[causallm] Fix the W8_CX embedding quantization axis`

바디에 담을 것: 결함(메모리 [vocab][hidden] vs 호출 인자 (N=hidden, K=vocab)), 증상(scale 1024개, ②와 값 불일치), 파일 크기 597,626,880 → 598,230,528 B.

- [ ] **Step 6: 산출물 고정 + 브랜치 전환**

`.bin`과 `nntr_config_w8cx.json` 경로를 진행 원장에 적고 `git checkout hvx_impl`. 이후 Task는 전부 hvx_impl이며 hvx_m3를 다시 건드리지 않는다.

---

### Task 2: W8_CX .bin 리더

**Files:**
- Create: `Applications/CausalLM/hexagon/qwen3_w8cx_bin.h`, `qwen3_w8cx_bin.cpp`
- Modify: `Applications/CausalLM/hexagon/meson.build` (없으면 `Applications/CausalLM/meson.build`의 hexagon 블록 — Task 7의 lowering 배선 위치를 따른다)
- Create: `test/hexagon/test_w8cx_bin.cpp`
- Create: `tools/hexagon/build_host_x86.sh` (x86 호스트 도구/테스트 공용 빌드 스크립트)

**Interfaces:**
- Consumes: `HexModelConfig`, `HexModelWeights`, `HexLayerWeights` (`nntrainer/tensor/hexagon/host/graph_lowering.h`).
- Produces:

```cpp
namespace nntrainer::hexagon {

/**
 * @brief Read-only view over an nntr_quantize W8_CX .bin. mmaps the file and
 *        hands out non-owning pointers into it; the object must outlive the
 *        HexModelWeights it returns.
 */
class Qwen3W8cxBin {
public:
  /** @throw std::runtime_error on open/size/structure mismatch. */
  Qwen3W8cxBin(const std::string &path, const HexModelConfig &cfg);
  ~Qwen3W8cxBin();
  Qwen3W8cxBin(const Qwen3W8cxBin &) = delete;
  Qwen3W8cxBin &operator=(const Qwen3W8cxBin &) = delete;

  const HexModelWeights &weights() const { return w_; }
  uint64_t file_size() const { return size_; }

  /** Byte size the checkpoint must have for this shape. */
  static uint64_t expected_size(const HexModelConfig &cfg);

private:
  int fd_ = -1;
  uint8_t *base_ = nullptr;
  uint64_t size_ = 0;
  HexModelWeights w_;
};

} // namespace nntrainer::hexagon
```

- [ ] **Step 1: 빌드 스크립트 작성** — `tools/hexagon/build_host_x86.sh`: x86에서 hexagon 호스트 소스만 골라 도구/테스트를 만든다(nntrainer 전체 빌드 불필요, M3 Task 7·8의 직접 g++ 호출을 스크립트화).

```bash
#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Builds the x86 host-side hexagon tools/tests (no SDK, no device).
# Usage: ./tools/hexagon/build_host_x86.sh [target ...]
#   targets: test_lowering test_w8cx_bin nntr_hexpack hexagon_ref_run (default: all)

set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO/build_x86_hexagon"
HOST="$REPO/nntrainer/tensor/hexagon/host"
HTP="$REPO/nntrainer/tensor/hexagon/htp"
APP="$REPO/Applications/CausalLM/hexagon"
SIM="$REPO/test/hexagon/sim"
CXX="${CXX:-g++}"

mkdir -p "$OUT"
INCS=(-I "$HOST" -I "$HTP" -I "$APP" -I "$SIM")
CXXFLAGS=(-std=c++17 -O2 -Wall -Werror "${INCS[@]}")

LOWER=("$HOST/graph_lowering.cpp" "$APP/qwen3_lowering.cpp")
IMAGE=("$APP/qwen3_w8cx_bin.cpp" "$APP/hex_image.cpp")

build_test_lowering() {
  "$CXX" "${CXXFLAGS[@]}" "$REPO/test/hexagon/test_lowering.cpp" "${LOWER[@]}" \
      -o "$OUT/test_lowering"
}
build_test_w8cx_bin() {
  "$CXX" "${CXXFLAGS[@]}" "$REPO/test/hexagon/test_w8cx_bin.cpp" \
      "$APP/qwen3_w8cx_bin.cpp" -o "$OUT/test_w8cx_bin"
}
...
```

(나머지 타깃 함수는 해당 Task에서 채운다. `hex_image.cpp`는 Task 3, `hexagon_ref_run`은 Task 4에서 추가.)

- [ ] **Step 2: 실패하는 테스트 작성** — `test/hexagon/test_w8cx_bin.cpp` (M2/M3 관례대로 gtest 아님, self-contained main, PASS 시 `W8CX_BIN_TEST PASS` + exit 0). `.bin` 경로를 `argv[1]`로 받는다(하드코딩 금지). 인자가 없으면 사용법 출력 후 exit 2.

```cpp
static const nntrainer::hexagon::HexModelConfig kQwen3 = {
  /*n_layers=*/28, /*n_heads=*/16, /*n_kv_heads=*/8, /*head_dim=*/128,
  /*hidden=*/1024, /*ffn=*/3072, /*vocab=*/151936, /*max_seq=*/2048,
  /*max_chunk=*/128, /*rms_eps=*/1e-6f, /*rope_theta=*/1e6f};

int main(int argc, char **argv) {
  if (argc < 2) { printf("usage: %s <w8cx.bin>\n", argv[0]); return 2; }

  // 1. shape arithmetic, independent of the file
  CHECK(Qwen3W8cxBin::expected_size(kQwen3) == 598230528ull);

  Qwen3W8cxBin bin(argv[1], kQwen3);       // throws on size mismatch
  const HexModelWeights &w = bin.weights();
  CHECK(bin.file_size() == 598230528ull);
  CHECK(w.layers.size() == 28u);

  // 2. every scale block is finite and strictly positive - a cursor that
  //    drifted by one tensor reads int8 payload as float and fails here
  CHECK(all_positive_finite(w.embed_s, kQwen3.vocab));
  CHECK(all_positive_finite(w.final_norm, kQwen3.hidden) == false ||
        true); /* gammas may be negative; only scales are checked */
  for (const auto &l : w.layers) {
    CHECK(all_positive_finite(l.wq_s, 2048));
    CHECK(all_positive_finite(l.wk_s, 1024));
    CHECK(all_positive_finite(l.wv_s, 1024));
    CHECK(all_positive_finite(l.wo_s, 1024));
    CHECK(all_positive_finite(l.w_gate_s, 3072));
    CHECK(all_positive_finite(l.w_up_s, 3072));
    CHECK(all_positive_finite(l.w_down_s, 1024));
  }

  // 3. RMSNorm gammas of a trained model sit near 1.0 - catches a cursor that
  //    landed on int8 payload (which reads as denormal/garbage floats)
  for (const auto &l : w.layers) {
    CHECK(mean_abs(l.attn_norm, 1024) > 0.05f && mean_abs(l.attn_norm, 1024) < 10.0f);
    CHECK(mean_abs(l.ffn_norm, 1024) > 0.05f && mean_abs(l.ffn_norm, 1024) < 10.0f);
    CHECK(mean_abs(l.q_norm, 128) > 0.05f && mean_abs(l.q_norm, 128) < 10.0f);
    CHECK(mean_abs(l.k_norm, 128) > 0.05f && mean_abs(l.k_norm, 128) < 10.0f);
  }
  CHECK(mean_abs(w.final_norm, 1024) > 0.05f && mean_abs(w.final_norm, 1024) < 10.0f);

  // 4. int8 payload is not degenerate (a zero block means a wrong offset)
  for (const auto &l : w.layers)
    CHECK(nonzero_frac(l.wq, 2048ull * 1024) > 0.5f);

  printf("W8CX_BIN_TEST PASS\n");
  return 0;
}
```

헬퍼 3종(`all_positive_finite`, `mean_abs`, `nonzero_frac`)은 같은 파일 안 static 함수로 둔다.

- [ ] **Step 3: 테스트 실패 확인 (컨트롤러 실행)**

```bash
./tools/hexagon/build_host_x86.sh test_w8cx_bin
```

Expected: 컴파일 실패(`qwen3_w8cx_bin.h` 없음). 구현 전이므로 정상.

- [ ] **Step 4: 구현** — `qwen3_w8cx_bin.cpp`. 공통 설계의 레이아웃 표를 그대로 순차 커서로 읽는다.

```cpp
namespace {
struct Cursor {
  uint8_t *p;
  uint8_t *end;
  const int8_t *i8(uint64_t n) { return (const int8_t *)take(n); }
  const float *f32(uint64_t n) { return (const float *)take(n * 4ull); }
  void *take(uint64_t bytes) {
    if ((uint64_t)(end - p) < bytes)
      throw std::runtime_error("w8cx bin: truncated");
    void *r = p;
    p += bytes;
    return r;
  }
};
/** int8 [n][k] blob followed by n fp32 scales. */
void quantized(Cursor &c, const int8_t *&q, const float *&s, uint64_t n,
               uint64_t k) {
  q = c.i8(n * k);
  s = c.f32(n);
}
} // namespace
```

**정렬 주의 (M3 minor 지적 계승):** `.bin`의 fp32 scale 블록은 앞선 int8 blob 크기가 4의 배수일 때만 4바이트 정렬된다. qwen3-0.6b는 모든 `N*K`가 4의 배수라 안전하지만, 리더는 `f32()`에서 `assert(((uintptr_t)p & 3u) == 0)`로 이 전제를 명시한다(어긋나면 조용한 UB 대신 즉시 실패).

`expected_size()`는 같은 산식을 포인터 없이 계산해 생성자가 mmap 전에 비교한다(크기 불일치는 즉시 예외 — 텐서 순서가 다르면 거의 항상 여기서 걸린다). 레이어 루프는 `attn_norm, wq, q_norm, wk, k_norm, wv, wo, ffn_norm, w_up, w_gate, w_down` 순서로 읽는다(**up이 gate보다 먼저**).

- [ ] **Step 5: 검증 (컨트롤러 실행)**

```bash
./tools/hexagon/build_host_x86.sh test_w8cx_bin
./build_x86_hexagon/test_w8cx_bin "$QWEN3/w8cx"/*.bin
```

Expected: `W8CX_BIN_TEST PASS` + exit 0.

- [ ] **Step 6: 텐서 순서 교차 검증 (컨트롤러 실행)**

크기·구조 검사는 **같은 크기 텐서의 뒤바뀜**(wk↔wv, up↔gate)을 잡지 못한다. safetensors 원본과 dequant 대조로 확정한다:

```bash
python3 tools/hexagon/check_w8cx_bin.py "$QWEN3/w8cx"/*.bin "$QWEN3/model.safetensors"
```

`tools/hexagon/check_w8cx_bin.py`(이 Step에서 함께 작성): 레이아웃 표대로 오프셋을 계산해 레이어 0·13·27의 `wq/wk/wv/wo/up/gate/down` 각각에 대해 `dequant(q)·s` vs HF 텐서(`model.layers.{l}.self_attn.{q,k,v,o}_proj.weight`, `mlp.{up,gate,down}_proj.weight`)의 상대 RMS 오차를 출력하고, 전부 `< 3%`면 `W8CX_LAYOUT OK`를 찍는다. 순서가 뒤바뀌면 오차가 100% 근처로 튄다.

Expected: `W8CX_LAYOUT OK`, 각 오차 1% 내외.

- [ ] **Step 7: 승인 후 커밋** — `[causallm] Add the W8_CX checkpoint reader for the DSP weight packer`

---

### Task 3: 가중치 이미지 빌더 `nntr_hexpack`

**Files:**
- Create: `Applications/CausalLM/hexagon/hex_image.h`, `hex_image.cpp` (`.hexcfg` 읽기/쓰기 + 파일 IO 헬퍼)
- Create: `Applications/CausalLM/hexagon/hex_pack.cpp` (도구 main)
- Modify: `Applications/CausalLM/meson.build` (또는 `hexagon/meson.build`) — `nntr_hexpack` 실행 파일 정의, M3의 `nntr_fakequant` 블록 패턴 복제
- Modify: `tools/hexagon/build_host_x86.sh` (`nntr_hexpack` 타깃)
- Modify: `test/hexagon/test_lowering.cpp` (`.hexcfg` 왕복 테스트 추가)

**Interfaces:**
- Consumes: `Qwen3W8cxBin`(Task 2), `lower_qwen3`/`pack_weights`(M3 Task 7·8).
- Produces:

```cpp
namespace nntrainer::hexagon {
/** Write/read the 11-field .hexcfg text file. @throw std::runtime_error */
void write_hexcfg(const std::string &path, const HexModelConfig &cfg);
HexModelConfig read_hexcfg(const std::string &path);
} // namespace nntrainer::hexagon
```

그리고 CLI:

```
nntr_hexpack <w8cx.bin> <out-prefix> [--layers N] [--max-seq N] [--max-chunk N]
```

출력은 `<out-prefix>.hexw`, `<out-prefix>.hexcfg`. `--layers N`은 이미지에 담을 레이어 수를 N으로 줄인다(디바이스 브링업용 1레이어 모델). 형상 상수(hidden/heads/ffn/vocab/eps/theta)는 qwen3-0.6b 값을 도구 안에 상수로 둔다 — `config.json` 파싱은 소비자가 하나뿐이라 불필요(YAGNI).

- [ ] **Step 1: 실패하는 테스트 작성** — `test/hexagon/test_lowering.cpp`에 `.hexcfg` 왕복 검사를 추가:

```cpp
  /* .hexcfg round trip: every field survives text serialization. */
  {
    const char *path = "/tmp/hexcfg_roundtrip.hexcfg";
    write_hexcfg(path, tiny);            /* tiny = the existing test config */
    HexModelConfig back = read_hexcfg(path);
    CHECK(back.n_layers == tiny.n_layers && back.n_heads == tiny.n_heads &&
          back.n_kv_heads == tiny.n_kv_heads && back.head_dim == tiny.head_dim &&
          back.hidden == tiny.hidden && back.ffn == tiny.ffn &&
          back.vocab == tiny.vocab && back.max_seq == tiny.max_seq &&
          back.max_chunk == tiny.max_chunk);
    CHECK(back.rms_eps == tiny.rms_eps && back.rope_theta == tiny.rope_theta);
    remove(path);
  }
```

`rms_eps`/`rope_theta`의 `==` 비교가 성립하도록 `%g`는 왕복 정확도가 필요하다 — 구현에서 `%.9g`를 쓴다.

- [ ] **Step 2: 테스트 실패 확인 (컨트롤러 실행)**

```bash
./tools/hexagon/build_host_x86.sh test_lowering
```

Expected: 컴파일 실패(`hex_image.h` 없음).

- [ ] **Step 3: 구현**

`hex_image.cpp`:

```cpp
void write_hexcfg(const std::string &path, const HexModelConfig &c) {
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error("hexcfg: cannot write " + path);
  f << "n_layers=" << c.n_layers << "\nn_heads=" << c.n_heads
    << "\nn_kv_heads=" << c.n_kv_heads << "\nhead_dim=" << c.head_dim
    << "\nhidden=" << c.hidden << "\nffn=" << c.ffn << "\nvocab=" << c.vocab
    << "\nmax_seq=" << c.max_seq << "\nmax_chunk=" << c.max_chunk << "\n";
  char buf[64];
  snprintf(buf, sizeof(buf), "rms_eps=%.9g\nrope_theta=%.9g\n", c.rms_eps,
           c.rope_theta);
  f << buf;
  if (!f)
    throw std::runtime_error("hexcfg: write failed " + path);
}
```

`read_hexcfg`는 `key=value` 줄을 맵에 모은 뒤 11개 키가 전부 있는지 확인하고(없으면 예외) 채운다.

`hex_pack.cpp` main 흐름:

```cpp
  HexModelConfig full = kQwen3;                       /* checkpoint shape */
  full.max_seq = opt_max_seq;                         /* default 2048 */
  full.max_chunk = opt_max_chunk;                     /* default 128 */
  Qwen3W8cxBin bin(argv[1], full);                    /* reads all 28 layers */

  HexModelConfig cfg = full;
  cfg.n_layers = opt_layers ? opt_layers : full.n_layers;

  HexModelWeights w = bin.weights();
  w.layers.resize(cfg.n_layers);                      /* 1-layer bring-up */

  HexLoweredGraph g = lower_qwen3(cfg);
  std::vector<uint8_t> image(g.weights_size);
  pack_weights(g, cfg, w, image.data());
  write_file(prefix + ".hexw", image.data(), image.size());
  write_hexcfg(prefix + ".hexcfg", cfg);
  printf("HEXPACK weights=%llu kv=%llu act=%llu n_ops=%u\n", ...);
```

- [ ] **Step 4: 검증 (컨트롤러 실행)**

```bash
./tools/hexagon/build_host_x86.sh test_lowering nntr_hexpack
./build_x86_hexagon/test_lowering
./build_x86_hexagon/nntr_hexpack "$QWEN3/w8cx"/*.bin /tmp/qwen3_full
./build_x86_hexagon/nntr_hexpack "$QWEN3/w8cx"/*.bin /tmp/qwen3_l1 --layers 1
ls -l /tmp/qwen3_full.hexw /tmp/qwen3_l1.hexw && cat /tmp/qwen3_l1.hexcfg
```

Expected:
- `LOWER_TEST PASS` + exit 0.
- full: `HEXPACK weights=598631424 kv=234881024 act=... n_ops=451` — `weights_size`는 M3 Task 7에서 재계산된 598,623,744 B에 RoPE 테이블(2048·128·2 = 524,288 B)과 정렬 패딩을 더한 값 근처(±1MB). 정확한 기대값은 도구 출력으로 고정하고 이후 Task가 이 수치를 인용한다.
- l1: `n_ops=19`, `.hexcfg`의 `n_layers=1`.
- `.hexw` 파일 크기 = 출력된 `weights=` 값과 일치.

- [ ] **Step 5: 승인 후 커밋** — `[causallm] Add nntr_hexpack building the DSP weight image from a W8_CX checkpoint`

---

### Task 4: x86 참조 실행기 `hexagon_ref_run` (기준선 ②')

**Files:**
- Modify: `test/hexagon/sim/ref_ops.h` (x86용 `__fp16` shim + `ref_graph_forward_upto` 선언)
- Modify: `test/hexagon/sim/ref_ops.c` (`ref_graph_forward`를 `_upto`로 위임)
- Create: `test/hexagon/hexagon_ref_run.cpp` (CLI)
- Create: `tools/hexagon/make_tokens.py`
- Modify: `tools/hexagon/build_host_x86.sh` (`hexagon_ref_run` 타깃)

**Interfaces:**
- Consumes: `read_hexcfg`(Task 3), `lower_qwen3`, `ref_*`(M2), `.hexw`/`.hexcfg`/`.tokens.i32`.
- Produces:
  - `void ref_graph_forward_upto(const uint8_t *oplist, uint8_t *weights, uint8_t *kv, uint8_t *act, const int32_t *tokens, uint32_t n_tokens, uint32_t pos, float *logits, uint32_t n_ops_limit);`
  - CLI `hexagon_ref_run <prefix> --tokens <f> [--chunk N] [--eval] [--dump-op i --dump-out <f>] [--logits <f>] [--steps N]` — Task 5·8·9가 소비하는 오라클.

- [ ] **Step 1: 실패하는 테스트 작성**

`test/hexagon/sim/test_graph.c`(시뮬레이터, M2 골든 테스트)에 부분 실행 케이스를 추가한다 — 이 파일은 hexagon-sim에서 도는 기존 회귀이므로, 여기에 넣으면 x86과 sim 양쪽에서 같은 계약을 지킨다:

```c
  /* Partial execution: running ops [0, limit) then the rest must equal a
   * single full run (the executor keeps no per-call state besides KV). */
  {
    float logits_a[VOCAB], logits_b[VOCAB];
    memset(kv, 0, KV_BYTES);
    ref_graph_forward(oplist, weights, kv, act, tokens, 3u, 0u, logits_a);
    memset(kv, 0, KV_BYTES);
    ref_graph_forward_upto(oplist, weights, kv, act, tokens, 3u, 0u, logits_b,
                           N_OPS);
    for (uint32_t i = 0; i < VOCAB; ++i)
      CHECK(logits_a[i] == logits_b[i]);
    /* a truncated run must leave the logits buffer untouched */
    memset(kv, 0, KV_BYTES);
    memset(logits_b, 0, sizeof(logits_b));
    ref_graph_forward_upto(oplist, weights, kv, act, tokens, 3u, 0u, logits_b,
                           N_OPS - 1u);
    CHECK(logits_b[0] == 0.0f);
  }
```

- [ ] **Step 2: 테스트 실패 확인 (컨트롤러 실행)**

```bash
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.3.0.0/setup_sdk_env.source
./tools/hexagon/build_sim_test.sh
```

Expected: 컴파일 실패(`ref_graph_forward_upto` 미정의).

- [ ] **Step 3: 구현 — ref_ops 확장**

`ref_ops.h` 맨 위(다른 선언보다 먼저):

```c
/* __fp16 is an ARM/Hexagon spelling; on x86 the equivalent storage type is
 * _Float16 (gcc 12+). The reference executor is compiled for both. */
#if !defined(__hexagon__) && !defined(__ARM_FP16_FORMAT_IEEE) &&               \
  !defined(__aarch64__)
typedef _Float16 __fp16;
#endif
```

`ref_ops.c`: 기존 `ref_graph_forward` 본문을 `ref_graph_forward_upto(..., uint32_t n_ops_limit)`로 옮기고 루프 상한을 `h.n_ops < n_ops_limit ? h.n_ops : n_ops_limit`로, 원래 함수는 `n_ops_limit = h.n_ops`로 호출하는 3줄 래퍼로 만든다. **다른 동작 변경 금지**(M2 골든 테스트가 이 함수에 걸려 있다).

- [ ] **Step 4: 구현 — CLI**

`test/hexagon/hexagon_ref_run.cpp`:

```cpp
/*
 * Modes (exactly one of --eval / --dump-op / default):
 *   default  : prefill in chunks then greedy-decode --steps tokens, print ids
 *   --eval   : teacher-forced PPL over the token file (one token per step)
 *   --dump-op: run ops [0, i) for the given chunk and write the op's output
 *              tensor bytes to --dump-out (used by the divergence search)
 */
```

- 실행 준비: `read_hexcfg(prefix + ".hexcfg")` → `lower_qwen3(cfg)` → `.hexw`를 `weights_size`만큼 읽어 128B 정렬 버퍼에, `kv_size`/`act_size`는 `calloc`.
- forward 한 번 = `ref_graph_forward(g.oplist.data(), weights, kv, act, tokens+p, n, p, logits)`.
- `--eval`: `n_tokens=1`, `pos=p`로 `p = 0..n-2` 반복, 매 스텝 `logits[tokens[p+1]]`의 log-softmax를 누적해 `PPL <v> wall_ms <t> steps <n-1>` 출력. 판정 규약은 M3 `--eval`과 동일하게 맞춘다(같은 텍스트에서 비교하기 위함).
- `--logits <f>`: 스텝별 마지막-토큰 로짓 벡터를 fp32로 이어붙여 파일에 쓴다(`--steps`로 스텝 수 제한 — 128스텝 × 151936 × 4 = 74MB).
- `--dump-op i --dump-out f`: `ref_graph_forward_upto(..., i)` 후 `ops[i-1].out`(`buf`,`offset`)이 가리키는 영역을 `m * n * 2` 바이트(op 종류별 크기는 `nntr_htp_oplist_validate`의 산식과 동일하게 계산) 덤프. 덤프 대상 버퍼 id/오프셋/바이트를 stdout에 `DUMP buf=%u off=%u bytes=%u` 형식으로 함께 출력한다 — 디바이스 쪽이 같은 인자를 그대로 쓰기 위함.

`tools/hexagon/make_tokens.py`:

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tokenize a text file into an int32 LE token-id file for the hexagon harnesses.

Usage: make_tokens.py <tokenizer.json|model-dir> <text> <out.tokens.i32> [--limit N]
       --limit N also writes <out>.txt, the detokenized prefix of N tokens, so
       the CPU baselines can be evaluated on exactly the same tokens.
"""
```

`transformers.AutoTokenizer`(M3 Task 3에서 설치됨)를 쓰고, 없으면 `tokenizers.Tokenizer.from_file`로 폴백한다.

- [ ] **Step 5: 검증 (컨트롤러 실행)**

```bash
# sim 회귀(부분 실행 계약)
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.3.0.0/setup_sdk_env.source
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh graph

# x86 참조 실행기: 1레이어 이미지로 스모크 (전체 모델은 Task 5)
./tools/hexagon/build_host_x86.sh hexagon_ref_run
python3 tools/hexagon/make_tokens.py "$QWEN3" /tmp/eval_long.txt /tmp/eval_long.tokens.i32 --limit 128
./build_x86_hexagon/hexagon_ref_run /tmp/qwen3_l1 --tokens /tmp/eval_long.tokens.i32 --steps 4
```

Expected: `graph` 테스트 PASS(부분 실행 케이스 포함), 참조 실행기가 1레이어 모델에서 토큰 4개를 생성하고 exit 0(값의 의미는 없음 — 경로 스모크). `/tmp/eval_long.txt`가 없으면 M3의 평가 텍스트를 다시 만들어 두고(1061토큰 산문) 그 사실을 원장에 기록한다.

- [ ] **Step 6: 승인 후 커밋** — `[htp] Add the x86 reference op-list runner for accuracy baselines`

---

### Task 5: x86 정확도 게이트 (②' vs ②)

**Files:** 신규 코드 없음 (Task 1–4 산출물 + M3 도구 사용). 결과는 Task 13 문서에 기록.

**Interfaces:**
- Consumes: `hexagon_ref_run --eval`, M3의 `nntr_causallm --eval`, `make_tokens.py`.
- Produces: 디바이스로 넘어가도 되는지에 대한 판정 + eval_short 토큰/텍스트 쌍(Task 8이 재사용).

- [ ] **Step 1: 평가 세트 고정 (컨트롤러 실행)**

```bash
python3 tools/hexagon/make_tokens.py "$QWEN3" /tmp/eval_long.txt \
    /tmp/eval_short.tokens.i32 --limit 128        # writes /tmp/eval_short.txt too
python3 tools/hexagon/make_tokens.py "$QWEN3" /tmp/eval_short.txt /tmp/roundtrip.i32
cmp /tmp/eval_short.tokens.i32 /tmp/roundtrip.i32 && echo "TOKENS ROUNDTRIP OK"
```

Expected: `TOKENS ROUNDTRIP OK`. 실패하면(디토크나이즈→재토크나이즈 불일치) `--limit`를 몇 토큰 줄여 경계가 깨지지 않는 지점을 찾는다 — 이후 모든 비교가 "같은 토큰 열"이라는 전제 위에 있다.

- [ ] **Step 2: ① ② 기준선 확보 (컨트롤러 실행)**

M3 수치는 eval_long 기준이므로, eval_short(128토큰)에서 ①②를 새로 잰다:

```bash
./build_x86/Applications/CausalLM/nntr_causallm /tmp/qwen3_fp32   --eval /tmp/eval_short.txt
./build_x86/Applications/CausalLM/nntr_causallm /tmp/qwen3_fq_bin --eval /tmp/eval_short.txt
```

(`/tmp/qwen3_fp32`, `/tmp/qwen3_fq_bin`은 M3 Task 5·6 산출물. 없으면 M3 계획 Task 6 Step 1의 명령으로 재생성한다 — `nntr_fakequant` → `nntr_quantize --fc_dtype FP32`.)

Expected: 두 줄의 `PPL <v> wall_ms <t>`.

- [ ] **Step 3: ②' 측정 (컨트롤러 실행)**

```bash
time ./build_x86_hexagon/hexagon_ref_run /tmp/qwen3_full \
    --tokens /tmp/eval_short.tokens.i32 --eval \
    --logits /tmp/ref_short.logits.f32 --steps 128
```

Expected: `PPL <v> wall_ms <t>`. 스칼라 구현이라 토큰당 1~3초(128토큰 = 수 분) 예상. 훨씬 느리면(>10분) 중단하고 `-O2`/`-march=native` 적용 여부를 먼저 확인한다.

- [ ] **Step 4: 게이트 판정** — 표로 보고(정확도+시간 병기 규칙):

| 기준선 | eval_short PPL | wall_ms | 델타 |
|---|---|---|---|
| ① 원본 fp32 | | | — |
| ② fake-quant fp32 | | | vs ① |
| ②' 참조 실행기 (w8a8) | | | vs ② |

**게이트: `(PPL②' − PPL②)/PPL② ≤ 3%`.** 참고로 M3의 eval_long ①vs② = +0.412%(19.7010 → 19.7821)를 표에 병기한다.

초과 시 **디바이스로 넘어가지 않는다.** 원인 후보를 이 순서로 좁힌다:
1. `hexagon_ref_run --dump-op`로 레이어 0의 op별 출력을 뽑아 NaN/폭주 확인 → lowering 오프셋 결함.
2. `--layers 1` 이미지의 은닉 상태를 numpy 참조(HF 가중치 fp32 재계산)와 대조 → packing 결함.
3. 위 둘이 깨끗하면 활성화 양자화 손실 자체 → 스펙대로 양자화 설계 재검토(활성화 per-token int8의 outlier, KV fp16 등).

- [ ] **Step 5:** 커밋 없음(측정 Task). 수치를 진행 원장에 기록.

---

### Task 6: 디버그 RPC 경로 + ABI v3

**Files:**
- Modify: `nntrainer/tensor/hexagon/htp/nntr_htp.idl` (`forward`에 `dsp_pcycles` 추가, `forward_debug` 신설)
- Modify: `nntrainer/tensor/hexagon/htp/nntr_htp_common.h` (`NNTR_HTP_ABI_VERSION` 2 → 3)
- Modify: `nntrainer/tensor/hexagon/htp/htp_graph.h`, `htp_graph.c` (`htp_graph_forward_upto`, 버퍼 해석 헬퍼, pcycles)
- Modify: `nntrainer/tensor/hexagon/htp/executor.c` (`nntr_htp_forward` 시그니처, `nntr_htp_forward_debug`)
- Modify: `nntrainer/tensor/hexagon/host/hexagon_runner.h`, `hexagon_runner.cpp` (`forward`에 pcycles out, `forward_debug`)
- Modify: `test/hexagon/hexagon_rpc_test.cpp` (M1 더미 왕복이 새 시그니처로 계속 통과)
- Modify: `test/hexagon/sim/test_graph.c` (부분 실행 + 인자 거부 케이스)
- Modify: `Applications/CausalLM/hexagon/qwen3_lowering.cpp` / `test/hexagon/test_oplist_header.c` — 버전 상수 참조가 있으면 갱신

**Interfaces:**
- Consumes: M2 실행기.
- Produces:

```c
/**
 * @brief Run ops [0, n_ops_limit) of one chunk. n_ops_limit >= n_ops runs the
 *        whole list (identical to htp_graph_forward).
 * @return 0 ok, non-zero on bad runtime arguments
 */
int htp_graph_forward_upto(struct htp_graph *g, const int32_t *tokens,
                           uint32_t n_tokens, uint32_t pos, float *logits,
                           uint32_t n_logits, uint32_t n_ops_limit,
                           uint64_t *pcycles);

/**
 * @brief Resolve a (buf id, offset, bytes) triple against the mapped buffers.
 * @return pointer, or 0 when out of bounds / not a mapped buffer
 */
const uint8_t *htp_graph_buf_ref(const struct htp_graph *g, uint32_t buf,
                                 uint32_t offset, uint32_t bytes);
```

호스트:

```cpp
  int forward(const int32_t *token_ids, uint32_t n_tokens, uint32_t pos,
              float *logits, uint32_t n_logits, uint64_t *dsp_pcycles = nullptr);
  int forward_debug(const int32_t *token_ids, uint32_t n_tokens, uint32_t pos,
                    uint32_t n_ops_limit, uint32_t dump_buf,
                    uint32_t dump_offset, uint8_t *dump, uint32_t dump_bytes,
                    uint64_t *dsp_pcycles = nullptr);
```

- [ ] **Step 1: 실패하는 테스트 작성** — `test/hexagon/sim/test_graph.c`에 실행기 레벨 계약을 추가한다(디바이스 없이 sim에서 도는 게이트):

```c
  /* Partial execution matches the reference executor at the same cut. */
  {
    const uint32_t cut = 9u;   /* right after layer 0's ATTN */
    uint64_t pc = 0;
    memset(kv, 0, KV_BYTES); memset(act, 0, act_size);
    CHECK(htp_graph_forward_upto(&g, tokens, 3u, 0u, logits, VOCAB, cut, &pc) == 0);
    memcpy(act_dsp, act, act_size);
    memset(kv, 0, KV_BYTES); memset(act, 0, act_size);
    ref_graph_forward_upto(oplist, weights, kv, act, tokens, 3u, 0u, ref_logits, cut);
    cmp_f16("attn out", (const __fp16 *)(act_dsp + ao_off),
            (const __fp16 *)(act + ao_off), 3u * QDIM, 1e-2f, 1e-2f);
    CHECK(pc > 0);            /* pcycles counter actually moved */
  }

  /* Out-of-range limits and dump refs are rejected, not clamped. */
  CHECK(htp_graph_forward_upto(&g, tokens, 3u, 0u, logits, VOCAB, N_OPS + 1u, &pc) != 0);
  CHECK(htp_graph_buf_ref(&g, NNTR_HTP_BUF_COUNT, 0u, 4u) == 0);
  CHECK(htp_graph_buf_ref(&g, NNTR_HTP_BUF_ACT, act_size - 64u, 128u) == 0);
```

- [ ] **Step 2: 테스트 실패 확인 (컨트롤러 실행)**

```bash
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.3.0.0/setup_sdk_env.source
./tools/hexagon/build_sim_test.sh
```

Expected: 컴파일 실패(`htp_graph_forward_upto` 미정의).

- [ ] **Step 3: 구현**

1. `nntr_htp_common.h`: `#define NNTR_HTP_ABI_VERSION 3u`. **와이어 구조체·validate 로직은 건드리지 않는다.**
2. `htp_graph.c`: 기존 `htp_graph_forward` 본문을 `htp_graph_forward_upto`로 옮기고, 런타임 인자 검사(`n_tokens`, `pos`) 뒤에 `if (n_ops_limit > g->cfg.n_ops) return -1;`를 추가. 루프는 `for (i = 0; i < n_ops_limit; ++i)`. 진입/종료에서 `HAP_perf_get_pcycles()` 차분을 `*pcycles`에 쓴다(널 허용). `htp_graph_forward`는 `n_ops_limit = g->cfg.n_ops`로 위임.
3. `htp_graph_buf_ref`: `bufs[]`와 `buf_size[]`를 그래프에 보관해 경계 검사 후 포인터 반환. TOKENS/LOGITS는 forward 인자라 매핑이 없으므로 `NNTR_HTP_BUF_WEIGHTS/KV/ACT`만 허용(그 외는 0).
4. `executor.c`: `nntr_htp_forward`에 `uint64 *dsp_pcycles` 추가, `nntr_htp_forward_debug` 신설 — 인자 검사 → `htp_graph_forward_upto` → `htp_graph_buf_ref(dump_buf, dump_offset, dumpLen)`로 얻은 포인터에서 `memcpy`. 실패는 `AEE_EBADPARM`. **M1 더미 경로(`n_ops == 0`)는 유지**(디바이스 브링업 안전망, M2 결정).
5. 호스트 `hexagon_runner`: 두 메서드 배선. `forward`의 `dsp_pcycles`는 기본 인자로 두어 기존 호출부(M1 테스트)를 깨지 않는다.

- [ ] **Step 4: 검증 (컨트롤러 실행)**

```bash
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.3.0.0/setup_sdk_env.source
./tools/hexagon/build_sim_test.sh && ./tools/hexagon/run_sim_test.sh graph
gcc -Wall -Werror -o /tmp/test_oplist test/hexagon/test_oplist_header.c && /tmp/test_oplist
./tools/hexagon/build_host_x86.sh test_lowering && ./build_x86_hexagon/test_lowering
HEX_ARCH=v79 ./tools/hexagon/build_skel.sh
./tools/hexagon/build_host_test.sh
```

Expected: `graph` PASS(신규 케이스 포함), 헤더/lowering 테스트 PASS(ABI 3 반영), skel·M1 테스트 바이너리 빌드 성공.

- [ ] **Step 5: M1 왕복 회귀 (컨트롤러 실행, 디바이스)**

```bash
./tools/hexagon/run_device_test.sh R3CY205ZMND
```

Expected: `RPC_TEST ... ok` 라인들 + exit 0. ABI 3 skel과 ABI 3 호스트가 핸드셰이크에 성공하는지 확인(옛 skel이 남아 있으면 여기서 버전 불일치로 걸린다 — 그게 이 Task에서 버전을 올린 이유다).

- [ ] **Step 6: 승인 후 커밋** — `[htp] Add the partial-execution debug RPC and DSP cycle reporting`

---

### Task 7: 디바이스 e2e 하네스 + 1레이어 브링업

**Files:**
- Create: `test/hexagon/hexagon_e2e_test.cpp`
- Create: `tools/hexagon/build_e2e_test.sh` (NDK 크로스빌드 — `build_host_test.sh` 패턴 복제, 소스 목록만 다름)
- Create: `tools/hexagon/run_e2e_test.sh` (push + 실행 + 로그 수집)

**Interfaces:**
- Consumes: `HexagonRunner`(Task 6 확장판), `RpcmemBuffer`, `read_hexcfg`, `lower_qwen3`, `.hexw`/`.hexcfg`/`.tokens.i32`.
- Produces: 디바이스 실행 파일. 출력 라인은 전부 `E2E ` 접두(로그 파서·이후 스크립트가 의존):
  - `E2E init ok weights=%llu kv=%llu act=%llu n_ops=%u`
  - `E2E step %u pos=%u pcycles=%llu us=%llu top1=%d logprob=%.6f`
  - `E2E ppl %f steps %u wall_ms %llu`
  - `E2E dump buf=%u off=%u bytes=%u -> %s`

CLI:

```
hexagon_e2e_test <prefix> --tokens <f> [--chunk N] [--eval] [--gen N]
                 [--logits <f> --steps N] [--dump-op i --dump-out <f>]
```

- [ ] **Step 1: 구현 — 하네스**

흐름(예외·에러는 즉시 `E2E FAIL ...` 출력 후 exit 1):

```cpp
  HexModelConfig cfg = read_hexcfg(prefix + ".hexcfg");
  HexLoweredGraph g = lower_qwen3(cfg);

  RpcmemBuffer weights(g.weights_size), kv(g.kv_size), act(g.act_size);
  CHECK(weights.valid() && kv.valid() && act.valid());
  read_file_into(prefix + ".hexw", weights.data(), g.weights_size);
  memset(kv.data(), 0, g.kv_size);

  auto runner = HexagonRunner::create();
  CHECK(runner && runner->init(g.oplist.data(), g.oplist.size(), weights, kv, act) == 0);
```

- prefill: `--chunk`(기본 `cfg.max_chunk`)로 토큰을 잘라 청크당 `forward` 1회.
- `--eval`: `n_tokens=1`로 `p = 0..n-2` 반복, 스텝별 `logprob(tokens[p+1])`·top-1·pcycles·wall(us) 출력, 마지막에 PPL.
- `--gen N`: prefill 후 greedy로 N토큰 디코드(생성 토큰 id를 그대로 출력 — 디토크나이즈는 호스트 스크립트 몫).
- `--logits`: 스텝별 로짓 벡터를 파일로.
- `--dump-op i --dump-out f`: `forward_debug(tokens, n, pos=0, i, dump_buf, dump_offset, ...)`. `dump_buf/off/bytes`는 `--dump-buf/--dump-off/--dump-bytes`로 받는다(참조 실행기가 `DUMP buf=… off=… bytes=…`로 알려준 값을 그대로 전달).
- 매 forward 전후로 `std::chrono::steady_clock`. `n_logits`는 `cfg.vocab`.

- [ ] **Step 2: 구현 — 빌드/실행 스크립트**

`build_e2e_test.sh`는 `build_host_test.sh`를 복제하되 소스를 바꾼다:

```bash
"$TC/aarch64-linux-android${API}-clang++" -std=c++17 -O2 -Wall -Werror \
    -static-libstdc++ "${INCS[@]}" -I "$REPO/Applications/CausalLM/hexagon" \
    "$HOST_DIR/rpcmem_allocator.cpp" "$HOST_DIR/hexagon_runner.cpp" \
    "$HOST_DIR/graph_lowering.cpp" \
    "$REPO/Applications/CausalLM/hexagon/qwen3_lowering.cpp" \
    "$REPO/Applications/CausalLM/hexagon/hex_image.cpp" \
    "$REPO/test/hexagon/hexagon_e2e_test.cpp" \
    "$OUT/host/nntr_htp_stub.o" -L "$CDSPRPC_DIR" -lcdsprpc \
    -o "$OUT/host/hexagon_e2e_test"
```

`run_e2e_test.sh <prefix> [adb-serial] -- <args...>`: skel·바이너리·`<prefix>.hexw`·`<prefix>.hexcfg`·토큰 파일을 `/data/local/tmp/nntr_htp/`로 push(이미 있고 크기가 같으면 건너뛴다 — 598MB 재전송 방지), FARF 파일 생성, 실행, stdout과 logcat을 `logs/hexagon/`에 저장. 결과 파일(`--logits`/`--dump-out`)은 `adb pull`로 회수한다.

- [ ] **Step 3: 검증 — 1레이어 브링업 (컨트롤러 실행)**

```bash
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.3.0.0/setup_sdk_env.source
HEX_ARCH=v79 ./tools/hexagon/build_skel.sh && ./tools/hexagon/build_e2e_test.sh
# 참조값(x86)
./build_x86_hexagon/hexagon_ref_run /tmp/qwen3_l1 --tokens /tmp/eval_short.tokens.i32 \
    --logits /tmp/ref_l1.logits.f32 --steps 8
# 디바이스
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_l1 R3CY205ZMND -- \
    --tokens /tmp/eval_short.tokens.i32 --logits /tmp/dsp_l1.logits.f32 --steps 8
python3 tools/hexagon/compare_logits.py /tmp/ref_l1.logits.f32 /tmp/dsp_l1.logits.f32
```

`tools/hexagon/compare_logits.py`(이 Step에서 함께 작성): 두 fp32 파일을 같은 길이로 읽어 스텝별 `max|Δ|`, `mean|Δ|`, `max rel`, top-1 일치 여부를 표로 출력하고, 혼합 판정식 `|Δ| ≤ atol + rtol·|ref|`(기본 atol=rtol=2e-2)을 전부 만족하면 `LOGITS OK`를 찍는다.

Expected: `E2E init ok ... n_ops=19`, 8스텝 실행, `LOGITS OK` + top-1 8/8 일치.

실패 시: (a) `E2E init` 실패면 버퍼 크기/rpcmem 할당(598MB 아님, 1레이어는 ~180MB) 또는 validate 코드 확인, (b) 값이 어긋나면 **Task 9로 가지 말고** `--dump-op`로 19개 op를 순차 비교(19개는 이진 탐색이 필요 없다).

- [ ] **Step 4: 승인 후 커밋** — `[htp] Add the on-device e2e harness running a packed qwen3 image`

---

### Task 8: 전체 모델 디바이스 정확도 게이트 (③ vs ②')

**Files:** 신규 코드 없음(Task 4·7 산출물). 결과는 Task 13 문서에 기록.

**Interfaces:**
- Consumes: `hexagon_e2e_test`, `hexagon_ref_run`, `compare_logits.py`, `/tmp/qwen3_full.*`.
- Produces: 스펙의 3-way 판정 확정 — 통과 시 M4 완료, 실패 시 Task 9 투입.

- [ ] **Step 1: 전체 모델 로짓 비교 (컨트롤러 실행)**

```bash
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY205ZMND -- \
    --tokens /tmp/eval_short.tokens.i32 --eval \
    --logits /tmp/dsp_short.logits.f32 --steps 128
python3 tools/hexagon/compare_logits.py /tmp/ref_short.logits.f32 \
    /tmp/dsp_short.logits.f32 --atol 2e-2 --rtol 2e-2
```

Expected: `E2E init ok ... n_ops=451`, `E2E ppl ...`, `LOGITS OK`.

- [ ] **Step 2: 장문 PPL + RPC 횟수 확인 (컨트롤러 실행)**

```bash
python3 tools/hexagon/make_tokens.py "$QWEN3" /tmp/eval_long.txt /tmp/eval_long.tokens.i32
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY205ZMND -- \
    --tokens /tmp/eval_long.tokens.i32 --eval
grep -c "^E2E step" logs/hexagon/e2e_*.log | tail -1
grep -c "nntr_htp: forward" logs/hexagon/device_farf_*.log | tail -1
```

Expected: `E2E ppl <v> steps 1060 wall_ms <t>`. 스텝 수와 FARF의 forward 로그 수가 같아야 한다 — **토큰당 RPC 정확히 1회**(스펙 요구). FARF에 forward 로그가 없으면 executor에 `FARF(LOW, ...)` 한 줄을 추가하는 대신, 스텝 수와 `E2E step` 라인 수 일치 + 하네스가 forward를 스텝당 1회만 호출한다는 코드 사실로 판정하고 그 근거를 원장에 남긴다.

- [ ] **Step 3: 게이트 판정** — 표로 보고:

| 비교 | 지표 | 측정값 | 기준 |
|---|---|---|---|
| ①vs② (M3, eval_long) | PPL 델타 | +0.412% | < 1% ✔ |
| ②vs②' (Task 5, eval_short) | PPL 델타 | | ≤ 3% |
| ②'vs③ (eval_short) | max·mean 로짓 오차 | | `≤ 2e-2 + 2e-2·|ref|` |
| ②'vs③ (eval_short) | top-1 일치율 | | ≥ 99.5% |
| ②'vs③ (eval_short) | PPL 델타 | | ≤ 0.5% |
| ②vs③ (eval_long) | PPL 델타 | | ≤ 3% |

시간 병기: 각 행에 `wall_ms`, 디바이스 행은 `Pcycles/step`, `forward_us/step` 중앙값.

전부 통과하면 **M4 정확도 게이트 통과**. 하나라도 초과하면 Task 9로 간다.

- [ ] **Step 4: 생성 품질 육안 확인 (컨트롤러 실행)**

```bash
python3 tools/hexagon/make_tokens.py "$QWEN3" /tmp/prompt.txt /tmp/prompt.tokens.i32
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY205ZMND -- \
    --tokens /tmp/prompt.tokens.i32 --chunk 128 --gen 64
python3 tools/hexagon/detokenize.py "$QWEN3" <생성된 id 목록>
```

`/tmp/prompt.txt`는 짧은 영어 지시문. `detokenize.py`는 `make_tokens.py`의 역함수(같은 Step에서 작성, 10줄). Expected: 문법적으로 온전한 영어 문단. 숫자 게이트가 아니라 "무언가 크게 잘못됐는지"의 최종 육안 확인이다.

- [ ] **Step 5:** 커밋 없음(측정 Task). 수치를 원장에 기록. `compare_logits.py`/`detokenize.py`는 Task 9 커밋에 함께 싣는다.

---

### Task 9: 발산 op 이진 탐색

**Files:**
- Create: `tools/hexagon/find_divergence.py`
- Create: `tools/hexagon/compare_logits.py`, `tools/hexagon/detokenize.py` (Task 7·8에서 작성, 여기서 커밋)

**Interfaces:**
- Consumes: `hexagon_ref_run --dump-op`, `hexagon_e2e_test --dump-op`, `run_e2e_test.sh`.
- Produces: 첫 발산 op의 인덱스·kind·레이어를 출력하는 스크립트. Task 8이 통과해도 **도구 자체는 검증한다**(스펙이 요구하는 장치이며, 이후 커널 변경 시 즉시 쓰인다).

- [ ] **Step 1: 구현** — `find_divergence.py`:

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Binary-search the first op where the DSP diverges from the x86 reference.

Both runners execute ops [0, i) for the same chunk at pos=0 and dump the
output tensor of op i-1; the predicate "outputs agree" is monotone in i, so
~log2(n_ops) comparisons (9 for 451 ops) locate the first bad op.

Usage: find_divergence.py <prefix> <tokens> [--serial S] [--chunk N]
                          [--atol 2e-2] [--rtol 2e-2]
"""
```

- `lo = 0`(항상 일치), `hi = n_ops`(불일치로 알려짐)에서 시작해 `mid`마다 두 러너를 돌려 비교. 매 호출은 `pos=0`으로 시작해 KV 상태를 오염시키지 않는다(디바이스 쪽은 `forward_debug`가 KV를 쓰지만, 각 비교가 `pos=0`이라 같은 구간을 덮어쓴다).
- 참조 러너가 stdout으로 알려주는 `DUMP buf/off/bytes`를 그대로 디바이스 인자로 전달한다.
- 종료 시 `FIRST_DIVERGENCE op=%d kind=%s layer=%d max_abs=%g max_rel=%g` 출력. 불일치가 없으면 `NO_DIVERGENCE`.
- `n_ops`와 op 메타데이터는 `hexagon_ref_run --list-ops`(참조 러너에 인덱스·kind·layer·out ref를 한 줄씩 찍는 플래그 추가, 15줄)로 얻는다.

- [ ] **Step 2: 도구 검증 (컨트롤러 실행)**

정상 상태에서 `NO_DIVERGENCE`가 나오는지, **그리고 인위적 결함을 넣었을 때 그 지점을 짚는지** 확인한다:

```bash
python3 tools/hexagon/find_divergence.py /tmp/qwen3_full /tmp/eval_short.tokens.i32 \
    --serial R3CY205ZMND
# 인위적 발산 주입: 레이어 5의 ffn_norm gamma를 이미지에서 한 원소만 바꿔 굽는다
python3 tools/hexagon/poison_image.py /tmp/qwen3_full /tmp/qwen3_poison --layer 5 --tensor ffn_norm
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_poison R3CY205ZMND -- --tokens ... # 이미지 push
python3 tools/hexagon/find_divergence.py /tmp/qwen3_poison /tmp/eval_short.tokens.i32 \
    --serial R3CY205ZMND --ref-prefix /tmp/qwen3_full
```

`poison_image.py`(이 Step에서 함께 작성, 20줄): `.hexcfg`를 읽어 `lower_qwen3`가 아는 오프셋 대신 — 파이썬에는 lowering이 없으므로 — `hexagon_ref_run --list-offsets`(woff를 한 줄씩 출력하는 플래그, 10줄)로 오프셋을 받아 해당 위치의 fp16 한 원소를 2배로 만든다.

Expected: 정상 이미지에서 `NO_DIVERGENCE`, 오염 이미지에서 `FIRST_DIVERGENCE op=<레이어 5의 ffn_norm op 인덱스=1+16*5+11=92>`. 비교 횟수가 9회 내외인지도 로그로 확인한다.

> Task 8이 실패해서 여기 온 경우: Step 2의 인위적 주입은 건너뛰고 실제 발산 op를 찾은 뒤, 해당 커널의 M2 sim 테스트에 그 형상·값을 재현하는 케이스를 추가해 sim에서 고친다(디바이스에서 디버깅하지 않는다). 수정 후 Task 8을 다시 돌린다.

- [ ] **Step 3: 승인 후 커밋** — `[htp] Add the divergence bisect and logit comparison tooling`

---

### Task 10: 성능 baseline 측정 (M5 시작)

**Files:** 신규 코드 없음. 결과는 Task 13 문서에.

**Interfaces:**
- Consumes: `hexagon_e2e_test`(pcycles·us 출력), Task 3의 `--max-chunk` 옵션.
- Produces: 튜닝 전 기준 수치 — 이후 Task 11·12의 개선 여부 판정 기준.

- [ ] **Step 1: 측정 (컨트롤러 실행)**

```bash
# decode: 1토큰/스텝 × 256스텝
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY205ZMND -- \
    --tokens /tmp/eval_long.tokens.i32 --eval --steps 256
# prefill: 128토큰 청크 × 8회
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY205ZMND -- \
    --tokens /tmp/eval_long.tokens.i32 --chunk 128 --gen 0
```

- [ ] **Step 2: baseline 표 작성** — 스텝별 `us`의 중앙값으로 계산(첫 스텝은 워밍업으로 제외):

| 항목 | 값 | 비고 |
|---|---|---|
| decode TPS | | `1e6 / median(us)` |
| decode Pcycles/token | | |
| prefill TPS | | `chunk / median(us)` |
| prefill Pcycles/token | | |
| 토큰당 RPC 횟수 | | =1 (Task 8 Step 2에서 확인) |
| 기존 HexKL decode TPS | 0.16 | 개요 문서 인용 |

**감각치(스펙):** decode는 토큰당 0.6GB 읽기 — 실효 대역폭 ~40GB/s 가정 시 이론 상한 ~60 TPS대. 측정값이 5 TPS 미만이면 대역폭이 아니라 다른 병목(동기화·워커 수·VTCM 미사용)이므로 Task 11·12에서 그 방향을 먼저 판다. 60 TPS를 넘으면 측정 방법을 의심한다(캐시된 가중치·잘못된 op-list).

- [ ] **Step 3:** 커밋 없음. 수치 기록.

---

### Task 11: HAP_power DCVS·클럭 설정

**Files:**
- Modify: `nntrainer/tensor/hexagon/htp/htp_graph.c` (init에서 전력 요청, destroy에서 해제)
- Modify: `nntrainer/tensor/hexagon/htp/htp_graph.h` (컨텍스트에 power client id 필드)

**Interfaces:**
- Consumes: `HAP_power.h`(SDK). 현재 코드베이스에는 전력 API 호출이 전혀 없다(확인됨) — DCVS가 기본 정책대로 동작해 클럭이 낮게 유지될 수 있다.
- Produces: init 시 1회 전력 요청, destroy 시 해제. 실패해도 실행은 계속한다(경고 FARF 후 진행 — 전력 정책은 최적화이지 정확성 요건이 아니다).

- [ ] **Step 1: 구현**

```c
  /* Power vote: DCVS on with a performance floor plus an explicit HVX/clock
   * request. Losing the vote is not fatal - the graph still runs, just at
   * whatever clock the default policy picks. */
  g->power_ctx = (void *)g;                /* unique client id per graph */
  HAP_power_request_t req;
  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_apptype;
  req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
  if (HAP_power_set(g->power_ctx, &req) != 0)
    FARF(ALWAYS, "nntr_htp: apptype vote failed");

  memset(&req, 0, sizeof(req));
  req.type = HAP_power_set_DCVS_v3;
  req.dcvs_v3.set_dcvs_enable = TRUE;
  req.dcvs_v3.dcvs_enable = TRUE;
  req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
  req.dcvs_v3.set_core_params = TRUE;
  req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_TURBO;
  req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
  req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_TURBO;
  req.dcvs_v3.set_sleep_disable = TRUE;
  req.dcvs_v3.sleep_disable = TRUE;
  if (HAP_power_set(g->power_ctx, &req) != 0)
    FARF(ALWAYS, "nntr_htp: dcvs vote failed");
```

`htp_graph_destroy`에서 `req.type = HAP_power_set_apptype; req.apptype = HAP_POWER_UNKNOWN;`으로 해제. 필드 이름은 SDK 6.3.0.0의 `HAP_power.h` 실물로 확인해 맞춘다(버전에 따라 `dcvs_v3` 멤버 구성이 다르다 — 컴파일 에러가 나면 헤더의 정의를 따르되 의미는 "TURBO 하한 + sleep 억제"로 동일하게 유지).

- [ ] **Step 2: 검증 (컨트롤러 실행)**

```bash
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.3.0.0/setup_sdk_env.source
HEX_ARCH=v79 ./tools/hexagon/build_skel.sh
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY205ZMND -- \
    --tokens /tmp/eval_long.tokens.i32 --eval --steps 256
```

Expected: Task 10과 같은 형식의 출력. 판정은 표로:

| 항목 | Task 10 baseline | HAP_power 적용 | 변화 |
|---|---|---|---|
| decode TPS | | | |
| decode Pcycles/token | | | |
| 스텝별 us 편차(p90/중앙값) | | | |

**Pcycles는 거의 그대로인데 us만 줄면** 클럭이 올라간 것(의도한 효과). 둘 다 그대로면 전력 요청이 반려됐거나 이미 최대 클럭이었던 것 — FARF 경고 유무로 구분한다.

- [ ] **Step 3: 정확도 회귀 (컨트롤러 실행)**

```bash
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY205ZMND -- \
    --tokens /tmp/eval_short.tokens.i32 --logits /tmp/dsp_pow.logits.f32 --steps 32
python3 tools/hexagon/compare_logits.py /tmp/ref_short.logits.f32 /tmp/dsp_pow.logits.f32
```

Expected: `LOGITS OK`(전력 설정은 수치에 영향이 없어야 한다).

- [ ] **Step 4: 승인 후 커밋** — `[htp] Vote for a performance clock corner at graph init`

---

### Task 12: DMA 더블 버퍼링 실측 검증 + prefill 청크 스윕

**Files:**
- Modify: `nntrainer/tensor/hexagon/htp/ops/hvx-matmul.c` (측정용 컴파일 스위치 1줄)

**Interfaces:**
- Consumes: 이미 구현된 VTCM/DMA 더블 버퍼 스트리밍 경로(`hvx-matmul.c:56-122`, ring cap 4, `buf[2]` 교대).
- Produces: 스펙이 요구하는 "DMA 더블 버퍼링 실측 검증" 근거 + 최적 prefill 청크 크기.

- [ ] **Step 1: 측정용 스위치 추가**

스트리밍 경로 진입 함수 맨 앞에 한 줄만 넣는다(기본 빌드는 영향 없음):

```c
#ifdef HTP_MM_NO_VTCM
  return false; /* measurement-only: forces the direct DDR read path */
#endif
```

- [ ] **Step 2: DMA 효과 측정 (컨트롤러 실행)**

```bash
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.3.0.0/setup_sdk_env.source
HEX_ARCH=v79 ./tools/hexagon/build_skel.sh                       # VTCM/DMA on
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY205ZMND -- \
    --tokens /tmp/eval_long.tokens.i32 --eval --steps 128
HEX_ARCH=v79 HEX_EXTRA_CFLAGS=-DHTP_MM_NO_VTCM ./tools/hexagon/build_skel.sh
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY205ZMND -- \
    --tokens /tmp/eval_long.tokens.i32 --eval --steps 128
HEX_ARCH=v79 ./tools/hexagon/build_skel.sh                       # 원복
```

`build_skel.sh`에 `HEX_EXTRA_CFLAGS`(기본 빈 값)를 컴파일 인자로 넘기는 한 줄을 추가한다.

| 경로 | decode TPS | Pcycles/token | prefill TPS |
|---|---|---|---|
| VTCM/DMA 더블 버퍼 (기본) | | | |
| 직접 DDR 읽기 (`HTP_MM_NO_VTCM`) | | | |

판정: decode는 대역폭 병목이라 차이가 작을 수 있고(가중치를 한 번씩만 읽으므로 중첩 이득이 제한적), **prefill에서 유의미한 차이**가 나는 것이 정상이다. 더블 버퍼 쪽이 더 느리면 ring depth·타일 크기가 잘못 잡힌 것이므로 원인을 기록하고 후속 과제로 남긴다(범위 외 최적화 착수 금지).

- [ ] **Step 3: prefill 청크 스윕 (컨트롤러 실행)**

```bash
for C in 32 64 128 256; do
  ./build_x86_hexagon/nntr_hexpack "$QWEN3/w8cx"/*.bin /tmp/qwen3_c$C --max-chunk $C
  ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_c$C R3CY205ZMND -- \
      --tokens /tmp/eval_long.tokens.i32 --chunk $C --gen 0
done
```

| max_chunk | prefill TPS | Pcycles/token | ACT 크기 | 비고 |
|---|---|---|---|---|
| 32 | | | | |
| 64 | | | | |
| 128 | | | | 기본값 |
| 256 | | | | |

최고 TPS 지점을 기본값으로 삼을지 판단해 `nntr_hexpack`의 기본 `--max-chunk`를 갱신한다(변경 시 Task 8의 정확도 비교를 그 이미지로 1회 재확인).

- [ ] **Step 4: 승인 후 커밋** — `[htp] Add the VTCM bypass switch used to measure DMA overlap`

---

### Task 13: engine="htp" 통합

**Files:**
- Create: `nntrainer/tensor/hexagon/host/hexagon_backend.h`, `hexagon_backend.cpp` (세션 lifecycle — 개요 문서의 디렉토리 설계에 있으나 아직 없음)
- Modify: `nntrainer/tensor/hexagon/meson.build`
- Modify: `Applications/CausalLM/models/causal_lm.cpp` / `transformer.cpp` (htp 경로 분기 — 실제 배선 지점은 구현 시 확정)
- Modify: `Applications/CausalLM/hexagon/meson.build`
- Modify: `docs/backend_guide/HEXAGON.md`

**Interfaces:**
- Consumes: 검증이 끝난 `HexagonRunner`, `Qwen3W8cxBin`, `lower_qwen3`, `pack_weights`.
- Produces: 앱이 `engine="htp"`로 초기화되면 CausalLM의 forward 전체가 DSP에서 돌고, 실패 시 CPU 폴백.

**설계 원칙(개요 문서 §통합 지점, §부분 오프로드 금지):** 부분 오프로드는 하지 않는다. htp 경로는 "전 그래프 DSP" 아니면 "전부 CPU"다. 초기화 실패·예상 밖 그래프는 경고 후 CPU 경로로 떨어진다.

- [ ] **Step 1: 배선 설계 확정 (구현자 조사 → 사용자 승인)**

구현 전에 다음을 코드로 확인해 한 문단으로 보고한다: (a) `engine="htp"`가 현재 어디까지 인식되는지(`nntrainer/engine.cpp`, `docs/backend_guide/HEXAGON.md` §6), (b) CausalLM의 forward 진입점 중 대체 지점(`incremental_inference` 호출부 vs `CausalLM::run`), (c) W8_CX .bin을 앱이 로드하는 현재 경로와 `Qwen3W8cxBin`을 끼우는 위치. **설계 승인 후에만 구현으로 넘어간다** — 여기서 갈리는 선택지가 크고, 잘못 잡으면 되돌리는 비용이 크다.

- [ ] **Step 2: 구현 — `HexagonBackend` 세션**

```cpp
namespace nntrainer::hexagon {
/**
 * @class HexagonBackend
 * @brief Owns the rpcmem buffers, the packed weight image and the runner for
 *        one model. create() returning nullptr means "run on CPU".
 */
class HexagonBackend {
public:
  static std::unique_ptr<HexagonBackend>
  create(const std::string &w8cx_bin, const HexModelConfig &cfg);

  /** @return 0 on success; logits holds cfg.vocab floats of the last token. */
  int forward(const int32_t *tokens, uint32_t n_tokens, uint32_t pos,
              float *logits);
  const HexModelConfig &config() const { return cfg_; }

private:
  HexModelConfig cfg_;
  HexLoweredGraph graph_;
  std::unique_ptr<RpcmemBuffer> weights_, kv_, act_;
  std::unique_ptr<HexagonRunner> runner_;
};
} // namespace nntrainer::hexagon
```

`create()`는 이미지를 rpcmem에 **직접 패킹**한다(파일 경유 없음): `Qwen3W8cxBin` → `pack_weights(g, cfg, w, (uint8_t *)weights_->data())` → `runner_->init(...)`. 어느 단계든 실패하면 `nullptr` 반환 + `fprintf(stderr, ...)`.

- [ ] **Step 3: 구현 — 앱 배선**

Step 1에서 승인된 지점에 분기를 넣는다. 규약: prefill은 `max_chunk` 단위, decode는 `n_tokens=1`, `pos`는 앱의 KV 위치와 동일. 샘플링은 그대로 CPU.

- [ ] **Step 4: 검증 (컨트롤러 실행)**

```bash
# 안드로이드 빌드 (기존 스크립트)
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.3.0.0/setup_sdk_env.source
./Applications/CausalLM/build_android.sh          # 실패 시 출력 기반으로 대응
./Applications/CausalLM/install_android.sh
adb -s R3CY205ZMND shell "cd /data/local/tmp/... && ./nntr_causallm <model-dir>"
```

Expected: 앱이 DSP 경로로 초기화(로그에 `hexagon: ...` 성공 라인)되고, 프롬프트에 대해 Task 8 Step 4와 같은 품질의 문장을 생성. TPS를 Task 10·11 하네스 수치와 비교해 앱 오버헤드(토크나이저·샘플링·KV 관리)를 표로 보고한다.

- [ ] **Step 5: CPU 폴백 확인 (컨트롤러 실행)**

skel을 일부러 치워 DSP 초기화를 실패시킨다:

```bash
adb -s R3CY205ZMND shell "mv <dev-dir>/libnntr_htp_skel.so <dev-dir>/skel.bak"
adb -s R3CY205ZMND shell "cd ... && ./nntr_causallm <model-dir>"
adb -s R3CY205ZMND shell "mv <dev-dir>/skel.bak <dev-dir>/libnntr_htp_skel.so"
```

Expected: 경고 출력 후 **CPU로 전체 실행**(크래시 없음). 이것이 개요 문서가 말한 "안전망으로서의 폴백"이다.

- [ ] **Step 6: 승인 후 커밋** — `[htp] Wire the DSP graph runner into the engine="htp" path`

---

### Task 14: 문서 갱신 + 전체 회귀 + 핸드오프

**Files:**
- Modify: `docs/backend_guide/HEXAGON.md`
- Create: `docs/superpowers/dones/04-e2e-verification.md` (커밋하지 않음 — 원장용)

**Interfaces:**
- Consumes: Task 1–13 전부.

- [ ] **Step 1: 문서 갱신** — `HEXAGON.md`에 추가:
  - "e2e pipeline (M4)": `.bin` → `nntr_hexpack` → `.hexw`/`.hexcfg` → 하네스/앱 흐름도, 파일 포맷, CLI 사용법.
  - "Accuracy baselines": 4-way 표(①②②'③)와 Task 5·8의 실측 수치(정확도+시간 병기), 게이트 기준과 근거.
  - "Debugging": `forward_debug` 규약, `find_divergence.py` 사용법(451 op → ~9회 비교).
  - "Performance (M5)": Task 10·11·12 표, HAP_power 정책, 권장 `max_chunk`, 이론 상한 대비 위치.
  - "Current status": M5 완료 + engine="htp" 통합 완료, 후속 범위 외 항목(HMX matmul, Q4 가중치, int8 KV, M>1 decode)을 개요 문서 인용으로 명시.
  - ABI v3 표기(v2 → v3 변경 사유: 디버그 RPC + pcycles).

- [ ] **Step 2: 전체 회귀 (컨트롤러 실행)**

```bash
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.3.0.0/setup_sdk_env.source
# sim 커널 회귀 (M2 전체)
./tools/hexagon/build_sim_test.sh
for t in smoke pool exp quant matmul matmul_dma rmsnorm rope eltwise embed attn logits graph; do
  ./tools/hexagon/run_sim_test.sh $t || echo "FAILED: $t"
done
# x86 호스트 회귀
./tools/hexagon/build_host_x86.sh
./build_x86_hexagon/test_lowering
./build_x86_hexagon/test_w8cx_bin "$QWEN3/w8cx"/*.bin
gcc -Wall -Werror -o /tmp/test_oplist test/hexagon/test_oplist_header.c && /tmp/test_oplist
# 디바이스 회귀
HEX_ARCH=v79 ./tools/hexagon/build_skel.sh && ./tools/hexagon/build_host_test.sh
./tools/hexagon/run_device_test.sh R3CY205ZMND
./tools/hexagon/build_e2e_test.sh
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY205ZMND -- \
    --tokens /tmp/eval_short.tokens.i32 --logits /tmp/dsp_final.logits.f32 --steps 32
python3 tools/hexagon/compare_logits.py /tmp/ref_short.logits.f32 /tmp/dsp_final.logits.f32
# nntrainer 본체 회귀 (x86)
ninja -C build_x86
```

Expected: 전부 PASS/exit 0, `LOGITS OK`.

- [ ] **Step 3: 핸드오프 작성** — `docs/superpowers/handoffs/`에 M2·M3 핸드오프와 같은 형식으로: 진행 표, 사용자 계약, 명명 규칙, 이번에 밟은 툴체인 함정, 검증 절차, 이연 항목(전 Task의 Minor 리뷰 지적), 최종 수치 표.

- [ ] **Step 4: 승인 후 커밋** — `[docs] Update the backend guide for the M4-5 e2e pipeline`

---

## Self-Review 결과

**1. 스펙 커버리지 (`04-e2e-verification.md`)**

| 스펙 요구 | Task |
|---|---|
| ① 원본 fp32 기준선 | M3 측정 인용 + Task 5 Step 2(eval_short 재측정) |
| ② fake-quant fp32 기준선 | M3 측정 인용 + Task 5 Step 2 |
| ③ DSP 실행 | Task 7(1레이어) → Task 8(전체) |
| ①vs② = 순수 양자화 손실, <1% 게이트 | M3 통과(+0.412%), Task 8 Step 3 표에 병기 |
| ②vs③ = 커널 구현 오차, 작은 허용 오차 | Task 5(②vs②')·Task 8(②'vs③)로 분해 — 스펙의 단일 비교를 두 단계로 나눈 정제(공통 설계 §4-way) |
| 지표: 토큰별 logits 상대 오차·top-1 일치율·소규모 perplexity | Task 8 Step 3 표 전 항목 |
| 발산 국소화: op 인덱스까지 실행 + IO 버퍼 반환 + 이진 탐색(~9회) | Task 6(RPC·실행기), Task 9(스크립트·검증) |
| 디바이스 테스트(8 Elite): 3-way 하니스 + 이진 탐색 | Task 7·8·9, S25 Ultra |
| 성능: prefill/decode TPS, 토큰당 RPC=1 | Task 8 Step 2, Task 10 |
| DMA 더블 버퍼링 실측 검증 | Task 12 Step 2 |
| HAP_power DCVS·클럭 | Task 11 |
| prefill 청크 크기 스윕(예: 128) | Task 12 Step 3 |
| decode 상한 돌파는 범위 외 | Task 14 Step 1 문서에 명시 |
| (개요 문서) engine="htp" 통합 지점, 부분 오프로드 금지, CPU 폴백은 안전망 | Task 13 (Step 5에서 폴백 실증) |

**2. 의도적 범위 결정 (스펙과의 차이 — 구현자 주의)**

1. **3-way → 4-way.** ②(가중치만 양자화, fp32 활성화)와 ③ 사이에 활성화 양자화·fp16이라는 두 번째 변수가 있어, 스펙 그대로면 "②vs③ 초과 = 커널 버그" 판정이 성립하지 않는다. 같은 packed 이미지를 해석하는 x86 참조 실행기 ②'를 끼워 두 원인을 분리했다. 부수 효과로 **lowering·packing·리더 결함이 디바이스 이전(Task 5)에 걸린다.**
2. **op-list 파일을 만들지 않는다.** 소비자가 `.hexcfg`로 `lower_qwen3`를 직접 호출하므로 이미지·op-list 불일치가 구조적으로 불가능하다(파일 하나 감소).
3. **디바이스 경로는 독립 크로스빌드 하네스 우선, engine 통합은 마지막**(사용자 결정). 정확도 게이트를 앱 변수 없이 통과시킨 뒤 배선한다.
4. **성능 튜닝(Task 10–12)이 통합(Task 13)보다 앞선다.** 튜닝 대상이 전부 DSP 측이라 하네스 측정 루프가 훨씬 빠르다.
5. **ABI v2 → v3.** op-list 와이어 포맷은 그대로지만 IDL이 바뀌므로, 디바이스에 남은 옛 skel을 init 핸드셰이크에서 잡기 위해 올린다.
6. **PPL 게이트 텍스트가 둘이다.** ②'가 스칼라 구현이라 1061토큰이 비현실적(수 시간) → 정밀 비교는 eval_short(128토큰), 모델 수준 PPL은 eval_long(1061토큰, 디바이스만). 토큰 열 동일성은 Task 5 Step 1의 왕복 검사로 보장한다.
7. **`--layers N`(1레이어 이미지)** 은 스펙의 "디바이스 1레이어 → 전체 모델"을 구현하는 수단이며, 리더가 항상 28레이어 체크포인트를 읽고 packing 단계에서만 자른다.
8. **Task 1은 hvx_m3의 결함 수정을 포함한다** — 원래 M3 범위지만 M3 진행 원장이 "M4 어댑터에서 축 불일치가 드러나면 재양자화"로 이연해 둔 항목이고, 파일 크기 산술로 실재가 확정됐다(597,626,880 vs 598,230,528). ②와 ③의 임베딩 값이 달라지므로 M4의 전제 조건이다.

**3. 타입·명칭 일관성 확인**

- `HexModelConfig`/`HexModelWeights`/`HexLayerWeights`/`HexWeightOffsets`/`HexLoweredGraph`/`lower_qwen3`/`pack_weights`/`align128` — M3 산출물 그대로, 이 계획에서 시그니처 변경 없음.
- `Qwen3W8cxBin::{weights,file_size,expected_size}` — Task 2 정의, Task 3·13 소비. 동일 철자.
- `read_hexcfg`/`write_hexcfg` — Task 3 정의, Task 4·7·9 소비.
- `ref_graph_forward_upto(oplist, weights, kv, act, tokens, n_tokens, pos, logits, n_ops_limit)` — Task 4 정의, Task 6 sim 테스트·Task 9 소비. 인자 순서 동일.
- `htp_graph_forward_upto(g, tokens, n_tokens, pos, logits, n_logits, n_ops_limit, pcycles)` / `htp_graph_buf_ref(g, buf, offset, bytes)` — Task 6 정의, executor·sim 테스트가 동일 시그니처로 호출.
- `HexagonRunner::forward(..., uint64_t *dsp_pcycles = nullptr)` / `forward_debug(...)` — Task 6 정의, Task 7·9·13 소비.
- 출력 접두: 참조 실행기·하네스 모두 `PPL <v> wall_ms <t>`(M3 `--eval`과 동일 형식), 하네스는 추가로 `E2E ` 접두. `compare_logits.py`는 `LOGITS OK`, `find_divergence.py`는 `FIRST_DIVERGENCE`/`NO_DIVERGENCE`.

**4. 알려진 리스크**

1. **.bin 텐서 순서**가 그래프 생성 순서와 다를 가능성(형제 레이어의 정렬 규칙). 크기 검사는 같은 크기 텐서의 뒤바뀜(wk↔wv, up↔gate)을 못 잡으므로 **Task 2 Step 6의 safetensors 대조가 진짜 게이트**다. 어긋나면 순서 표만 고치면 된다(리더 국소 수정).
2. **rpcmem 단일 598MB 할당**이 디바이스 dma-buf 힙 정책에 걸릴 수 있다. Task 7 Step 3을 1레이어(~180MB)로 먼저 하는 이유가 이것이다. 실패 시 대응은 WEIGHTS를 여러 버퍼로 쪼개는 것이 아니라(와이어 포맷이 buf id 5개 고정) 힙 한도 확인 → `--layers`를 늘려가며 한계 지점 특정 → 필요하면 `max_seq`를 줄여 KV를 축소.
3. **②' 실행 시간.** 스칼라 fp16 x86에서 토큰당 1~3초 예상. 10분/128토큰을 크게 넘으면 Task 5를 멈추고 최적화 여부를 사용자와 결정한다(범위 외 작업 착수 금지).
4. **`HAP_power_request_t`의 `dcvs_v3` 필드 구성**이 SDK 6.3.0.0에서 다를 수 있다. 컴파일 에러 시 헤더 실물을 따르되 의미(TURBO 하한·sleep 억제)는 유지한다.
5. **`forward_debug`의 KV 오염.** 부분 실행도 KV를 쓰므로 모든 디버그 비교는 `pos=0`에서 시작해야 한다 — Task 9 스크립트가 이 규약을 지킨다. 어기면 이진 탐색의 단조성이 깨져 엉뚱한 op를 지목한다.
6. **Task 13 배선 지점**이 코드 조사 전에는 확정 불가라 Step 1을 승인 게이트로 두었다. 여기서 큰 편차가 나오면 Task 13만 별도 계획으로 분리하는 것도 정당한 선택이다.
7. **M3 원장이 남긴 계획 결함 전례**(추정 수치가 실측과 어긋난 사례 4건: fq_tensors 197→198, weights_size 655MB→598MB, suffix `_w8_cx`→`_w8cx`, save_with_dtype 36→34)를 감안해, 이 계획의 기대 수치 중 **실측 기반이 아닌 것**은 다음 셋뿐임을 명시한다: `.hexw` 크기(Task 3 Step 4에서 도구 출력으로 고정), ②' 실행 시간(1~3초/토큰 추정), 정확도 허용 오차 임계값(2e-2·3%·0.5% — 판단값). 어긋나면 코드 결함이 아니라 계획의 추정 오차일 수 있으므로, 재계산 근거를 원장에 남기고 기준을 비준한다.

**5. M3 원장에서 승계한 운영 규칙**

- 검증 실행은 컨트롤러, 커밋은 사용자 승인(위 Global Constraints에 반영).
- 각 Task는 구현 서브에이전트(ponytail full) → 리뷰 서브에이전트 → Critical/Important 픽스 후 재리뷰 → 컨트롤러 검증 → 승인 커밋. 리뷰 라운드는 최대 5회.
- 계획과 실제가 어긋나면 **Ruling**(판정 + 근거 + 틀릴 경우 비용)을 원장에 기록하고 진행한다.
- Minor 지적은 보류하고 Task 14에서 일괄 triage한다(M2 이연 항목 7건 + M3 이연 항목 포함).
- 브리프/리포트/리뷰 diff는 `.superpowers/sdd/2026-08-20-hexagon-hvx-m4-m5-e2e/`에 둔다(`git clean -fdx`로 소실되므로 핸드오프 문서와 git log로 복구 가능하게 유지).
