# Qualcomm AI Hub: HuggingFace 모델 → 양자화 → QNN Binary 파이프라인

> 조사일: 2026-08-15. nntrainer HTP/QNN 백엔드 작업(Qwen3-0.6B on-device) 참고용.

## 개요

Qualcomm AI Hub는 HF 모델을 양자화하고 클라우드 실기기에서 QNN context binary로
컴파일해주는 파이프라인을 제공한다. 두 가지 레벨이 있다.

## 1. qai-hub-models (모델별 레시피, LLM용)

레포: https://github.com/qualcomm/ai-hub-models

```bash
# 1. AIMET 기반 양자화 (HF repo명 또는 로컬 safetensors 체크포인트)
python -m qai_hub_models.models.<모델명>.quantize \
    --checkpoint <HF-repo-or-local-path> -o ./quantized_model

# 2. 정확도 평가 (wikitext PPL 등)
python -m qai_hub_models.models.<모델명>.evaluate --checkpoint ./quantized_model

# 3. 클라우드 컴파일 + 실기기 프로파일링
python -m qai_hub_models.models.<모델명>.export \
    --checkpoint ./quantized_model --device "Snapdragon 8 Elite QRD"
```

- LLM 기본 레시피: **w4a16** (4-bit weight, 16-bit activation). 옵션으로
  Sequential MSE (정확도↑, 5시간+ 소요).
- 커스텀 파인튜닝 체크포인트 지원. 단 **아키텍처 변경 불가** — 지원 모델 목록 내
  구조만 가능.
- **Qwen 지원**: Qwen3-0.6B / 1.7B / 4B / 8B, Qwen2.5-VL, Qwen3-VL 등.
- 일부 모델(Qwen3-4B 등)은 https://huggingface.co/qualcomm 에 기기별
  프리컴파일 바이너리가 이미 업로드되어 있음 (Snapdragon 8 Elite / 8 Elite Gen 5 /
  X Elite / X2 Elite 등).
- 최신 CLI: `qai-hub-models export qwen3_4b_instruct_2507 --target-runtime
  geniex_qairt --precision w4a16 --device "..."`. 신규 런타임
  [GenieX](https://github.com/qualcomm/geniex)도 등장.
- LLM 출력물은 **Genie bundle** 형태:
  https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie

## 2. qai-hub 로우레벨 API (임의 ONNX 모델)

- `submit_quantize_job(onnx모델, 캘리브레이션 데이터)` → QDQ 포맷 양자화 ONNX.
  캘리브레이션은 500–1000 샘플 권장.
- dtype: w8a8, w8a16 (QNN 타겟은 int8 weight / int8·int16 activation).
- 이어서 `submit_compile_job(..., target_runtime=qnn_context_binary,
  quantize_io)` → QNN context binary.
- context binary는 **OS 무관, SoC 종속, NPU 전용**.
- 클라우드 실기기에서 컴파일/프로파일링 (무료 계정 + API 토큰 필요).

문서:
- https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html
- https://workbench.aihub.qualcomm.com/docs/hub/compile_examples.html
- https://github.com/qualcomm/ai-hub-models/blob/main/tutorials/llm/quantize_llama3.md

## Genie bundle ↔ nntrainer QNN 호환성

**Genie bundle은 패키징일 뿐, 내부의 `*.bin`은 표준 QNN context binary다.**
nntrainer QNN 백엔드는 `QNNContext::load()` → `makeContext()` →
`contextCreateFromBinary`로 context binary를 로드하므로
(`nntrainer/qnn/jni/qnn_context_var.h:210`, `:313`) 원칙적으로 그대로 사용 가능.

번들 구성:
- `*.bin` — QNN context binary 여러 개 (모델이 3~5개 파트로 split)
- `genie_config.json` — Genie 런타임 전용 설정
- `tokenizer.json`, HTP backend extension config

### nntrainer에서 직접 쓰려면 Genie가 하던 일을 호스트에서 구현해야 함

1. **멀티파트 로딩** — 파트 간 weight sharing, 순서대로 실행하는 오케스트레이션
2. **prefill/decode 그래프 분리** — binary 안에 seq_len 다른 그래프 복수 존재
   (예: prefill 128, decode 1), 그래프 선택 로직 필요
3. **KV cache I/O 관리** — KV cache가 그래프 입출력 텐서로 노출됨,
   shift/update는 호스트 처리
4. **quantized I/O** — 입출력 int quantize, scale/offset 변환 필요
5. **RoPE position 입력, 샘플링, 토크나이저** — 전부 호스트 몫

### 호환성 제약

- context binary는 SoC 종속 (8 Elite용은 8 Elite에서만 동작)
- 컴파일에 쓰인 QNN SDK 버전 ↔ 디바이스 QNN 라이브러리 버전 호환 필요

## Qwen3-0.6B Genie bundle 뽑는 실제 절차

레퍼런스: [ai-hub-models v0.60.0 qwen3_0_6b README](https://github.com/qualcomm/ai-hub-models/blob/v0.60.0/src/qai_hub_models/models/qwen3_0_6b/README.md),
[llm_on_genie 튜토리얼](https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie)

### 경로 A: 프리컴파일 에셋 다운로드 (가장 빠름, 클라우드 컴파일 불필요)

Qwen3-0.6B는 8 Elite / 8 Elite Gen 5 / X Elite / X2 Elite / SA8775P 등용
w4a16 프리컴파일 에셋이 이미 있음.

```bash
pip install qai_hub_models_cli   # 또는 qai-hub-models 패키지에 포함

qai-hub-models info Qwen3-0.6B          # 다운로드 옵션 확인
qai-hub-models perf Qwen3-0.6B          # 성능 수치 확인
qai-hub-models fetch Qwen3-0.6B --runtime genie --precision w4a16
# (GenieX용은 --runtime geniex_qairt)
```

HF에서 직접도 가능: https://huggingface.co/qualcomm/Qwen3-0.6B 에서 기기별 zip.
**주의: 에셋에 표기된 QAIRT SDK 버전과 디바이스의 QAIRT 라이브러리 버전이 맞아야 함.**

### 경로 B: 직접 export (클라우드 컴파일)

```bash
# 1. 환경 (Python 3.10 ~ 3.13)
python3.10 -m venv venv && source venv/bin/activate
pip install "qai-hub-models[qwen3-0-6b]"

# 2. AI Hub Workbench API 토큰 설정
#    workbench.aihub.qualcomm.com 로그인 → Account → Settings → API Token
qai-hub configure --api_token <API_TOKEN>

# 3-a. Qualcomm이 이미 양자화해둔 체크포인트로 export (양자화 스킵, GPU 불필요)
qai-hub-models export qwen3_0_6b --checkpoint DEFAULT_W4A16 \
    --target-runtime genie --device "Samsung Galaxy S25 (Family)"
# --checkpoint: DEFAULT | DEFAULT_W4A16 | DEFAULT_Q4_0

# 3-b. 커스텀 체크포인트를 직접 양자화 후 export (CUDA GPU 필요, 3B는 40GB VRAM 권장)
python -m qai_hub_models.models.qwen3_0_6b.quantize \
    --precision w4a16 --output-dir ./quantized_checkpoint
qai-hub-models export qwen3_0_6b --checkpoint ./quantized_checkpoint \
    --target-runtime genie --device "Samsung Galaxy S25 (Family)"
```

컴파일/프로파일링은 Qualcomm 클라우드 실기기에서 수행됨. 출력물:
`*.bin` (QNN context binary), `genie_config.json`, tokenizer, HTP config.

### 디바이스에서 실행 (Android)

전제: QAIRT SDK v2.29.0+, Hexagon v73+ (8 Elite는 v79 → `hexagon-v79` 경로 사용).

```bash
adb push genie_bundle /data/local/tmp
adb push ${QAIRT_HOME}/lib/aarch64-android/*.so /data/local/tmp/genie_bundle
adb push ${QAIRT_HOME}/lib/hexagon-v79/unsigned/*.so /data/local/tmp/genie_bundle
adb shell
cd /data/local/tmp/genie_bundle
export LD_LIBRARY_PATH=$PWD ADSP_LIBRARY_PATH=$PWD
./genie-t2t-run -c genie_config.json \
  -p "<|im_start|>user"$'\n'"What is France's capital?<|im_end|>"$'\n'"<|im_start|>assistant"$'\n'
# 성능 측정: --profile perf.txt
```

### 참고 수치 (HF 모델 카드, Qwen3-0.6B @ Snapdragon 8 Elite)

- GENIEX_QAIRT w4a16, ctx 4096: **~119.7 TPS**, TTFT 0.017–0.556s
- GENIEX_LLAMACPP q4_0, ctx 512: ~101 TPS

→ 현재 nntrainer HTP 경로(~0.16 TPS) 대비 비교 기준선.

### 주의

- **Genie 런타임은 곧 deprecated 예정** — 신규는 GenieX(`geniex_qairt`)가 기본.
  QNN context binary 자체가 필요하면 genie/geniex_qairt 번들 안의 `.bin`을 쓰면 됨.
- v0.60.0부터 레포 레이아웃이 `src/qai_hub_models/...`로 변경됨.

## HTP 병목 관점 시사점

현재 HTP 경로는 op 단위 FastRPC 호출이 per-token 병목 (Qwen3-0.6B ~0.16 TPS).
context binary 실행은 그래프 전체가 NPU 상주 + 토큰당 `graphExecute` 파트 수만큼
호출이라 RPC 횟수 차원이 다름. Qualcomm 공식 Qwen3-0.6B Genie 파이프라인 결과가
성능 비교 기준선(baseline)으로 유용.
