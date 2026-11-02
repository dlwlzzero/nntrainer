# #4456 / #4451 리팩토링 handoff (2026-11-02 기준)

이 문서는 `Applications/CausalLM/layers/mha_core.cpp`의 백엔드 분기를 `ComputeOps` whole-op 뒤로 옮기는 작업(#4451, umbrella #4456)을 이어받는 사람을 위한 문서다. 현재 열린 PR, 머지 후 할 일, 남은 단계, 작업 규칙과 로컬 검증 방법을 정리한다. PR별 상세 내용은 같은 폴더의 `PR_<번호>.md`에 있다.

## 1. 목표

- `MHACoreLayer::one_batch_incremental_forwarding`이 `tensor.getOps()`를 통해 Q RoPE → `kv_cache_write` → `attention`만 호출하게 만든다.
- 레이어에서 `#if ENABLE_OPENCL` / `ENABLE_CUDA`, `getenv`, 백엔드 include를 없앤다.
- 어떤 백엔드를 쓸지는 context의 op 테이블이 정하고, 그 안에서 어떤 커널을 쓸지는 op가 정한다.
- 설계 전체(인터페이스 방향, 백엔드 규칙 4가지, 단계 6a~9, 메인테이너 질문 Q1~Q5)는 #4466 본문의 "Design of the remaining `mha_core.cpp` steps" 절에 있다.

## 2. 열린 PR 스택

모든 PR의 head 브랜치는 fork `dlwlzzero/nntrainer`에 있고, base는 `nntrainer/nntrainer:main`이다. 저장소는 **rebase-merge만 허용**하므로 PR에 merge 커밋이 있으면 머지할 수 없다. 각 PR은 아래 표의 "쌓인 위치" 위에 일렬로 쌓여 있다.

| PR | 내용 | head 브랜치 (fork) | head | 쌓인 위치 | 상태 |
|---|---|---|---|---|---|
| #4462 | `unittest_mha_core` 테스트 안전망 | `claude/nntrainer-refactoring-priority-80xzum` | 867f4c6 | main | 승인 4, READY2MERGE |
| #4464 | `rotary_embedding` op | `...-80xzum-rope` | 3c06690 | #4462 | 승인 2, READY2MERGE |
| #4465 | softcap 버그 수정 (#4461) | `...-80xzum-softcap` | 41d12ee | #4462 | 승인 2, READY2MERGE |
| #4466 | `kv_cache_write` op + 설계 요약 | `...-80xzum-kvwrite` | b3d57b4 | #4464 | 승인 1, 봇 COMMENT |
| #4469 | `attention` op, per-row arm 이동 (6c-1) | `...-80xzum-attn` | e800656 | #4466 | 봇 **REQUEST_CHANGES** |

머지 순서는 #4462 → (#4464, #4465는 서로 독립) → #4466 → #4469이다.

## 3. 머지가 진행될 때 할 일

### 3.1 #4462가 머지되면
- #4464와 #4465는 #4462 위에 쌓여 있다. rebase-merge로 #4462 커밋의 SHA가 바뀌므로, 두 PR을 `origin/main` 위로 rebase하고 force-push해야 할 수 있다.
- 방법: `git rebase --onto origin/main 867f4c6 <branch>`
- 확인: rebase 전후로 `git diff <old-head> <new-head>`가 #4462 외의 변경 없이 같아야 한다.
- force-push는 `--force-with-lease=<branch>:<old-sha>`로 하고, **반드시 사용자 승인 후** 진행한다.
- 제목의 "[Wait for #4462]"도 지운다.

### 3.2 #4464와 #4465 중 하나가 먼저 머지되면
두 PR은 `Applications/CausalLM/layers/unittest_mha_core.cpp`에서 충돌한다. 남은 PR을 rebase하면서 다음처럼 합친다.

- **허용 오차:**
  ```cpp
  const float tol = c.fp16_activations ? 1e-2f
                    : flashStoresFp16Probs(c, rows) ? 2e-3f : 1e-4f;
  ```
- **필드:** #4465는 `softcap_applied` 필드와 `softcapAt(cap, applied)`의 두 번째 인자를 없앴다. #4464는 `fp16_activations` 필드와 `fp16Activations()`를 추가했다. 둘 다 반영한다.
- **헤더 주석:** #4464의 CI 셀 설명(fp16=false/true)과 #4465의 "off x86에서는 32행 이상 step이 있는 gemm 케이스만 실행" 설명을 모두 남긴다.
- **새 테스트 케이스 추가:**
  ```cpp
  MhaCase("fp16_softcap_gemm_long_prefill", 4, 2, 16, {40, 1, 1})
    .softcapAt(<작은 cap>).gemmArm().fp16Activations()
  ```
  ARM FP16 `Sp16` softcap 경로를 실행하는 케이스다. 예전에 qemu에서 cap 0.5로 FP16 경로의 softcap을 빼 봤더니 0.0108 대 1e-2로 겨우 실패했다. 그래서 cap을 더 작게(예: 0.1) 해서 확실히 실패하게 만든다. qemu에서 실패하는 것을 확인한다.

### 3.3 #4466, #4469
- 아래 PR이 머지될 때마다 같은 방식으로 rebase한다.
- #4466 본문의 Dependency 줄("608e158, ten commits")은 오래됐으니 rebase할 때 고친다.

## 4. #4469 리뷰 대응 (다음에 할 일)

myungjoo-bot이 e800656에 REQUEST_CHANGES를 남겼다 (2026-11-02 04:21 UTC). 원문은 PR 페이지에 있다.

1. **Blocking: `cpu_attention.cpp` 커버리지 49.8% (230/462줄), 목표 370줄 이상.**
   `test/unittest/unittest_compute_ops_dispatch.cpp`의 `AttentionDispatchesAndMatchesTheFormula` 옆에 op 단위 케이스를 추가한다. 각 케이스는 단순한 기준값 계산과 비교한다.
   - UINT8 plane + `key_scale` / `value_scale`: decode와 다중 행, `ENABLE_FP16` 빌드에서. int8을 dtype으로 판별하는 부분과 scale 전달을 확인한다.
   - FP32 cache plane: decode와 다중 행.
   - FP16 query + sink: decode와 다중 행. `softmax_row<_FP16>`(x86 forwarder)를 실행한다.
   - FP16 softcap, FP16 non-causal.

   추가한 뒤 `-Db_coverage=true`로 커버리지를 다시 잰다.
2. **Blocking: e800656 squash.** e800656(CodeQL `size_t` 캐스트)을 afa62f8에 합치고 force-push한다 (**사용자 승인 필요**). PR 본문의 커밋 목록과 "two commits" 문구도 고친다.
3. **Minor:**
   - `attention()`에 `p.kv_len <= kv.key->height()`(value도 같이), `q`와 `out`의 dtype 일치, FP32 `q`일 때 UINT8 plane 거부 검사를 추가한다.
   - 각각에 `EXPECT_THROW` 케이스를 만든다.
   - 문서가 약속한 "shapes that do not match"를 지키는 변경이다.
4. **NIT:** `mha_core.cpp` 1751행 근처의 주석("pick these up through layer members")을 `KVCacheView`를 통해 전달한다는 내용으로 고친다.
5. **Minor (기존 공백, 답만):** ARM FP16 per-row arm을 도는 CI 셀이 없다. 로컬 qemu 결과로 답한다.

## 5. 남은 단계 (#4466 설계의 6~10)

| 단계 | 내용 | 검증 | 선행 |
|---|---|---|---|
| 6c-2 | host flash arm(`gemm_attention`)과 두 host arm 사이 선택(`min_prefill_thr`, `host_decode_prefers_per_row`)을 `CpuComputeOps::attention` 안으로. `gemm_attention` 안의 CUDA arm(`cuda_attention_interleaved_fp16`)은 `CudaComputeOps` override로 분리 | x86, qemu | #4469 |
| 6d | CPU K-OHWI scatter와 int8 쓰기를 `kv_cache_write` 뒤로. `KVCacheView`에 layout과 element 타입 추가 | x86, qemu | Q4 답 (int8 유지 여부) |
| (테스트) | `unittest_mha_core`의 device 버전 (#4452) | 빌드만 | 7a 전에 필요 |
| 7a | OpenCL concat 쓰기와 attention을 `ClComputeOps`로 | 빌드만, 장비 필요 | |
| 7b | OpenCL image mirror와 `KVCacheState` | Adreno 필요 | Q2 |
| 8 | CUDA 쓰기와 attention을 `CudaComputeOps`로 | CUDA 장비 필요 | |
| 9 | 레이어에 남은 `#if`와 device 멤버 정리 | x86 | 7, 8 |
| 10 | arm 선택 환경변수(`NNTR_MHA_GPU*`, `NNTR_CUDA_ATTN`, `NNTR_KV_*`)를 DeviceCaps/ExecPlan으로 | | 9 |
| 이후 | `incremental_forwarding` 제거 (#4059 계획 참고, #4456 항목) | | #4451 |

**메인테이너 질문** (#4466 본문):
- **Q1** (host 커널 위치): 사용자 결정으로 **core**. #4469에 반영했다.
- **Q2~Q5:** 답 없음.

## 6. 발견했지만 아직 처리하지 않은 것

- **`ScopedEnv` 공유 헤더:**
  - `unittest_mha_core.cpp`, `unittest_kv_ring.cpp`(`_WIN32` 분기 없음), `test/unittest/unittest_util_func.cpp`에 복사본이 각각 있다.
  - 하나의 테스트 헤더로 합치고, copy/move를 삭제한다 (#4462 jaemini-shin NIT).
- **ARM `-Werror` 빌드 실패:** `nntrainer/tensor/cpu_backend/arm/neon_impl_fp16.cpp`의 사용하지 않는 `hsumq_f16` 때문에 GCC 13 `-Werror`에서 빌드가 멈춘다. 로컬 ARM 빌드는 `werror=false`로 우회했다.
- **ARM `copy_fp32_u16` 버그:** ARM에서는 `__fallback_copy_fp32_u16`이 fp16 변환이 아니라 정수 캐스트를 한다. #4466이 KV 쓰기 경로만 고쳤고, 다른 호출처는 그대로다.
- **non-causal + sliding window:** per-row arm에서는 결과가 정의되지 않는다(소프트맥스와 K/V 커널의 범위가 다름). flash arm은 window를 적용한다. Gemma3/4가 이 설정을 쓸 수 있다. #4469에서는 문서화만 했다.
- **batch > 1:** `unittest_mha_core` 하네스가 batch 1만 지원한다 (`write_kv_cache`의 batch offset 미검증).
- **fallback 아키텍처:** x86/ARM이 아닌 경우 float `softmax_row`, `compute_kcaches`가 NYI를 던진다 (기존 상태).
- **CUDA:** #4465의 CUDA softcap 인자 변경, #4466/#4469의 CUDA 쪽 코드는 컴파일/실행하지 않았다 (stub 헤더로 `mha_core.cpp` 문법 검사만 함).

## 7. #4456의 다른 항목

- #4458 / #4459: Manager, NetworkGraph가 레이어 종류를 직접 판단하는 문제
- #4449, #4450, #4463: 작은 정리 항목
- #4452: device 테스트 하네스
- #4453: CausalLM ↔ core 경계
- #4454: singleton 상태
- #4455: 큰 파일 분리 (`mha_core.cpp`는 #4451 이후)

## 8. 작업 규칙 (이 작업에서 지켜 온 것)

- **커밋 author:** `dlwlzzero <dlwlzzero@gmail.com>`. trailer는 이 순서로 쓴다.
  ```
  Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_015G6v1QJ9tYYPgMHC9YchT3
  ```
- **커밋 메시지:** 제목 앞에 `[CausalLM]`, `[ComputeOps]`, `[cpu_backend]`, `[CI]` 같은 접두어를 붙인다. 본문 끝에는 `**Self evaluation:**` 블록을 둔다. 모델 ID는 쓰지 않는다.
- **PR 본문:** `.github/PULL_REQUEST_TEMPLATE.md` 형식을 따른다 (Dependency, 커밋별 `<details>`, Summary, Signed-off-by). 끝에 Claude Code 생성 문구와 세션 링크를 넣는다.
- **GitHub 코멘트:** 끝에 `---` 다음 줄로 `_Generated by [Claude Code](https://claude.ai/code)_`를 붙인다.
- **force-push, amend:** 이미 push한 브랜치에는 사용자 승인 없이 하지 않는다. 할 때는 `--force-with-lease`로 한다.
- **upstream PR:** 사용자가 요청할 때만 만든다. 이 문서 브랜치(`...-80xzum-docs`)는 upstream PR을 만들지 않는다.
- **편집 금지:** `subprojects/`는 수정하지 않는다.
- **구현과 리뷰:** ponytail(가장 작은 완결된 변경) 원칙을 따르고, PR을 올리기 전에 ponytail-review를 돌린다.
- **커뮤니케이션:** 사용자에게는 한국어로, 비유 없이 쉽게 설명한다.
- **clang-format:** 저장소는 clang-format 14 기준이다. 시스템 기본은 v18이라 결과가 다르니, 14 바이너리를 따로 받아서 쓴다.

## 9. 로컬 검증 방법

빌드 디렉터리 구성 (모두 `-Denable-transformer=true -Denable-test=true`, CausalLM 포함):

| 디렉터리 | 옵션 | 용도 |
|---|---|---|
| `build` | `enable-fp16=false`, werror | x86 fp32 전체 `meson test` (48개) |
| `build16` | `enable-fp16=true`, werror | x86 fp16 전체 `meson test` (51개) |
| `buildcl` | `enable-fp16=true enable-opencl=true`, werror, test off | OpenCL 컴파일 확인 |
| `buildarm` | `enable-fp16=true arm-arch=armv8.2-a enable-blas=false`, werror=false, aarch64 cross | qemu로 ARM 테스트 |

ARM cross 파일 (`aarch64-cross.ini`):
```ini
[binaries]
c = 'aarch64-linux-gnu-gcc'
cpp = 'aarch64-linux-gnu-g++'
ar = 'aarch64-linux-gnu-ar'
strip = 'aarch64-linux-gnu-strip'
pkg-config = '/bin/false'
cmake = 'cmake'
exe_wrapper = ['qemu-aarch64', '-L', '/usr/aarch64-linux-gnu']

[properties]
needs_exe_wrapper = true

[host_machine]
system = 'linux'
cpu_family = 'aarch64'
cpu = 'armv8.2-a'
endian = 'little'
```

qemu 실행:
```sh
cd buildarm
QEMU_LD_PREFIX=/usr/aarch64-linux-gnu \
LD_LIBRARY_PATH=$PWD/nntrainer:$PWD/Applications/CausalLM:$PWD/Applications/CausalLM/layers \
qemu-aarch64 ./Applications/CausalLM/unittest_mha_core
```

자주 쓴 확인:
- `unittest_mha_core`를 `VJEPA_BQ=0`으로 실행해도 통과해야 한다 (#4462에서 테스트가 tile 환경변수를 직접 관리하게 바꿨다).
- 커널이 어느 라이브러리에 있는지 확인: `nm -DC --defined-only build16/Applications/CausalLM/layers/libmha_core_layer.so | grep compute_kcaches`의 결과가 0이어야 한다.
- 변이 테스트: 고친 부분을 일부러 되돌려서 해당 테스트가 실패하는지 확인한다. 예를 들어 cache 행을 +1 밀기, softcap을 제거하기, 홀수 `head_dim`을 convert-only 경로로 보내기.
- CUDA: 툴킷이 없어서 `mha_core.cpp`만 `-DENABLE_CUDA=1`과 stub 헤더(`cuda_runtime.h` 등)로 `-fsyntax-only` 컴파일한다.

## 10. 참고

- umbrella: https://github.com/nntrainer/nntrainer/issues/4456
- 백엔드 분기 이동: https://github.com/nntrainer/nntrainer/issues/4451
- softcap 버그: https://github.com/nntrainer/nntrainer/issues/4461
- 설계 요약: #4466 본문
- `docs/backend_guide/ARCHITECTURE_REFACTOR.md` §4, §10 (각 PR에서 갱신)
