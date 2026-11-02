# #4456 / #4451 리팩토링 handoff (2026-11-02 기준, 같은 날 Android 실기 결과 반영)

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
| #4462 | `unittest_mha_core` 테스트 안전망 | `claude/nntrainer-refactoring-priority-80xzum` | 43ebf10 | main | 승인 4, READY2MERGE. 11-02에 커밋 2개 추가(Android 허용 오차, Android.mk 모듈) → CI·봇 재리뷰 대기 |
| #4464 | `rotary_embedding` op | `...-80xzum-rope` | 3c06690 | #4462 | 승인 2, READY2MERGE |
| #4465 | softcap 버그 수정 (#4461) | `...-80xzum-softcap` | 41d12ee | #4462 | 승인 2, READY2MERGE |
| #4466 | `kv_cache_write` op + 설계 요약 | `...-80xzum-kvwrite` | b3d57b4 | #4464 | 승인 1, 봇 COMMENT |
| #4469 | `attention` op, per-row arm 이동 (6c-1) | `...-80xzum-attn` | b558643 | #4466 | 봇 리뷰 대응 완료, 재리뷰 대기 |

머지 순서는 #4462 → (#4464, #4465는 서로 독립) → #4466 → #4469이다.

## 3. 머지가 진행될 때 할 일

### 3.1 #4462가 머지되면
- #4464와 #4465는 #4462의 **옛 head 867f4c6** 위에 쌓여 있다. #4462에는 11-02에 커밋 2개(bf327f5 Android 허용 오차, 43ebf10 Android.mk 모듈)가 더 붙었고, 둘 다 `unittest_mha_core.cpp`의 허용 오차 줄을 건드리므로 두 PR 모두 rebase 때 그 줄에서 충돌한다. rebase-merge로 SHA도 바뀌므로 어차피 `origin/main` 위로 rebase하고 force-push해야 한다.
- 방법: `git rebase --onto origin/main 867f4c6 <branch>`
- 이 워크스테이션에는 충돌을 이미 푼 브랜치가 로컬로 있다(push 안 함): `wt/4464-on-4462`(`~/nntrainer-4456-stack`), `wt/4465-on-4462`와 결합본 `wt/4465-on-4464`(`~/nntrainer-4456-stack65`). 모두 43ebf10 위에 있으므로 머지 뒤에는 `git rebase --onto origin/main 43ebf10 <wt-branch>`로 옮기면 되고, `git rerere`가 해결 내용을 기록해 두어 같은 충돌은 자동으로 풀린다.
- 확인: `git range-diff 867f4c6..<old-head> origin/main..<new-head>`. main이 141871a5보다 이미 6커밋 앞서 있어서(#4458 머지, ARM ISA 재작업 등) head끼리의 `git diff`에는 그 변경이 섞여 나온다.
- force-push는 `--force-with-lease=<branch>:<old-sha>`로 하고, **반드시 사용자 승인 후** 진행한다.
- 제목의 "[Wait for #4462]"도 지운다.

### 3.2 #4464와 #4465 중 하나가 먼저 머지되면
두 PR은 `Applications/CausalLM/layers/unittest_mha_core.cpp`에서 충돌하고, #4462의 새 커밋(bf327f5)까지 세 갈래다. 결합본은 `wt/4465-on-4464`에 풀어 두었다(위 3.1). 내용은 다음과 같다.

- **허용 오차:**
  ```cpp
  #if defined(ENABLE_FP16) && defined(__ANDROID__)
    const float tolerance = 1e-2f;   // bf327f5: Android fp16 빌드는 FP32 입력도 FP16으로 계산
  #else
    const float tolerance = c.fp16_activations ? 1e-2f
                            : flashStoresFp16Probs(c, rows) ? 2e-3f : 1e-4f;
  #endif
  ```
  비교는 #4464의 `readAt(out, i)`를 쓴다(FP16 출력을 읽기 위해). #4465의 `const float *actual` 줄은 없앤다.
- **필드:** #4465는 `softcap_applied` 필드와 `softcapAt(cap, applied)`의 두 번째 인자를 없앴다. #4464는 `fp16_activations` 필드와 `fp16Activations()`를 추가했다. 둘 다 반영한다.
- **헬퍼:** #4464의 `writeRows`/`readAt`와 #4465의 `flashStoresFp16Probs`가 같은 자리에 들어가므로 둘 다 남긴다.
- **헤더 주석:** #4464의 CI 셀 설명(fp16=false/true), #4465의 "x86 밖에서는 32행 이상 step이 있는 gemm 케이스만 실행", bf327f5의 Android 축소 설명을 모두 남긴다.
- **케이스 목록:** #4465의 softcap gemm 케이스들 뒤에 #4464의 `fp16_*` 케이스를 둔다.
- **새 테스트 케이스:** 결합본에 별도 커밋으로 추가했다.
  ```cpp
  MhaCase("fp16_softcap_gemm_long_prefill", 4, 2, 16, {40, 1, 1})
    .softcapAt(0.1f).gemmArm().flashTile(16, 16).fp16Activations()
  ```
  ARM FP16 `Sp16` softcap 경로를 FP16 입력으로 직접 돌리는 케이스다. 예전에 qemu에서 cap 0.5로 FP16 경로의 softcap을 빼 봤더니 0.0108 대 1e-2로 겨우 실패해서 cap을 0.1로 잡았다.
  - 참고: Android fp16 빌드에서는 FP32 입력이 FP16으로 좁혀지므로 기존 `softcap_gemm_long_prefill`만으로도 이 경로가 실기에서 돈다. S26에서 FP16 경로의 softcap을 빼면 그 케이스만 0.127 차이로 실패했다(11-02). 새 케이스는 qemu·Linux aarch64와 x86 fp16 빌드에서 그 경로를 FP16 입력으로 명시적으로 덮기 위한 것이다.
  - 결합본 검증(11-02): x86 enable-fp16=true에서 `unittest_mha_core` 26/26, `unittest_compute_ops_dispatch` 23/23. S26(Android fp16)에서 26케이스 중 14 통과(두 40행 케이스 포함), 32행 step이 없는 gemm 케이스 12개 skip. FP16 경로의 softcap을 빼는 mutation에서는 정확히 `softcap_gemm_long_prefill`(첫 초과 원소 0.0108, Android 허용 1e-2)과 `fp16_softcap_gemm_long_prefill`(0.0375)만 실패했다. x86 enable-fp16=false 셀은 CI에 맡긴다.

### 3.3 #4466, #4469
- 아래 PR이 머지될 때마다 같은 방식으로 rebase한다.
- #4466 본문의 Dependency 줄("608e158, ten commits")은 오래됐으니 rebase할 때 고친다.

## 4. #4469 리뷰 대응 (완료, 재리뷰 대기)

myungjoo-bot이 e800656에 남긴 REQUEST_CHANGES는 b558643에서 모두 대응했다 (squash + force-push, 2026-11-02). 내용은 다음과 같다.
- op 단위 테스트 3개 추가. FP32/FP16 query, FP32·fp16 비트·int8 cache, decode/prefill, window, softcap, sink, non-causal을 다룬다.
- `cpu_attention.cpp` 커버리지 93% (GCC 13 gcov 기준 339/364줄).
- `attention()`에 캐시 높이와 dtype 검사 추가.
- CodeQL 수정 커밋 squash.
- 오래된 주석 수정.

남은 일은 다음 두 가지다.
- 봇 재리뷰 결과를 확인한다.
- ARM FP16 per-row arm을 도는 CI 셀이 없다는 지적(Minor 4)은 기존 공백이라 답만 했다.

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
- **Android fp16 빌드의 FP32 → FP16 축소 (설계 동작, main에도 있음):** `mha_core.cpp`의 네 곳이 `#if ENABLE_FP16 && defined(__ANDROID__)`로 FP32 Q/K/V/O를 FP16 복사본으로 좁혀 FP16 커널로 계산한다. 그래서 S26에서는 `unittest_mha_core`의 FP32 입력 per-row 케이스가 전부 1e-4를 넘겼다(첫 원소 기준 1.2e-4~3.2e-4, 전체 최대 8e-4; 출력값이 fp16 격자 위에 있고 `-ffast-math` 유무와 무관). qemu(Linux aarch64)는 `__ANDROID__`가 없어서 통과했던 것이다. #4462의 bf327f5가 그 빌드에서 허용 오차를 1e-2로 잡는다.
- **ARM + enable-fp16=false에서 CausalLM 링크 불가 (main 동일):** `mha_core.cpp`가 `__ARM_NEON` 가드 아래에서 fp16 `shgemm`/`neon::custom_hgemm`를 호출해, enable-fp16=false로 빌드한 libnntrainer와는 링크되지 않는다(S26용 fp16=false 빌드에서 확인). 따라서 #4466의 유일한 동작 변경(ARM, UINT16 캐시의 fp16 변환)은 Android에서는 도달할 수 없고, op 단위 `KvCacheWrite*` dispatch 테스트로만 실기 확인했다(27/27).
- **CUDA:** #4465의 CUDA softcap 인자 변경, #4466/#4469의 CUDA 쪽 코드는 컴파일/실행하지 않았다 (stub 헤더로 `mha_core.cpp` 문법 검사만 함).

## 7. #4456의 다른 항목

- #4458 (머지됨) / #4459 (승인, READY2MERGE, GitHub 기준 MERGEABLE이라 rebase 불필요): Manager, NetworkGraph가 레이어 종류를 직접 판단하는 문제
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

### 9.1 Android 실기 (위 표와 다른 워크스테이션: qemu·aarch64 cross 툴체인 없음, Galaxy S26 Ultra 두 대 연결)

11-02에 다섯 PR의 head를 S26 Ultra(SM8850, Android 16, R5KL30G6MCX)에서 돌렸다. 절차는 다음과 같다.

1. 워크트리에서 `git submodule update --init --depth 1` 후 `Applications/CausalLM/lib/libtokenizers_android_c.a`를 복사하고, `cd Applications/CausalLM && ANDROID_NDK=~/android-ndk-r30 ./build_android.sh`(CPU 전용, 약 8분). 산출물: `builddir/jni/arm64-v8a/libnntrainer.so`, `libccapi-nntrainer.so`, `builddir_app/cpu/libcausallm.so`. fp16=false 엔진만 필요하면 `tools/package_android.sh -Denable-fp16=false`.
2. `ln -s $PWD/subprojects/googletest/googletest Applications/CausalLM/jni/googletest` (test/jni도 같은 방식).
3. `unittest_mha_core`: #4462의 43ebf10부터 `Applications/CausalLM/jni/Android.mk`에 모듈이 있다. `cd Applications/CausalLM/jni && $ANDROID_NDK/ndk-build NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk APP_BUILD_SCRIPT=./Android.mk NNTRAINER_ROOT=$PWD/../../.. unittest_mha_core`. 43ebf10 이전 트리에서는 모듈을 로컬로 붙여야 한다(위 커밋을 참고).
4. `unittest_compute_ops_dispatch`: `test/jni/Android.mk`에는 모듈이 없다. `unittest_nntrainer_cpu_backend_fp16` 모듈을 복사해 소스만 `../unittest/unittest_compute_ops_dispatch.cpp`로 바꾸고, `MESON_ARM_MARCH="-march=armv8.2-a+fp16+dotprod+i8mm"`로 ndk-build한다(fp16=false 엔진이면 `-DENABLE_FP16=1 -DUSE__FP16=1`을 뺀다).
5. `adb push`로 두 .so, NDK의 `libc++_shared.so`, 테스트 바이너리를 `/data/local/tmp/<dir>`에 올리고 `LD_LIBRARY_PATH=<dir> ./unittest_...`로 실행한다.

결과(11-02, 전부 fp16=true, `-march=armv8.2-a+fp16+dotprod+i8mm`):

| head | dispatch | `unittest_mha_core` |
|---|---|---|
| #4462 867f4c6 (수정 전) | - | 0 통과, 9 skip, **10 실패** (FP32 입력 케이스, 1e-4 초과) |
| #4462 43ebf10 | - | 10 통과, 9 skip; 최대 오차 8.0e-4 |
| #4465 41d12ee | 17/17 | Android 허용 오차 적용 시 11 통과, 11 skip; flash 40행 step 최대 4.7e-4. FP16 경로 softcap을 빼면 `softcap_gemm_long_prefill`만 실패(0.127) |
| #4469 e800656 | 28/28 | 13 통과, 10 skip (허용 오차 적용 시) |
| #4469 b558643 | 31/31 (새 attention 테스트 3개 포함) | 13 통과, 10 skip (허용 오차 적용 시); 최대 8.0e-4 |
| #4469 e800656, fp16=false | 27/27 | 링크 불가 (§6) |

기기 작업 디렉터리: `/data/local/tmp/refactor4456_1102`.

## 10. fork 브랜치 정리 (완료)

2026-11-02에 사용자가 아래 7개 원격 브랜치의 삭제를 요청했다. 그 세션은 git 권한이 `claude/` 브랜치로 제한돼 있어서 push가 403으로 거부됐고 우회하지 않았다. 같은 날 늦게 확인했을 때 7개 모두 origin에서 지워져 있었다(남은 head 13개). 복구가 필요할 때를 위해 삭제 당시의 SHA를 남긴다.

| 브랜치 | SHA | 상태 |
|---|---|---|
| `refactor/manager-input-lifespan` | c23579e04960668e103a04342d89e6402da993ae | #4458 머지 완료 |
| `security/4334-h1-safetensors-bounds` | 1cdfd7316c414a8f57f78632a9c2d5098a2c5554 | #4342 닫힘 (미머지) |
| `htp/258-fcwh-q8-act` | 25124a1ee24a3b2eaf6b0c738fe9b7e88ea08cca | PR 없음 |
| `htp/4428-attn-q2-from-4432` | 5614d403dcc07e6e4a38f19daf29bb39373cd958 | PR 없음 |
| `htp/4428-prefill-readers` | c8ca04c9567c3c37dcc40e1d52b61534b0099b82 | PR 없음 |
| `htp_hadamard` | 63167235ff936ffd3269a0f511614ad0c6d17508 | PR 없음 |
| `htp_libs_integration` | e2047e9971c8fa37f53c3dfed8a1577f2f7f4680 | PR 없음 |

남겨 둔 브랜치는 다음과 같다.
- 이번 작업 브랜치 6개 (PR 5개 + docs)와 `main`.
- 열린 PR이 연결된 브랜치: `htp_decode`(#4412), `htp_first_version`(#4385), `htp_moe`(#4366), `htp_profiling`(#4417), `hvx_impl`(#4282), `refactor/graph-output-lifespan`(#4459).

#4458이 머지됐지만 #4459는 GitHub 기준 MERGEABLE이라 rebase 없이 머지할 수 있다(main보다 6커밋 뒤).

## 11. 참고

- umbrella: https://github.com/nntrainer/nntrainer/issues/4456
- 백엔드 분기 이동: https://github.com/nntrainer/nntrainer/issues/4451
- softcap 버그: https://github.com/nntrainer/nntrainer/issues/4461
- 설계 요약: #4466 본문
- `docs/backend_guide/ARCHITECTURE_REFACTOR.md` §4, §10 (각 PR에서 갱신)
