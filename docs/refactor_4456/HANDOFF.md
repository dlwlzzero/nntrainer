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
| #4462 | `unittest_mha_core` 테스트 안전망 | `claude/nntrainer-refactoring-priority-80xzum` | d987f6a | main | 승인 4, READY2MERGE. 11-02에 커밋 2개 추가(Android 허용 오차, Android.mk 모듈) → CI·봇 재리뷰 대기 11-03: 7번째 커밋 d987f6a(run_unittest_android.sh·TESTING.md 연결) 추가, fast-forward push. |
| #4464 | `rotary_embedding` op | `...-80xzum-rope` | e9326dd | #4462 (43ebf10) | 승인 2, READY2MERGE. 11-02 43ebf10 위로 rebase, force-push → CI·재리뷰 대기 11-03: d987f6a 위로 재배치(내용 변화 없음), force-push. |
| #4465 | softcap 버그 수정 (#4461) + `fp16_softcap_gemm_long_prefill` | `...-80xzum-softcap` | b0645a8 | **#4464** (564a6ec) | 승인 2, READY2MERGE. 11-02 #4464 위로 옮기고 커밋 1개 추가, force-push → CI·재리뷰 대기 11-03: e9326dd 위로 재배치 + 3번째 커밋 메시지 정정(봇 NIT 1), force-push. |
| #4466 | `kv_cache_write` op + 설계 요약 | `...-80xzum-kvwrite` | 38c94a8 | #4464 | 승인 1, 봇 COMMENT 11-03: #4465(b0645a8) 위로 재배치, 제목 [Wait for #4465]. |
| #4469 | `attention` op, per-row arm 이동 (6c-1) | `...-80xzum-attn` | 9f5578b | #4466 | 봇 리뷰 대응 완료, 재리뷰 대기 11-03: #4466(38c94a8) 위로 재배치 + 봇 Minor 2개(sink 검사, 거부 테스트) 둘째 커밋에 병합. 이 세션이 처리함. |

머지 순서는 #4462 → #4464 → #4465 → #4466 → #4469이다 (11-02부터 #4465가 #4464 위에 있다; #4466, #4469는 아직 #4464의 옛 head 3c06690 위에 있어서 #4464 머지 때 rebase한다).

11-02 저녁에 그 다음 단계를 로컬에서 먼저 쌓았다 (fork에 push, PR은 아직). 자세한 내용은 §12.

| 브랜치 (fork) | head | 내용 | 쌓인 위치 |
|---|---|---|---|
| (로컬 `wt/4466-on-4465`, 워크트리 `~/nntrainer-4456-stack65`) | c7bc1c5 | #4466 두 커밋 + #4469 두 커밋을 c584e44 위로 cherry-pick (충돌 없음) | #4465 (c584e44) |
| `claude/nntrainer-refactoring-priority-80xzum-flash` | c27af69 | 6c-2: host flash arm + host arm 선택 → `CpuComputeOps::attention`, CUDA arm → `CudaComputeOps::attention` | 위 재배치본 |
| `claude/nntrainer-refactoring-priority-80xzum-devtest` | b0dfa2c | `unittest_mha_core`의 device 모드 `NNTR_ENGINE=gpu` (#4452, 7a 준비; 워크트리 `~/nntrainer-4456-devtest`) | 6c-2 |

## 3. 머지가 진행될 때 할 일

### 3.1 #4462가 머지되면
- #4464와 #4465는 #4462의 **옛 head 867f4c6** 위에 쌓여 있다. #4462에는 11-02에 커밋 2개(bf327f5 Android 허용 오차, 43ebf10 Android.mk 모듈)가 더 붙었고, 둘 다 `unittest_mha_core.cpp`의 허용 오차 줄을 건드리므로 두 PR 모두 rebase 때 그 줄에서 충돌한다. rebase-merge로 SHA도 바뀌므로 어차피 `origin/main` 위로 rebase하고 force-push해야 한다.
- 방법: `git rebase --onto origin/main 867f4c6 <branch>`
- 11-02에 사용자 승인으로 force-push했다: #4464 ← `wt/4464-on-4462`(564a6ec), #4465 ← 결합본 `wt/4465-on-4464`(c584e44). 그래서 두 PR은 이미 43ebf10 위에 있고 서로의 충돌도 풀려 있다. 머지 뒤에는 `git rebase --onto origin/main 43ebf10 <branch>`만 하면 되고, 이 워크스테이션의 `git rerere`(`~/nntrainer-4456-stack*`)가 해결 내용을 기록해 두었다. `wt/4465-on-4462`(#4465를 #4464 없이 43ebf10 위에만 올린 것)도 로컬에 남아 있다.
- 확인: `git range-diff 867f4c6..<old-head> origin/main..<new-head>`. main이 141871a5보다 이미 6커밋 앞서 있어서(#4458 머지, ARM ISA 재작업 등) head끼리의 `git diff`에는 그 변경이 섞여 나온다.
- force-push는 `--force-with-lease=<branch>:<old-sha>`로 하고, **반드시 사용자 승인 후** 진행한다.
- 제목의 "[Wait for #4462]"도 지운다.

### 3.2 #4464와 #4465의 결합 (11-02에 완료, #4465 브랜치에 반영됨)
두 PR은 `Applications/CausalLM/layers/unittest_mha_core.cpp`에서 충돌했고, #4462의 새 커밋(bf327f5)까지 세 갈래였다. 11-02에 #4465를 #4464 위로 옮기면서 다음처럼 합쳤고, 그 결과가 #4465의 현재 head(c584e44)다. 머지 순서가 바뀌어 #4464를 #4465 위에 올려야 하면 같은 내용을 반대로 적용한다.

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
- **새 테스트 케이스:** #4465의 세 번째 커밋(c584e44)으로 추가했다.
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
- 11-02 저녁: 두 PR을 #4465의 새 head(c584e44) 위로 옮긴 결과가 로컬 브랜치 `wt/4466-on-4465`(c7bc1c5, `~/nntrainer-4456-stack65`)에 있다. `git cherry-pick 2e14a9de8 b3d57b4 fd51d4855 b55864311`로 충돌 없이 올라갔다. #4469는 다른 세션 담당이므로 그 PR 브랜치에는 push하지 않았다. #4464가 머지되면 이 브랜치를 `git rebase --onto origin/main c584e44`(또는 머지된 SHA)로 옮겨 force-push하면 된다(승인 필요).

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
| 6c-2 | **구현 완료 (11-02, §12)**: host flash arm(`gemm_attention`)과 두 host arm 사이 선택(`min_prefill_thr`, `host_decode_prefers_per_row`)을 `CpuComputeOps::attention` 안으로. `gemm_attention` 안의 CUDA arm(`cuda_attention_interleaved_fp16`)은 `CudaComputeOps::attention` override로 분리 | x86, S26 (qemu 없음) | #4469 머지 후 PR |
| 6d | **구현 완료 (11-02 밤, §12.5)**: CPU K-OHWI scatter와 int8 쓰기를 `kv_cache_write` 뒤로. `KVCacheView`에 layout과 int8 element 추가 | x86, S26 | 6c-2 위; #4469 머지 후 PR |
| (테스트) | `unittest_mha_core`의 device 버전 (#4452) | 빌드만 | 7a 전에 필요 |
| 7a | OpenCL concat 쓰기와 attention을 `ClComputeOps`로 | 빌드만, 장비 필요 | |
| 7b | OpenCL image mirror와 `KVCacheState` | Adreno 필요 | Q2 |
| 8 | CUDA 쓰기와 attention을 `CudaComputeOps`로 | CUDA 장비 필요 | |
| 9 | 레이어에 남은 `#if`와 device 멤버 정리 | x86 | 7, 8 |
| 10 | arm 선택 환경변수(`NNTR_MHA_GPU*`, `NNTR_CUDA_ATTN`, `NNTR_KV_*`)를 DeviceCaps/ExecPlan으로 | | 9 |
| 이후 | `incremental_forwarding` 제거 (#4059 계획 참고, #4456 항목) | | #4451 |

**메인테이너 질문** (#4466 본문):
- **Q1** (host 커널 위치): 사용자 결정으로 **core**. #4469에 반영했다.
- **Q4** (int8 KV 유지 여부): 11-02 사용자 결정으로 **유지**. 6d는 int8 쓰기를 `kv_cache_write` 뒤로 옮긴다(막힘 해제).
- **Q2** (`KVCacheState` 소유권): 11-02 사용자 결정으로 **제안대로** — `KVCacheView`에 `std::shared_ptr<KVCacheState>`(내용은 백엔드 정의, 레이어는 불투명 포인터만 보관, Cl op가 처음 쓸 때 생성). ClContext 맵 대안은 캐시 재바인딩 시 수명 문제로 기각. 7b 착수 가능.
- **Q3, Q5:** 사용자 결정으로 **보류**.

## 6. 발견했지만 아직 처리하지 않은 것

- **`ScopedEnv` 공유 헤더:**
  - `unittest_mha_core.cpp`, `unittest_kv_ring.cpp`(`_WIN32` 분기 없음), `test/unittest/unittest_util_func.cpp`에 복사본이 각각 있다.
  - 하나의 테스트 헤더로 합치고, copy/move를 삭제한다 (#4462 jaemini-shin NIT).
- **ARM `-Werror` 빌드 실패:** `nntrainer/tensor/cpu_backend/arm/neon_impl_fp16.cpp`의 사용하지 않는 `hsumq_f16` 때문에 GCC 13 `-Werror`에서 빌드가 멈춘다. 로컬 ARM 빌드는 `werror=false`로 우회했다. → **11-02 #4472로 올렸다** (fork 브랜치 `claude/cpu-backend-drop-unused-hsumq`, 함수 삭제 한 건; NDK clang `-Wunused-function`으로 전후 확인).
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
- #4452: device 테스트 하네스 (11-02에 `unittest_mha_core` device 모드로 착수, §12.3)
- #4472: `hsumq_f16` 삭제 (이 작업 중 발견한 ARM `-Werror` 실패, 독립 PR)
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
- CUDA: 툴킷이 없어서 `mha_core.cpp`만 `-DENABLE_CUDA=1`과 stub 헤더(`cuda_runtime.h` 등)로 `-fsyntax-only` 컴파일한다. 11-02에 `nntrainer/cuda/cuda_compute_ops.cpp` 전체가 통과하는 stub 세트(`cuda.h`, `cuda_runtime.h`, `nvrtc.h`, `cublas_v2.h`)를 이 폴더의 `cuda_stubs/`에 두었다. 사용: `build16/compile_commands.json`에서 `cpu_attention.cpp`의 컴파일 명령을 가져와 `-o/-MD/-c`를 빼고 `-fsyntax-only -DENABLE_CUDA=1 -I <cuda_stubs> -I nntrainer/cuda nntrainer/cuda/cuda_compute_ops.cpp`를 붙인다.

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

## 12. 11-02 저녁 작업 (6c-2, #4472, device 하네스)

### 12.1 6c-2: host flash arm과 host arm 선택을 op로 (브랜치 `claude/nntrainer-refactoring-priority-80xzum-flash`, c27af69)

- 커밋 한 개. `mha_core.cpp`에서 약 830줄이 빠진다 (`gemm_attention` 본문, `mha_hsgemm_avx2`, `vjepa_expq_f32`, fp16 커널 재선언, 두 host-path guard, `host_decode_prefers_per_row`).
- `AttentionOpParams::flash` 추가 (= 레이어의 `use_gemm_attention`; "flash 계열 arm을 써도 된다", 어떤 arm을 쓸지는 백엔드가 정함).
- `cpu_attention.cpp`: `FlashAttention` 구조체(필드 이름은 레이어 멤버 그대로, 본문은 그대로 이동, `vjepa_expq_f32` → `neon_mathfun`의 `exp_ps`), `flash_min_rows()`(= `min_prefill_thr`, NNTR_MIN_PREFILL/NNTR_MHA_GPU를 그대로 읽음), `decode_prefers_per_row()`. 선택 규칙: `p.flash && !sink && cache != FP32 && rows >= flash_min_rows() && !(rows == 1 && decode_prefers_per_row())`. ARM fp16 커널 호출은 `ENABLE_FP16` 가드 뒤로(ARM fp16=false libnntrainer도 빌드되므로); 그 밖의 경우(generic)는 K/V 타일을 fp32로 넓혀 fp32 sgemm — 예전 레이어의 `#else` 분기는 `uint16_t*`를 sgemm에 넘겨 컴파일이 안 되던 코드였다. Windows ARM64 CI(clang-cl, fp16 없음)가 이 generic 분기를 실제로 컴파일한다.
- `cuda_compute_ops.cpp`: `attention` override = CUDA arm 그대로(p.flash, NNTR_CUDA_ATTN, FP16 q/o, int8 아님, dev_accessible) → 아니면 device-only 메모리 거부(레이어의 옛 메시지) + `drain_if_async()` → `CpuComputeOps::attention`. 레이어의 `MhaAttnTimerScope` 버킷은 CUDA arm에 더 이상 없음(진단용).
- 레이어: OpenCL arm 블록은 그대로 두고(조건에서 `_cuda_attn_on`, `per_row_decode` 제거), 두 host tail을 하나로 합쳐 `attend()`(옛 `attend_rowwise`)를 호출. ring guard는 op가(메시지에 힌트 포함), host-readable guard는 CUDA op가 맡는다.
- **동작 차이 두 가지(PR 본문에 적음):** ARM에서 device arm이 decode step을 거절하면 예전엔 host flash(M=1, 2.5~5배 느림)였고 지금은 per-row; CUDA arm의 ATTN_TIMER 버킷 삭제.
- 테스트: `unittest_compute_ops_dispatch`에 `AttentionFlashArm`, `AttentionFlashArmFp16Query` 추가(40행, 8x8 타일, softcap/window/non-causal/chunk/decode/int8, sink·FP32 cache는 per-row로 떨어짐 확인). 기존 attention 케이스의 q 진폭을 0.3→2.0으로 키워 softcap/window를 빼면 허용 오차를 넘게 했다(처음엔 2e-3로 통과해 버려서 x86 1e-4 / ARM 2e-3로 나눔).
- 검증: x86 fp16=true 27/27 + 33/33, fp16=false 23 통과/4 skip + 30/30, `VJEPA_BQ=0`도 통과. S26 dispatch 33/33, `unittest_mha_core` 15 통과/12 skip. 변이 4종(x86): softcap 삭제 → mha 5 + dispatch 2 실패; window 하한 삭제 → dispatch 2 실패; flash 출력 ×1.01 → mha gemm 13 + dispatch fp32 flash 실패; generic 분기를 x86에서 강제 → 모두 통과. NDK clang `-fsyntax-only` fp16/nofp16 통과(nofp16에서 `Qp_fp16` 미사용 경고 → `[[maybe_unused]]`). CUDA stub 문법 검사 통과.
- 커버리지(봇이 6c-1에서 요구했던 것): x86 enable-fp16 `-Db_coverage=true`(`buildcov`, tflite 옵션 둘 다 false 필요) 빌드에서 두 테스트를 돌린 gcov(GCC 13) 기준 `cpu_attention.cpp` **94.65% (617줄)**. 미실행 33줄 중 flash 영역은 11줄: ARM 전용 FP16-Q 복사 경로(x86에선 컴파일만 됨, S26에서 실행됨), `NNTR_MIN_PREFILL` 환경변수 분기, 오류 메시지 3개.
- PR 본문 초안은 세션 스크래치패드 `pr_6c2_body.md`(세션 종료 시 사라짐)와 같은 내용이 커밋 메시지에 있다. **PR은 #4469가 c584e44 위로 재배치된 뒤에 연다** (지금 열면 #4466/#4469 커밋이 PR 본체와 다른 SHA로 섞여 보인다).

### 12.2 #4472 (`hsumq_f16` 삭제)

upstream PR, CI 통과 중, 리뷰 대기. 독립적이라 스택과 무관하게 머지 가능.

### 12.3 #4452 device 하네스: `unittest_mha_core`의 `NNTR_ENGINE=gpu` 모드 (브랜치 `claude/nntrainer-refactoring-priority-80xzum-devtest`, b0dfa2c)

- 테스트 파일 하나만 바뀐다. `NNTR_ENGINE=gpu`이면(enable-opencl 빌드 + 장비) 입력 3개·출력·캐시 2개를 gpu context의 allocator 위 `TensorPool`(SVM)에서 받아 ContextData를 붙이고, 레이어가 latch하기 전에 `NNTR_MHA_GPU=1`, `NNTR_MHA_GPU_DECODE=1`을 테스트가 직접 세팅한다(이 모드에선 `kArmSelectingEnv` 검사와 ARM의 32행 gemm skip을 하지 않음). 모든 케이스 허용 오차 1e-2. 결과 읽기 전에 `cl_queue_finish()`. OpenCL 빌드가 아니거나 context가 없으면 skip. host 모드는 그대로.
- **S26 첫 실행 (11-02):** host 모드 15 통과/12 skip(이전과 동일), **gpu 모드 26/27 통과, `no_rope_gemm` 1건 실패** (step [0,5) 원소 0: -0.801 vs 기대 0.603). `NNTR_MHA_ATTN_TIMER=1` 출력으로 device arm이 실제로 돌았음을 확인(decode 30회, prefill 17회). 이 형상(KV heads 2, head_dim 16, 512행)에서는 K/V image mirror 생성이 거부돼(`[IMG-ATTN] ... Image attention is OFF`) buffer(flash) arm이 돌았다. 즉 image arm(7b)은 아직 이 하네스로 못 본다 — 더 큰 head_dim/행 수 케이스가 필요.
- **`no_rope_gemm` 실패의 의미:** `rope_theta=0`인 gemm 레이어의 device 경로. RoPE가 있으면 K는 device(`rope_inplace_f16_cl`)가 캐시에 쓰고, 없으면 host가 `write_kv_cache(..., rotate_key=false)`로 SVM 캐시에 쓴 뒤 device attention arm이 읽는다. 같은 설정의 host 경로(`no_rope_rowwise`, 그리고 host 모드의 모든 케이스)는 통과하므로 device 경로 쪽 문제다. 원인은 아직 확인하지 않았다(후보: host 쓰기 → device 읽기 사이의 SVM coherence, 또는 no-RoPE 케이스에서 GPU arm의 Q/K 전처리). 7a에서 KV write/attention이 `ClComputeOps`로 들어갈 때 "op가 coherence를 소유한다" 규칙으로 자연히 다뤄질 자리이며, 그 전에 별도 수정 PR로 잡을 수도 있다. upstream 이슈는 아직 안 올렸다.
- 빌드/실행 절차(§9.1에 더해): `cd Applications/CausalLM && ANDROID_NDK=~/android-ndk-r30 ./build_android.sh -Denable-opencl=true` (엔진에 `-DENABLE_OPENCL`이 들어가고 `android_build_result/nntrainer-abi.ini`로 app 빌드에도 전달됨; 산출물에 `builddir/jni/arm64-v8a/libOpenCL.so` 로더 스텁 포함). `Applications/CausalLM/jni/googletest` 심볼릭 링크 필요. ndk-build는 §9.1과 같고, 기기에는 `libOpenCL.so`도 올린다. **`LD_LIBRARY_PATH`에 `/vendor/lib64`를 넣으면 안 된다** (`CANNOT LINK EXECUTABLE ... libandroidfw.so`로 죽는다; 스텁 로더가 vendor 라이브러리를 알아서 연다). 실행: `NNTR_ENGINE=gpu NNTR_MHA_ATTN_TIMER=1 LD_LIBRARY_PATH=<dir> ./unittest_mha_core`.
- x86 non-OpenCL 빌드에서는 device 코드가 `#if defined(ENABLE_OPENCL)`로 빠지며 문법 검사 통과(build16 컴파일 플래그). CI의 `ubuntu_opencl_build`는 CausalLM을 빌드하지 않으므로 이 코드는 Android OpenCL 빌드에서만 컴파일된다.
- PR은 6c-2 위에 있지만 코드상 6c-2에 의존하지 않는다(#4462의 테스트 파일에만 의존). #4469 재배치 뒤 6c-2와 함께, 또는 #4469 위로 옮겨 따로 올릴 수 있다.

### 12.3.1 `no_rope_gemm` device 실패의 원인과 수정 (#4473, 11-02 밤)

- 원인: `attention_fp32_on_device()`가 FP32 Q/K/V를 device 커널로 FP16 plane에 좁혀 두는데(drain 없음, 같은 큐의 device arm이 읽는 전제), RoPE 없는 레이어의 no-RoPE 분기는 K/V를 **host**가 캐시에 쓴다. 그 host 읽기 전에 `settle_act_stage_for_host()`(plane을 map)를 안 해서 stale 바이트가 캐시에 들어갔다. host RoPE fallback 경로는 이미 settle을 호출한다.
- 확인: devtest 트리의 no-RoPE 분기에 `settle_act_stage_for_host()` 한 줄을 넣고 S26에서 `no_rope_gemm` 통과(나머지 케이스 영향 없음).
- 수정 PR: **#4473** (upstream main 기준, fork 브랜치 `claude/mha-core-settle-planes-no-rope`, 워크트리 `~/nntrainer-norope`). 8줄, `ENABLE_OPENCL` 가드 안. x86 빌드는 이 줄을 컴파일하지 않으므로 Android OpenCL 빌드(devtest)에서의 컴파일·실행이 검증 근거다.
- 영향 범위: gpu 엔진 + FP32 activation + RoPE 없는 attention 레이어. `Applications/CausalLM/models`에 `rope_theta=0` 레이어를 쓰는 모델은 현재 없다.

### 12.4 fork 브랜치 (11-02 저녁 추가)

- `claude/cpu-backend-drop-unused-hsumq` (#4472, upstream main 기준)
- `claude/nntrainer-refactoring-priority-80xzum-flash` (6c-2, c27af69; 아래에 #4466/#4469 재배치 커밋 포함)
- `claude/nntrainer-refactoring-priority-80xzum-devtest` (device 하네스, b0dfa2c; 6c-2 위)
- `claude/mha-core-settle-planes-no-rope` (#4473, upstream main 기준)
- `claude/nntrainer-refactoring-priority-80xzum-kvlayout` (6d, accc699; 6c-2 위)

### 12.5 6d: K-OHWI와 int8 KV 쓰기를 `kv_cache_write` 뒤로 (브랜치 `claude/nntrainer-refactoring-priority-80xzum-kvlayout`, accc699, 6c-2 위)

- 커밋 한 개. Q4(int8 유지) 결정에 따라 int8은 없애지 않고 op로 옮겼다.
- `KVCacheView`: `layout`(`KVCacheLayout::Concat` / `KOhwi`, K plane의 레이아웃)과 int8 element(UINT8 plane + fp16 scale plane 포인터 `key_scale`/`value_scale`, 쓰기 위해 non-const로 변경) 추가.
- `CpuComputeOps::kv_cache_write`: UINT8 plane이면 step을 fp16 bits로(K는 회전) 만든 뒤 (row, head)마다 amax/127 scale로 int8 양자화해 같은 행의 scale plane에 기록. `KOhwi` K plane이면 concat 순서의 fp16 step을 만든 뒤 head별 stride로 scatter. FP32 step도 fp16으로 바꿔 받으므로 이제 int8/OHWI 쓰기에 FP16 step이 필요 없다. scatter/gather/quantizer는 레이어에서 그대로 옮겨 `nntrainer/tensor/cpu_backend/kv_cache_layout.h`(내부 헤더, 미설치)에 두고 두 op가 공유.
- `CpuComputeOps::attention`: `KOhwi` plane이면 arm 전에 [0, kv_len) 행을 concat으로 gather(레이어가 하던 대로).
- 레이어: `kv_layout`을 finalize에서 한 번 고정(`NNTR_KV_OHWI` && !int8 && fp16 cache). 이전 write 조건은 "FP16 step"도 요구해 x86 FP32-activation 실행에선 concat으로 쓰고 read는 OHWI로 gather하는 불일치가 있었다(env 조합 버그, 이제 일치). 세 write 지점이 `write_kv_cache()`로 layout과(int8이면) scale plane을 넘긴다. int8 분기는 Q RoPE와 read용 scale 포인터 stash만 남김. **read 쪽 gather는 레이어에 남겼다**: OpenCL two_conv arm이 concat을 읽고 host 복사본(`svm_inputs=false`로 업로드)을 소비하기 때문에, 그 arm이 7a에서 op로 들어갈 때 같이 간다. op에는 gather된 Concat plane이 전달된다.
- 동작 차이(env 경로만): int8 write가 `key_step`을 제자리 회전하지 않음(이후 아무도 안 읽음); `NNTR_KV_OHWI` + FP32 activation x86에서 write/read 레이아웃 일치.
- 테스트: `unittest_mha_core`에 `internal_cache_rowwise`(3입력 모드), `int8_rowwise`, `int8_gemm_long_prefill`(FP16 activation, 허용 오차 2e-2: 레이어는 fp16 회전·참조는 fp32 회전이라 양자화 경계 근처 값이 한 step 어긋날 수 있음), `ohwi_rowwise`, `ohwi_no_rope_rowwise`, `ohwi_gemm_long_prefill`. 이 케이스들은 `NNTR_KV_INT8`/`NNTR_KV_OHWI`를 스스로 세팅(레이어가 finalize마다 읽으므로 가능; `kArmSelectingEnv` skip에서 제외). dispatch에 `KvCacheWriteScattersAKOhwiKeyPlane`, `AttentionReadsAKOhwiKeyPlane`(concat과 bit 동일), `KvCacheWriteQuantizesAnInt8Cache`(공식 정확 일치 + scale 없으면 거부), `AttentionReadsWhatKvCacheWriteQuantized`(fp16 빌드).
- 검증: x86 fp16=true 33/33 + 37/37; fp16=false 24 통과/9 skip(fp16·int8·OHWI 케이스) + 33/33; S26 37/37 + 21 통과/12 skip(새 6개 모두 통과, int8 NEON 커널이 op가 쓴 캐시를 읽음). 변이 4종: scatter stride 틀림 → ohwi 레이어 3 + dispatch 2 실패; int8 K 미회전 → int8 레이어 2 + dispatch 1; scale amax/126 → dispatch 1(레이어는 2e-2 안); gather 높이 틀림 → dispatch 1. NDK clang 문법 검사 fp16/nofp16 통과. 커버리지(gcov): `kv_cache_layout.h` 100%, `cpu_attention.cpp` 94.59%, `cpu_ops_table.cpp` 전체 49.73%(다른 op 포함; `kv_cache_write` 안 미실행은 오류 메시지 2개, FP16 step의 copyData 분기, "whole heads" 검사).
- PR 본문 초안은 커밋 메시지 + 위 검증 내용. **PR은 6c-2 PR 뒤에 연다**(6c-2 위에 쌓임).

### 12.6 이 시점의 전체 그림 (11-02 밤)

- 머지 대기: #4462 → #4464 → #4465(승인/CI 통과) → #4466 → #4469(재리뷰 대기, 다른 세션).
- 로컬·fork에 준비됨(PR 전): 재배치본(c7bc1c5) → 6c-2(c27af69) → 6d(accc699); device 하네스(b0dfa2c, 6c-2 위).
- 독립 PR: #4472(hsumq), #4473(no-RoPE settle). 둘 다 CI 통과/리뷰 대기.
- 결정: Q1 core, Q2 shared_ptr<KVCacheState> in view, Q4 int8 유지, Q3/Q5 보류.
- 다음 구현 후보: 7a(OpenCL concat write + attention을 `ClComputeOps`로; device 하네스로 검증, S26 있음), 7b(image mirror + `KVCacheState`, Q2 결정대로).

### 12.7 2026-11-03: 리뷰 대응 3건 (사용자 지시 "1.2.3 모두 진행")

- **#4462 (A안)**: 봇 Should fix 대응 커밋 d987f6a — `run_unittest_android.sh`가 `unittest_mha_core`를 같은 ndk-build로 빌드·push·실행하고 `--filter`를 두 바이너리에 전달, 둘 중 하나라도 실패하면 exit≠0(종전엔 `set -e` 때문에 첫 바이너리 실패 시 요약 줄 전에 종료). `TESTING.md` 동반 수정. CI(`causallm_android.yml`)는 테스트 바이너리를 하나도 안 만들므로 별건으로 남김. 스크립트를 처음으로 끝까지 돌린 결과(S26, NDK r30, 엔진 재빌드): `unittest_mha_core` 10 pass/9 skip; `unittest_causallm_models` 104/109 — `*_Q40_FP16 PromptProducesExpectedLogits` 4개(logit 0 vs fixture)와 `Gemma4DifferentialTest.QINT4CloseToFP32Reference`(nntr_quantize가 QINT4 저장 거부 "This dtype is not supported in save with quantization") 실패. 이 PR은 production 소스를 안 건드리므로 PR과 무관한 기존 문제(Android 빌드 또는 fixture 문제인지는 미조사). fixture는 tarball과 동일함을 확인. 로그: scratchpad `run_unittest_android-4462.log`.
- **#4464/#4465 재배치**: `wt/4464-on-4462` → e9326dd, `wt/4465-on-4464` → b0645a8 (모두 `~/nntrainer-4456-stack`). 옛 head 대비 diff는 d987f6a의 두 파일뿐. #4465 3번째 커밋 메시지 정정: "the path runs wherever enable-fp16 is on" → 케이스는 모든 enable-fp16 빌드에서 돌지만 FP16 score 경로는 `#if !x86 && __ARM_NEON` 아래라 x86 CI는 FP32 score 경로를 탐. S26 변이 근거(11-02 15:46 KST, 트리 c584e44, `if (softcap > 0.0f)`→`if (false)`: long_prefill 두 케이스만 0.0108/0.0375로 실패)를 답글로 게시.
- **#4473 재구성**: 하네스(devtest 브랜치, 6c-2 기반)를 #4462 head로 포팅(충돌 2곳: ARM skip 블록, tolerance 블록) + device 모드에서는 `c.softcap_applied = true`(OpenCL arm은 softcap을 적용하므로 `softcap_gemm_not_applied`가 device에서 0.0105로 실패하던 것을 reference 쪽에서 반영). 커밋 c647890(하네스) + 1202418(수정, f83bb92 cherry-pick). S26 OpenCL 빌드: host 10/9skip(수정 전후 동일); gpu 수정 전 18/19(`no_rope_gemm` 1.40 at step [0,5) element 0), 수정 후 19/19. x86 fp16=false: 19/19, `NNTR_ENGINE=gpu`는 19 skip. 브랜치 `claude/mha-core-settle-planes-no-rope` force-push, 워크트리 `~/nntrainer-4456-devtest`(`wt/4473-on-4462`, OpenCL builddir)와 `~/nntrainer-norope`(같은 커밋). 옛 `wt/device-harness`(6c-2 기반, b0dfa2c)와 fork 브랜치 `...-80xzum-devtest`는 이제 불필요.
- **후속 영향**: 하네스가 `unittest_mha_core.cpp`의 두 블록을 바꾸므로 #4464/#4465/#4466/#4469/6c-2/6d 중 #4473 뒤에 머지되는 쪽은 그 파일에서 작은 rebase 충돌을 겪는다(rerere에 해결 기록 없음 — 포팅 때 수동 해결). #4469는 다른 세션 소유.
- **push 교훈**: `--force-with-lease=<ref>:<sha>`의 `<ref>`는 원격 브랜치 이름(`claude/...`)이어야 한다. `refs/remotes/origin/...`을 쓰면 lease가 적용되지 않아 일반 push로 처리되어 non-fast-forward 거부가 난다(11-03 한 번 발생, 재push로 해결; 그 사이 올라간 답글·본문은 결과적으로 사실과 일치).

### 12.8 2026-11-03 오후: 이슈 1·2·3·6 처리 (사용자 지시 "1,2,3,6 이슈 해결해")

- **1 재배치**: 체인 전체를 #4465 새 head(b0645a8) 위로: #4466 38c94a8 → #4469 9f5578b → 6c-2 66bd5b5 → 6d 9c2406b. 옛 체인 대비 내용 차이는 #4462의 wiring 두 파일과 #4469 수정뿐. push: `...-kvwrite`(#4466), `...-attn`(#4469), `...-flash`(6c-2), `...-kvlayout`(6d). 로컬 브랜치 `wt/4466-on-4465`=#4469 head(이름과 달리 #4469까지 포함), `wt/6c2-flash`, `wt/6d-kvwrite`(`~/nntrainer-4456-stack65`). x86 fp16: #4469 시점 dispatch 31/31 + mha 27/27, 6d 시점 37/37 + 33/33. 6c-2를 다시 올릴 때 `compute_ops.h` 문서와 dispatch 테스트 주석 위치에서 충돌 2곳(rerere 기록됨).
- **2 #4469 봇 Minor**: `CpuComputeOps::attention`이 sink의 size≠num_heads_q 또는 dtype≠q를 거부; `AttentionRejectsShortCacheAndBadDtypes`에 short-K, sink 크기·dtype, (ENABLE_FP16) FP16 q over FP32 planes 추가; 커밋 메시지 정정; Co-Authored-By에 이 세션 추가. PR 본문·답글 게시.
- **3 폰 실패 5건 → upstream 이슈 2개**:
  - **#4475 QINT4**: `Layer::save`에 QINT4 분기가 없어 `nntr_quantize --fc_dtype QINT4`가 모든 tiny fixture에서 실패(x86 호스트 재현, gdb bt: Layer::save ← LayerNode::save ← NeuralNetwork::save ← Transformer::save_weight). x86 테스트는 그 전에 skip.
  - **#4476 Q40_FP16**: Android fp16 빌드에서 로짓 전부 0. 폰 프로브(`$S/probe/probe_q4_0_fp16.cpp`, libnntrainer 링크)로 `gemm_q4_0<_FP16>`는 정상 확인(fp32 arm과 동일 오차, 0 없음). x86 fp16은 meson 게이트를 풀어 빌드해도 `NYI : __fallback_gemm_q4_0`로 실행 불가 → 이 경로는 CI 어디서도 안 돎. 원인 미확정(embedding Q4_0→FP16, FP16 RMSNorm/residual, tied lm_head 모두 코드상 이상 없음). 호스트 `~/nntrainer-4456-base/build16x`는 게이트를 임시로 푼 fp16 빌드(소스는 되돌림).
- **6 정리**: fork `...-80xzum-devtest` 삭제; worktree attn/attn-nofp16/softcap/hsumq/norope 삭제; 로컬 브랜치 wt/device-harness, wt/4462-android-tol, wt/4465-on-4462 삭제; 폰 `/data/local/tmp/refactor4456_1102` 삭제(`nntr_causallm_test`는 스크립트용으로 유지).
- **다음**: 6c-2·6d PR 열기(#4469 재배치 끝났으므로 가능; 본문 초안 `pr_6c2_body.md`, `pr_6d_body.md`는 base 갱신 필요), #4473 하네스와의 충돌은 머지 순서에 따라 처리.

### 12.9 2026-11-03 저녁: 6c-2·6d PR 열림, 7a 1차 조각 구현 (사용자 지시 "keep going")

- **PR**: #4477 = 6c-2(`...-80xzum-flash` 66bd5b5, [Wait for #4469]), #4478 = 6d(`...-80xzum-kvlayout` 9c2406b, [Wait for #4477]). 6c-2는 재배치 뒤 x86 fp16 27/27·33/33으로 재확인하고 열었다.
- **7a 1차 조각** (브랜치 `wt/7a-cl-attention` = fork `claude/nntrainer-refactoring-priority-80xzum-clattn`, c8e505e, 6d 위; 워크트리 `~/nntrainer-4456-devtest`, OpenCL Android builddir): `ClComputeOps::attention`이 레이어의 OpenCL flash(decode/prefill, SVM)·two_conv(SVM 또는 host 복사본) arm을 레이어와 같은 순서·게이트(NNTR_MHA_GPU, NNTR_MIN_PREFILL, sink 없음, FP16 q/out/cache, concat K)로 돌리고 실패 시 `cl_queue_finish()` 후 host arm. 새 `AttentionOpParams::device_decode`(레이어의 `gpu_decode_attn` 속성 OR NNTR_MHA_GPU_DECODE). 레이어의 flash/two_conv 블록 삭제(image arm·int8 GPU arm·clmem 모드는 남김). **발견한 함정**: FP32 활성값 경로(`attention_fp32_on_device`)가 만드는 FP16 임시 텐서가 bare Tensor라 `getOps()`가 CPU 테이블을 돌려줘 device arm을 못 탔음 → `half[i].setContextData(src[i]->getContextData())`로 해결. 순서 변화: 그 경로의 `settle_act_stage_for_host()`(SVM map)가 이제 device arm보다 먼저 실행됨(Adreno fine-grain SVM에서는 drain과 동일; coarse-grain 장비 미검증).
- **검증** (Galaxy S26 R3CY70LV96T SM-S948U Android 16; 사용자 지시로 이 기기 사용. R5KL30G6MCX도 동일 결과): 하네스(= #4473의 device 모드 + 수정을 6d 위에 얹은 로컬 커밋, 브랜치 `wt/7a-verify`에 보존) host 21 pass/12 skip, gpu 32/33 — 7a 전 기준선과 동일. 추적: gemm 14케이스 50 step 모두 op의 arm 진입, 49 step 커널 처리. 변이(커널 없이 성공 주장): gemm 14개 전부 실패, per-row 18개 통과. x86 fp16: mha 33/33, dispatch 37/37. 커밋에는 하네스 미포함(#4473 소유).
- **기존 문제 발견**: `ohwi_gemm_long_prefill`이 device 모드에서 7a 전후 모두 0.149 실패(6d의 OHWI 케이스; op는 host 복사본 K라 거절하고 host arm이 틀린 값 → device 모드의 OHWI 쓰기 경로 문제, 7b에서 볼 것). 6d·#4478 본문에는 아직 안 적음.
- **PR #4479** (사용자 지시 "open 7a pr", 11-03 밤): 7a 1차 조각, #4478 위, 제목 `[Wait for #4478] [ComputeOps] Run MHACoreLayer's OpenCL flash and two_conv attention arms from ClComputeOps`. 머지 순서: #4462 → #4464 → #4465 → #4466 → #4469 → #4477 → #4478 → #4479 (#4473은 #4462 뒤 아무 때나).
- **장비 메모**: 태그별 디렉터리를 쓰는 `run_devtest_on_device.sh`는 `/data/local/tmp/refactor4456_1102/libc++_shared.so`가 있어야 한다(R3CY70LV96T·R5KL30G6MCX 모두 넣어 둠). Android 엔진은 meson의 custom command(ndk-build)로 빌드되므로 `ninja -C builddir && meson install --no-rebuild --skip-subprojects`로 증분 재빌드, 로그의 컴파일 줄은 `Compile++`.
- **다음 조각(7a-2)**: OpenCL concat 쓰기(`rope_inplace_f16_cl` K→cache slice, `gpu_copy_f16_cl` V)를 `ClComputeOps::kv_cache_write`로; 스테이징(q_cl/k_stage/undrained 평면) 소유권은 7b의 `KVCacheState`와 함께.

### 12.10 2026-11-04: 7a 2차 조각 (OpenCL RoPE·concat KV 쓰기) → #4479 둘째 커밋 fcb4cf5

- **옮긴 것**: `ClComputeOps::rotary_embedding`(FP16 SVM 활성값 in-place `rope_inplace_f16_cl`, drain)과 `ClComputeOps::kv_cache_write`(K를 `rope_inplace_f16_cl`로 cache 행에 직접 회전, V를 `gpu_copy_f16_cl`, 둘 다 drain; no-RoPE·OHWI·int8 쓰기는 host 유지). 게이트: NNTR_MHA_GPU, NNTR_NO_GPU_ROPE 아님, NNTR_MIN_PREFILL 행 또는 NNTR_MHA_GPU_DECODE 단일행, FP16 SVM, 표의 fp16 행이 위치를 덮음. op가 받는 `RotaryTable.cos_fp16/sin_fp16`은 레이어의 flat LUT와 같은 포인터라 커널의 상주 LUT 캐시(포인터 키)가 그대로 맞는다. 레이어: concat-RoPE device 블록 게이트에 `!plain_svm`(image arm·staging·cl_mem 핸들이 없으면 op로) 추가. **함정**: `NNTR_MHA_CLMEM`은 기본 ON이라 plain 조건에 넣으면 안 됨(스테이징·image에서만 의미).
- **중요한 사실**: Adreno에서는 gpu 컨텍스트가 `NNTR_KV_IMG_ATTN=1`을 기본으로 켠다(`nntrainer/cl_context.cpp:232`) → `imageAttnLayer()`가 kFast arm에서 모든 레이어에 true → gemm 레이어는 레이어의 **image 변형** RoPE 블록(7b 대상)을 그대로 탄다. 하네스에서 첫 케이스만 img=0인 것도 이 때문(컨텍스트 등록 전 finalize). plain 경로는 per-row 레이어, 또는 `NNTR_KV_IMG_ATTN=0`/image arm 없는 장비.
- **검증(R3CY70LV96T)**: image arm 기본(on): host 21/12skip, gpu 32/33(기존 OHWI 건). image arm off(`NNTR_KV_IMG_ATTN=0`): **33/33**(OHWI 건도 통과 → 그 실패는 image arm 문제, 7b). 추적(off): RoPE arm 63/63 step, 쓰기 arm 54/54 — RoPE 있는 FP16·스테이징 케이스 전부; int8·OHWI 쓰기는 host로 거절. 변이(K 비회전 쓰기, off): 그 15케이스 전부 실패, 나머지 18 통과. image arm on에서는 FP16 per-row 2케이스만 새 arm(8/8).
- **사고 2건과 교훈**: ① 추적 사이클의 `git checkout -- file`이 미커밋 op 코드를 지움 → 사이클 전에 반드시 커밋. ② `git cherry-pick -q`는 없는 옵션(usage 출력, 아무 일 안 함) → 그 사이 PR 본문·답글이 먼저 올라갔다가 바로 정정. push 뒤에는 항상 live head를 확인할 것.
- **브랜치**: `wt/7a-verify`(=7a 두 조각 + 하네스 + #4473 수정, 9c61d4d), `wt/7a-cl-attention`(PR, fcb4cf5). 폰 실행 스크립트는 tag 디렉터리마다 새로 만드니 `/data/local/tmp/refactor4456_1102/libc++_shared.so` 필요(두 폰에 있음).
- **다음 7b**: image arm(OHWI mirror, kvimg_view attention, staged Q/K, undrained 평면 bookkeeping)과 `KVCacheState`(Q2). 6d OHWI 케이스의 device 모드 실패도 거기서.

