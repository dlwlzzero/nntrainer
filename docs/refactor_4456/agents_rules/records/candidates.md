# AGENTS.md 규칙 후보 (통합본, 2026-11-05)

근거: upstream nntrainer 리뷰 코멘트 4000개 (2024-11 ~ 2026-11) 중 사람 리뷰어
최상위 코멘트 1,678개 + upstream main 640ac6e0e의 fix/revert 커밋.
각 규칙의 근거 링크/커밋은 괄호 안 원본 ID로 `rules_<group>.md`에서 찾으면 됨.
빌드/테스트는 아무것도 돌리지 않음. Verify 명령은 제안일 뿐.

## 0. 루트 AGENTS.md 정정 (cross: Root AGENTS.md corrections)

1. `check_count`는 테스트가 아니라 승인 리뷰어 수를 셈. "테스트 추가/삭제 시 count 갱신" 문장 삭제.
2. `pr-desc`는 PR 본문을 검증하지 않음 (`ai-describe` 라벨 시 요약 코멘트만).
3. minja는 `Applications/CausalLM/third_party/`. "수정 금지" 목록에 이 폴더와 `xgrammar` 서브모듈 추가.
4. format gate는 `git diff --check`로 모든 파일의 trailing whitespace/EOF 검사 (`.github/`, third_party 제외).
5. static.check 실제 요구사항 명시: doxygen 헤더, EOF newline, 실행 비트 금지, shellcheck, pylint, cppcheck, 30 MiB 제한.
6. 모든 게이트가 모든 PR에서 도는 게 아님 (check-rebuild, CausalLM은 fp16-off leg에서만).
7. Android 테스트는 `test/jni/Android.mk` 별도 목록, `tools/android_test.sh`로만 빌드, PR CI에 없음.
8. `clang-format-14 -i` (파일 전체) 대신 `git clang-format-14 <base>` (core G3).
9. (upstream 버그) `hardcoded-path.sh`는 절대 실패하지 않음. 루트가 CI가 잡는다고 암시하지 말 것.

## 1. 루트 AGENTS.md에 추가할 공통 규칙 (여러 그룹에서 중복 발견, 병합)

- R-G1 **EOF newline + trailing whitespace 없음**, 모든 텍스트 파일. (accel G1, apps G1, cross G2; 14+ PR, 6 메인테이너)
- R-G2 **SPDX + doxygen 헤더** `@file @brief @author @bug`, `/**` 주석, 복사한 파일은 `@file`/include guard 재작성. (core G4, apps G2, cross G1, tensor G1/G2; 10+ PR)
- R-G3 **바꿀 줄만 바꿈**: 무관한 포맷/재정렬/주석 삭제, 서브모듈 포인터 이동 금지. (apps G3, cross G7, tensor G2; 10+ PR)
- R-G4 **dtype 분기는 모든 타입 처리 또는 throw**, `_FP16`은 `#ifdef ENABLE_FP16`로, `_FP16`이 wrapper class일 수 있음. (core G1, accel G2, cross G5, tensor CB6; 5+ PR)
- R-G5 **설치 헤더/라이브러리 변경 시 `debian/*.install` + `packaging/nntrainer.spec` 같은 커밋에서 수정.** (cross O1, tensor G3; 3 PR + fix 커밋 6개)
- R-G6 **meson 플래그는 가장 좁은 조건으로** (compiler id / cpu_family / system), `meson_options.txt` 기본값은 주제가 아니면 안 바꿈, 새 백엔드 옵션은 off. (cross O2/O3, tensor G6)
- R-G7 **개인/절대 경로 커밋 금지.** (cross G3)
- R-G8 **로그는 `ml_log*`**, 헤더에 `<iostream>` 없음. (core G5)
- R-G9 **Windows**: `(std::min)(a,b)`, GCC 전용 attribute 가드. (cross G6, tensor CB7 일부)
- R-G10 **크기 곱셈 전에 캐스트** (CodeQL). include guard 사용, `#pragma once` 아님. (tensor G4/G5)
- 약함/선택: 주석은 짧게, 에이전트 내레이션 금지 (core G2); 영어로 작성 (cross G4); member shadowing 금지 (core G6).

## 2. 루트에 넣을 포인터 표 (새로 추가)

| 이 경로를 고치기 전에 | 읽을 파일 |
|---|---|
| `Applications/CausalLM/` | `Applications/CausalLM/AGENTS.md` |
| `nntrainer/` | `nntrainer/AGENTS.md` (+ 하위 파일) |
| `nntrainer/tensor/` | `nntrainer/tensor/AGENTS.md` |
| `nntrainer/tensor/cpu_backend/` | `nntrainer/tensor/cpu_backend/AGENTS.md` |
| `nntrainer/tensor/cl_operations/`, `nntrainer/opencl/` | `nntrainer/opencl/AGENTS.md` |
| `nntrainer/layers/` | `nntrainer/layers/AGENTS.md` |
| `test/` | `test/AGENTS.md` |
| `api/` | `api/AGENTS.md` |

## 3. 단위 파일 후보

### Applications/CausalLM/AGENTS.md — 8개 (apps C1–C8), 하위 폴더 전부 커버
- C1 weight 파일은 위치 기반: converter 쓰기 순서 = 레이어 생성 순서, `createMlp`(up, gate, down) 순서 불변. revert 2회.
- C2 새 `.cpp`/모델 폴더는 매니페스트 4곳 (in-tree meson, `app_build/meson.build`, `jni/Android.mk`, `model_registry.cpp`). fix 커밋 5개.
- C3 `incremental_forwarding(from,to)`: KV는 절대 위치, step 텐서는 `pos - from`, prefill 판정 `!from || (to-from) > 1`.
- C4 dtype/arch 분기를 함께 유지, CI는 x86 fp16-off만 빌드하므로 FP16/ARM은 직접 빌드.
- C5 `parallel_for`는 한 단계만, 프로세스 전역 스레드 설정 금지.
- C6 공개 트리에는 공개 모델만 (사내/다운스트림 모델명 금지).
- C7 `res/<model>/`에는 converter, `nntr_config.json`, README만.
- C8 `BATCH_SIZE` 존중: 배치별 인덱싱 또는 batch > 1 명시 거부.

### nntrainer/tensor/cpu_backend/AGENTS.md — 8개 (tensor CB1–CB8), arch 하위폴더 전부 커버
- CB1 dispatcher는 분기만, 커널은 ISA/정밀도별 파일에.
- CB2 새 함수는 arm/x86/fallback 세 헤더에 같은 시그니처로 선언+구현. fallback은 CI 미빌드.
- CB3 모든 SIMD fast path에 fallback `else`.
- CB4 arm/ 코드는 32-bit armv7 (Tizen)에서도 빌드.
- CB5 baseline 넘는 ISA 확장은 meson 옵션 + ACLE 매크로, `target` attribute 금지.
- CB6 → 루트 R-G4로 이동. CB7 → 일부 루트 R-G9, kernel 특화 부분만 남김.
- CB8 BLAS 호출은 `cblas_interface` 안, `USE_BLAS` 뒤.

### nntrainer/tensor/AGENTS.md — 4개 (tensor TC1–TC4)
- TC1 Tensor 타입은 `getOps()` (ComputeOps)로만 커널 호출, cpu_backend 밖에 arch `#ifdef` 없음.
- TC2 새 dtype은 모든 switch 지점과 양 정밀도 커버 (기존 형제 dtype grep).
- TC3 양자화 weight: scale은 `width()` 축, payload 레이아웃은 기존 helper만.
- TC4 mmap/FSU: 런타임 page size, Windows 분기, 실패 확인.

### nntrainer/opencl/AGENTS.md — 6개 (accel R1–R6), cl_operations도 이 파일
- R1 SVM: host 읽기 전 map, device 사용 전 unmap; drain 판단은 vendor가 아니라 SVM capability.
- R2 tensor data pointer를 캐시 키로 쓰지 않음 (pool이 주소 재사용).
- R3 host dispatcher가 커널의 shape 가정을 검사 → tail 커널 또는 CPU fallback, 나머지 잘라먹기 금지.
- R4 `-Denable-opencl=true`로 빌드하고 실 GPU에서 테스트 (CI에 GPU 없음, `.cl`은 런타임 컴파일).
- R5 `.cl` 수정 후 ndk-build 전에 임베드 문자열 재생성.
- R6 새 GPU 연산은 `ClComputeOps` 테이블에, 새 `*_cl` 레이어 아님.

### nntrainer/layers/AGENTS.md — 7개 (core L1–L7)
- L1 `parallel_for`에 실제 작업 수 전달, 스레드 수로 미리 나누지 않음, dtype 분기는 람다 밖.
- L2 FP16 norm/reduction은 FP32로 누적.
- L3 `layer_devel.h` ABI: 새 virtual은 끝에 추가, `PluggedLayer`에 forward.
- L4 수치 연산은 Tensor ops/compute backend에, 기존 커널 재사용.
- L5 CausalLM 레이어 승격 시 `namespace nntrainer`, `custom_` 제거 등 1급 레이어로.
- L6 forward 간 유지되는 버퍼는 context를 통해, 멤버 변수 아님.
- L7 `skip_prefill` 레이어는 공용 `is_prefill` 판정 사용.

### nntrainer/AGENTS.md — 6~7개 (core M1, C1, C2, X1–X3 + accel Q1)
- M1 weight 저장/로드 시 공유 텐서는 한 번만, offset은 `getMemoryBytes()`.
- C1 ONNX op 매핑은 스펙 준수 또는 throw.
- C2 레이어별 경로에서 `NetworkGraph::getLayerNodes()` 호출 금지 (복사 비용).
- X1 싱글턴: `.cpp`에 `Global()` + `NNTRAINER_SINGLETON_API`, `std::call_once`.
- X2 런타임 생성 파일은 `resolveUserDataDir()` 아래, CWD 금지, throw 금지.
- X3 + Q2 라이브러리/가속기 초기화 실패는 throw (+ CPU fallback), `exit()` 금지.
- Q1 Qualcomm SDK 파일 커밋 금지.

### test/AGENTS.md — 7개 (cross T1–T6 + CI C2)
- T1 테스트는 빌드 게이트가 맞는 파일에.
- T2 새 테스트는 `test/unittest/meson.build` + `test/jni/Android.mk` 둘 다, `tools/android_test.sh` 직접 실행.
- T3 커널은 `cpu_backend.h`로, BLAS/arch 헤더 직접 include 금지.
- T4 모든 TEST는 assert, 출력 없음.
- T5 golden/fixture는 in-tree generator로, `packaging/*.tar.gz`에.
- T6 기존 테스트를 고쳐 맞추지 말고 새 케이스 추가, 실제 진입점 테스트.
- CI-C2 `required_tests` 목록 (`ubuntu_clean_meson_build.yml`)과 동기화.

### api/AGENTS.md — 3개 (cross A1–A3)
- A1 기존 public enum 값 번호/이름 변경 금지, append만.
- A2 새 레이어 타입은 `nntrainer-api-common.h` + `LayerType` alias, 내부 전용은 1000+.
- A3 새 public 메서드 전에 기존 API 확장 검토.

## 4. 만들지 않는 것
- `htp_backend/`, `nntrainer/qnn/` 별도 파일: 근거 부족 (Q1/Q2는 `nntrainer/AGENTS.md`로).
- `Applications/` 기타 샘플 앱, `docs/`, optimizers/dataset/mem_allocator: 규칙 0~1개.
- `.github/AGENTS.md`: 규칙 2개 (CI-C1은 빈도 낮아 보류, C2는 test로).

## 5. 확인 안 된 것
- Qwen2/Qwen3/LFM2 `FP32MatchesHFReference`는 87f5fea19 이후 fixture 재생성 전까지 실패한다는 커밋 메시지. 미실행.
- 모든 Verify 명령 미실행.
