# Candidate agent rules: group "accel" (opencl, qnn_htp)

Sources: `unit_opencl.json` (125 comments, 53 PRs, 15 reviewers), `unit_qnn_htp.json`
(4 comments, all from one unmerged PR #4236), the accelerator paths in `unit_causallm.json`,
and `git log refs/remotes/upstream/main --since=2026-03-01` over `nntrainer/opencl`,
`nntrainer/tensor/cl_operations`, `nntrainer/layers/cl_layers`, `nntrainer/qnn`,
`nntrainer/tensor/htp_backend` and `Applications/CausalLM/qnn` (158 OpenCL commits, about 25 QNN/HTP commits).
I checked every rule against upstream main code (`git show refs/remotes/upstream/main:<path>`).

Note on the data: `unit_opencl.json` holds comments on PRs back to #2759 (2024), so the
date window in the brief does not hold for this file. Older comments were weighed lower when
the code has moved on since.

## opencl

### R1. Keep the host and the device in step at every SVM handoff
- **Rule**: Before host code reads an SVM buffer that a kernel wrote, map it
  (`cl_svm_map_force(ptr, bytes, read_only)` / `enqueueSVMMap`). After host code writes
  one that a kernel will read, unmap it (`cl_svm_unmap_force`). Map gives the buffer
  to the host and unmap gives it back to the device, not the other way round. Decide
  whether a drain is needed from `CommandQueueManager::coarseGrainSVM()` /
  `kernelsMayBeInFlight()`, never from the vendor name.
- **Why**: a missing or reversed map makes the host read a stale shadow. The output
  reads fluently but is wrong, and nothing crashes (Adreno and Intel both hit this).
- **Evidence**: djeong20 https://github.com/nntrainer/nntrainer/pull/3408#discussion_r2274932570 ,
  djeong20 https://github.com/nntrainer/nntrainer/pull/3415#discussion_r2283963180 ("As I mentioned in #3408, the usage of Map and Unmap is incorrect"),
  dkjung https://github.com/nntrainer/nntrainer/pull/3265#discussion_r2203867147 ;
  commits 3152a8e29 (restore `cl_svm_map_force`, "the missing half of the SVM boundary"),
  526fde29e (host op read an FP32 activation before the kernel finished: 0.6B model printed newlines),
  7d17dc180 (drain gated on vendor left Adreno wrong: "gate on the capability"), b36c29666, c03a1d6b0.
- **Strength**: 3 PRs in review (djeong20 repeated it across PRs) and 5 maintainer fix commits (jijoongmoon).
- **Verify**: `meson test unittest_cl_residency unittest_opencl_kernels_blas` on a machine with a
  real OpenCL device. Run one GPU end-to-end model (Adreno and/or Intel) and compare tokens with
  the CPU engine.

### R2. Never key a cache on a tensor's data pointer
- **Rule**: Do not reuse a derived buffer (quantized activation, packed weight,
  staging copy) because the input has the same `data` address, even with the shape added
  to the key. Key it on the producer, or invalidate it on every write. If neither is
  provable, recompute.
- **Why**: the tensor pool recycles a slot as soon as its last reader runs, many times in one
  forward pass. The next call gets the same address with different data and silently
  uses the old result (a 35-layer model diverged at token ~10).
- **Evidence**: jayden0701 https://github.com/nntrainer/nntrainer/pull/4110#discussion_r3663514883 (raises exactly this);
  commits 895360807 (scope the shared-quant cache to one pass), fb1559feb ("refuse the
  shared-quant cache where no pass boundary is provable"), 168244f13 (drop the address-keyed
  reuse entirely: "the address on its own cannot carry either").
- **Strength**: 1 review PR and 3 maintainer commits (jijoongmoon) that state the reason outright.
- **Verify**: no direct unit test. Compare GPU and CPU tokens on a model with KV-shared blocks
  (see 168244f13 for the reproducer shape).

### R3. Make the host dispatcher enforce every shape the kernel assumes
- **Rule**: When a kernel assumes something about its inputs (float4/float8 width so
  `W % 4 == 0`, fixed private arrays such as `head_dim <= 256`, a minimum block width,
  `K % 64 == 0`), the host wrapper must check it. It then picks a tail or scalar kernel
  or falls back to the CPU. Never clamp a loop bound or drop a tail silently.
- **Why**: truncating `W / 4` or capping `key_idx < 512` gives plausible wrong numbers with no error.
- **Evidence**: EunjuYang https://github.com/nntrainer/nntrainer/pull/3812#discussion_r2986362423 and
  https://github.com/nntrainer/nntrainer/pull/3812#discussion_r2986362700 ,
  myungjoo https://github.com/nntrainer/nntrainer/pull/3550#discussion_r2545173147 ("Is it ok to assume W % 4 == 0 always holds?"),
  jayden0701 https://github.com/nntrainer/nntrainer/pull/4111#discussion_r3663396854 (Intel block-read width, gate needs K >= 128),
  anupSnap21530358 https://github.com/nntrainer/nntrainer/pull/3759#discussion_r2922406089 ;
  commit cefcffb26 (FP32 RMSNorm dropped the `W % 4` tail. The FP16 wrapper already routed odd widths to a scalar kernel).
- **Strength**: 6 PRs (also EunjuYang #2759 r1835666277). Raised by maintainers myungjoo and EunjuYang.
- **Verify**: add an odd-width / non-multiple case to `unittest_opencl_kernels_*` and run it on a GPU.

### R4. A green build proves nothing about OpenCL kernels: build with OpenCL on and run them on a device
- **Rule**: Configure with `-Denable-opencl=true`, because the default build compiles none of the
  OpenCL sources. `.cl` files are compiled only at runtime by the driver, and no CI runner has
  a GPU. Before you claim a kernel works, run the matching `unittest_opencl_kernels_*` /
  `unittest_attention_kernels_cl` / `unittest_layers_*_cl` on a real device. Name that device
  in the PR self-evaluation. If you could not run it, say so.
- **Why**: kernels with syntax errors (missing semicolons) were submitted twice with
  every check green. CI's OpenCL job only compiles and links.
- **Evidence**: djeong20 https://github.com/nntrainer/nntrainer/pull/3326#discussion_r2209922096 ("Please test the code, confirm it works"),
  myungjoo https://github.com/nntrainer/nntrainer/pull/3339#discussion_r2480676173 ("No semicolons above...");
  upstream `.github/workflows/ubuntu_opencl_build.yml` header ("does NOT run the test suite -- GitHub
  runners have no OpenCL device"); commit 2a4b55b58 ("no CI leg can check it ... no runner has a GPU").
- **Strength**: 2 PRs plus 2 maintainers (myungjoo, djeong20), backed by a maintainer commit and a CI file.
- **Verify**: `meson setup build -Denable-opencl=true && ninja -C build && meson test -C build unittest_opencl_kernels_blas`
  (plus the other OpenCL targets in `test/unittest/meson.build`) on a GPU host.

### R5. After editing a `.cl` file, regenerate the embedded kernel strings before an ndk-build
- **Rule**: The `.cl` files are embedded as C++ strings by `configure_file` in
  `cl_kernels/meson.build`. The Android ndk-build flow never re-runs meson, so run
  `tools/regen_cl_kernels.sh <builddir>` after every `.cl` edit. Register a new `.cl` in
  `cl_op_kernels` (FP16 variants inside the `enable-fp16` block). Do not add the generated
  per-kernel headers to the install list.
- **Why**: a stale embedded string means the device silently runs the previous kernel
  ("costs an afternoon each time"). Installing the generated `swiglu.h` once shadowed
  CausalLM's own `swiglu.h` and broke the app build.
- **Evidence**: commits 387316753 (adds `tools/regen_cl_kernels.sh`, gives the reason),
  23a0b0c17 (install-list fix: missing OpenCL headers broke a from-scratch Android app build,
  and installed generated headers collided), both by jijoongmoon. jaemini-shin reviewed the
  `cl_kernels/meson.build` change in https://github.com/nntrainer/nntrainer/pull/4103#discussion_r3619020089 .
- **Strength**: maintainer-stated in 2 commits. No recurring review thread, because the failure is silent.
- **Verify**: `tools/regen_cl_kernels.sh build` prints `regen:` lines only for the changed kernels.

### R6. Put new GPU work in the `ClComputeOps` table, not in a new `*_cl` layer
- **Rule**: Implement a GPU op as an entry in `nntrainer/tensor/cl_operations/cl_compute_ops.cpp`,
  reached through `input.getOps()->...` from the backend-neutral layer. Do not add a `layers/cl_layers/*_cl`
  fork that copies a whole layer to change its arithmetic. When `cl_context.cpp` registers a
  layer, close the OpenCL table by hand over every op that layer calls.
- **Why**: forks copied properties, finalize and shape checks just to change two lines.
  Layer unit tests run on the CPU table, so an op missing from the OpenCL table stays green in CI
  and throws on device.
- **Evidence**: commits c19080d92 (delete the OpenCL addition/SwiGLU forks), 397e2f640 (collapse the FC fork onto
  `ComputeOps::fc`), 2a4b55b58 (closure rule "written down because no CI leg can check it"),
  269813538, 0595ff509, c4694d2ab. All by jijoongmoon.
- **Strength**: maintainer convention carried out in 6+ commits. No review comments, because the rule was set by refactor.
- **Verify**: `meson test unittest_compute_ops_dispatch`, plus the `unittest_layers_*_cl` targets on a GPU.

### R7 (weaker). Gate vendor-specific features on the extension or capability, and keep every other GPU working
- **Rule**: Wrap vendor extensions in kernels with `#ifdef cl_intel_required_subgroup_size` /
  `#elif defined(cl_qcom_reqd_sub_group_size)` and give other devices a plain path. Make host-side
  vendor checks only for features that belong to that vendor (XMX/DPAS = Intel). Adreno and Intel
  are both primary targets.
- **Why**: an Intel-only pragma breaks Adreno compilation. A vendor-gated fix left Adreno wrong,
  and an XMX warning misreported Adreno as an Intel GPU.
- **Evidence**: djeong20 https://github.com/nntrainer/nntrainer/pull/3415#discussion_r2283958249 ,
  jayden0701 https://github.com/nntrainer/nntrainer/pull/4111#discussion_r3663396854 ; commits 7d17dc180, e46e45c54.
  The pattern is in use upstream in `cl_kernels/convert_block_q4_0.cl`.
- **Strength**: 2 PRs and 2 maintainer commits. It overlaps R1 on the SVM side.
- **Verify**: run the touched kernel's test on one Adreno and one Intel device. If you have only one, say which.

## qnn_htp

The evidence is thin. The 4 review comments all come from unmerged PR #4236 and none of them qualifies (loop
parallelization ideas, one unused macro, a reviewer note on `rm`/`wh` layout). Both rules below come from
upstream commits and the code, not from review threads.

### Q1. Never commit Qualcomm SDK files: build against the user's SDK
- **Rule**: Do not add QNN or HexKL SDK headers or sample sources to the tree. `nntrainer/qnn/jni/vendor/`
  is gitignored and generated at configure time by `nntrainer/qnn/jni/tools/update_qnn_version.py`
  from `-Dqnn-sdk-root` / `$QNN_SDK_ROOT`. To move the pinned SDK version, edit
  `EXPECTED_API_VERSION_MINOR` / `EXPECTED_SDK_VERSION_TEXT` there and update
  `docs/backend_guide/QNN_BUILD.md`. Include the HexKL headers only through
  `nntrainer/tensor/htp_backend/sdkl_compat.h`. Never use a project-wide `-Drestrict`.
- **Why**: the SDK files carry "Confidential and Proprietary - Qualcomm" licenses and cannot land upstream.
  A half-generated vendor tree was later trusted by the next configure.
- **Evidence**: commits 262d46762 (jijoongmoon, removes 43k lines of proprietary SDK, BYO-SDK), 96ec40781
  (jayden0701, "Per review, stop committing the proprietary Qualcomm files"), dabb95f1e (jijoongmoon,
  version gate made a precondition), 282ce1f36 (HexKL "not vendored", `sdkl_compat.h` as the single
  include point). `nntrainer/qnn/jni/.gitignore` on upstream main states the policy.
- **Strength**: 4 commits, 2 of them by a maintainer. This is a licensing rule, so it is high-stakes despite the thin review data.
- **Verify**: `git status --porcelain nntrainer/qnn/jni/vendor` is empty after
  `meson setup build -Denable-npu=true -Dqnn-sdk-root=<sdk>`.

### Q2. Make accelerator bring-up failure throw and fall back to the CPU, never exit or crash
- **Rule**: In QNN/HTP (and OpenCL) context initialization, check every init return value and
  every `dlsym` / resolved symbol. On failure, throw into the context's existing catch so the
  engine installs `get_cpu_ops()` or skips the plugin. Never `exit()` and never call through a null
  function pointer. Code that runs on every engine must not demand an accelerator context.
- **Why**: a binary built with `-Denable-npu=true` hit SIGSEGV at `Engine::Global()` on every host that has no
  QNN SDK, and that includes GPU-only APKs.
- **Evidence**: commits 45c0fef2a (jijoongmoon, discarded `init()` return value plus a null symbol caused the SIGSEGV),
  4e845d8db (jayden0701, `exit(-1)` replaced by throw), 282ce1f36 (HtpContext degrades to the CPU table, see
  `nntrainer/htp_context.cpp` catch block), 3e38eabc9 and 70c8adb90 (the same rule on the OpenCL side).
- **Strength**: 5 commits across QNN, HTP and OpenCL, 3 of them by a maintainer. It also spans the opencl unit.
- **Verify**: run any CPU model from a build with `-Denable-npu=true` (or `-Denable-htp=true`) on a host
  with no SDK libraries on the library path. It must run, and log only a warning.

### Applications/CausalLM/qnn
Only one accelerator-path comment exists in `unit_causallm.json` (jaemini-shin #4047 r3534949866, leftover
lines in `qnn/meson.build`), and it is not a rule. Other comments on Android build scripts are CausalLM-specific
and are left to the CausalLM agent.

## Global (belongs in root AGENTS.md, not a unit file)

### G1. End every file with a newline, including `.cl`, `meson.build`, `.json`, `.py` and `.gitignore`
- **Rule**: Finish every new or edited file with exactly one trailing newline. clang-format covers only
  C/C++, and reviewers flag the rest by hand.
- **Why**: this is the most repeated review nit in the data set. jaemini-shin: "add a new line to pass cpp linter".
- **Evidence**: jaemini-shin https://github.com/nntrainer/nntrainer/pull/4103#discussion_r3619020089 ,
  jijoongmoon https://github.com/nntrainer/nntrainer/pull/3759#discussion_r2915119975 ,
  skykongkong8 https://github.com/nntrainer/nntrainer/pull/3299#discussion_r2191609491 ,
  baek2sm https://github.com/nntrainer/nntrainer/pull/3548#discussion_r2768031438 ,
  jijoongmoon https://github.com/nntrainer/nntrainer/pull/3984#discussion_r3420308656 .
- **Strength**: about 14 PRs across 7 units, from 6 maintainers.
- **Verify**: `git diff --name-only <base> | xargs -I{} sh -c 'tail -c1 "{}" | od -An -c | grep -q "\\\\n" || echo "no EOF newline: {}"'`

### G2 (for the cpu_backend / tensor agents to confirm). Use `_FP16` and the `ENABLE_FP16` compile-time guard
- **Rule**: In backend-neutral, OpenCL and test code, spell half precision `_FP16` and guard it with
  `#ifdef ENABLE_FP16`, not with a runtime boolean. `__fp16` belongs only in ARM-NEON-only files.
  On Windows, `_FP16` may be `uint16_t`.
- **Evidence**: EunjuYang https://github.com/nntrainer/nntrainer/pull/2759#discussion_r1835667041 ,
  jaemini-shin https://github.com/nntrainer/nntrainer/pull/4000#discussion_r3526749987 ,
  jaemini-shin https://github.com/nntrainer/nntrainer/pull/3820#discussion_r3205746483 ,
  djeong20 https://github.com/nntrainer/nntrainer/pull/3456#discussion_r2485601791 ;
  counterpoint skykongkong8 https://github.com/nntrainer/nntrainer/pull/2975#discussion_r1978427863 (`__fp16` is fine in NEON-only files).
- **Strength**: 5 PRs across 5 units.

## Rejected
- Compile or register a kernel only at init (EunjuYang #3083, jijoongmoon #3759): already solved in code. `registerClKernel`
  caches, there is a disk kernel cache (75a7ff086), and the kernel ring (e6f9c7383) now *wants* a per-dispatch lookup. The tree mixes both styles (`cl_layers` keep static pointer vectors), so there is no single rule to state.
- Merge the FP16 and FP32 `.cl` twins with macros, or share a common `.cl` header (EunjuYang #3812, anupSnap #3759): suggestions, not a convention. The tree keeps paired `_fp16.cl` files.
- Pass kernel tunables as `-D` compile options (djeong20 #3523, EunjuYang #3812): suggestion-level only, and the `compile_options` parameter of `registerClKernel` is easy to find.
- Work-group size is part of the kernel contract / global size must be a multiple of local size (dkjung #3225, djeong20 #3515, EunjuYang #2759): questions and notes, not a stated rule.
- Add a new `.cl` instead of modifying an existing kernel (djeong20 #3491): single PR.
- `do { } while (0)` macros (myungjoo #3569): single PR, and it is generic C advice.
- Unused variables, includes or dead code, missing `@param`, typos (songgot, djeong20, EunjuYang; about 15 comments): generic, and doxygen is already checked.
- `ml_loge` instead of `printf`, and size_t overflow in size math (EunjuYang #3812): single PR, generic.
- Release GPU caches at model teardown (20061b51c, 1a0ed7f7b): one series, not seen in review.
- Q4_0 x4 vs x8 interleave on ARM for the GPU GEMM (526fde29e): single commit, very specific.
- HTP: parallelize the HMX loops (jijoongmoon #4236 x2): performance suggestion on one unmerged PR.
- QNN: keep the context alive across unload/reload (3c7ff1b44): single commit.

## Unit verdict
- **opencl**: file worthwhile, 6 rules (R1 to R6) plus 1 weaker one (R7). Target directory `nntrainer/tensor/cl_operations/`,
  with `nntrainer/opencl/` pointing to it (or one shared file, since R1 and R4 apply to both).
- **qnn_htp**: `nntrainer/qnn/` file worthwhile but small, 2 rules (Q1, Q2). For `nntrainer/tensor/htp_backend/`, skip:
  the evidence is too thin (4 comments from one unmerged PR, 1 skeleton commit). Q1's `sdkl_compat.h` line can live
  in the qnn file or in a short note next to it.
- **Applications/CausalLM/qnn**: skip, no accelerator rules found.
