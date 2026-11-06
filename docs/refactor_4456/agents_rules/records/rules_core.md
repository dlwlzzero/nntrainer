# Candidate agent rules — group "core"

Sources: unit_layers.json (182 comments / 79 PRs), unit_models.json (82 / 36),
unit_compiler_graph.json (49 / 27), unit_nntrainer_misc.json (161 / 60), plus
`git log refs/remotes/upstream/main --since=2026-03-01` fix/revert commits on
these paths. Every rule below was checked against `refs/remotes/upstream/main`
(640ac6e0e). URL prefix `PR` = `https://github.com/nntrainer/nntrainer/pull/`.

## layers (nntrainer/layers/)

### L1. Pass the natural task count to `ThreadManager::parallel_for`
- **Rule**: Call `tm.parallel_for(0, rows, ...)` with the real number of
  independent work items (rows, b*c, ...). Do not pre-chunk by
  `getComputeThreadCount()`, and pick the dtype/kernel branch *outside* the
  lambda (one `parallel_for` per branch).
- **Why**: `ThreadManager::parallelize()` already splits `[begin,end)` across
  workers (thread_manager.h:262ff); chunking by thread count makes load
  imbalance, and a branch in the lambda sits on the hot path.
- **Evidence**: jaemini-shin PR4094#discussion_r3619004159,
  PR4095#discussion_r3619024321, PR4096#discussion_r3618904580,
  PR4096#discussion_r3618911117.
- **Strength**: 3 PRs, same regular reviewer, phrased as "Please do not bound
  the loop index to thread count". Older code (`llm/tie_word_embedding.cpp:604`)
  still chunks by thread_num, so an agent copying it gets this wrong.
- **Verify**: `meson test unittest_thread_manager unittest_layers`; perf only on
  device.

### L2. FP16 normalization/reduction paths accumulate in FP32
- **Rule**: In `ENABLE_FP16` paths of norm/reduction layers use
  `rms_norm_wrt_width_fp32_intrinsic` (or an FP32 accumulator); never
  `rms_norm_wrt_width_fp16_intrinsic(float*)`.
- **Why**: FP16 sum-of-squares overflows on real inputs; the float* overload
  now throws `"deprecated due to overflow in fp16"` at runtime
  (neon_impl_fp16.cpp:2320), so the bug shows up only on ARM+FP16 builds.
- **Evidence**: commits ba4f3cc8c (Jungwon-Lee), 0983f8070 (jayden0701),
  15c9a27d5; jaemini-shin suggested the fp16 intrinsic in
  PR3890#discussion_r3144828970 before the overflow was found.
- **Strength**: 3 fix commits; code carries "DO NOT USE
  rms_norm_wrt_width_fp16_intrinsic" (layer_normalization_layer.cpp:289).
- **Verify**: `meson test unittest_layers` with `-Denable-fp16=true` on an ARM
  build; x86 never hits the throwing overload.

### L3. Keep `layer_devel.h` ABI-stable: append virtuals, forward them in `PluggedLayer`
- **Rule**: New virtuals on `Layer` go at the end of the class, never in the
  middle; every new virtual that the graph calls must also get a forwarding
  override in `nntrainer/layers/plugged_layer.h`.
- **Why**: `layer_devel.h` is installed (layers/meson.build:74); inserting a
  virtual shifts vtable slots so out-of-tree plugin `.so`s dispatch to the
  wrong function, and a non-forwarded virtual silently returns the base
  default for every plugin layer.
- **Evidence**: jaemini-shin PR4458#discussion_r4398008493,
  PR4458#discussion_r4398008386; Seunghui98 PR4458#discussion_r4408084194.
- **Strength**: 1 PR, two reviewers, stated as a correctness/ABI rule. Upstream
  now has `getInputLifespan`/`getInputGradLifespan` last in the class
  (layer_devel.h:603,612) and forwarded in plugged_layer.h:140,147.
- **Verify**: `meson test unittest_layers` (plugin tests in
  `unittest_layers_impl`/plugin layer tests); review `git diff` of
  layer_devel.h for virtuals added above existing ones.

### L4. Put numeric work in Tensor ops / compute backend, reuse existing kernels
- **Rule**: Before writing a loop, quantization step, dtype-specific copy, or
  activation in a layer, look for the Tensor method or `cpu_backend` kernel
  (e.g. `copyData`, `divide`, `rms_norm_wrt_width_*`, activation ops) and call
  it; if missing, add it to the tensor/backend, not to the layer.
- **Why**: Layers are dtype-agnostic; hand-rolled code in a layer misses SIMD,
  duplicates kernels and must be edited again when the backend gains
  acceleration.
- **Evidence**: jijoongmoon PR4085#discussion_r3591760352 (quantization belongs
  in tensor), PR4085#discussion_r3591748083 (activation reimplemented);
  djeong20 PR3053#discussion_r2017886528 (move to `UIntTensor::copyData`);
  myungjoo PR2912#discussion_r1943023431 (use `Tensor::divide`); jaemini-shin
  PR3890#discussion_r3144831273 (existing intrinsic).
- **Strength**: 5 PRs, 4 maintainers.
- **Verify**: none mechanical; review.

### L5. When promoting a CausalLM layer into `nntrainer/layers`, make it a first-class nntrainer layer
- **Rule**: Move it to `namespace nntrainer`, drop `custom_`/"custom" from file,
  class and `@file` names (custom is reserved for user-plugged layers), fix
  `@file` to the new filename, and strip debug code.
- **Why**: Copied files keep `namespace causallm` and `custom_*` names, which
  reviewers reject line by line.
- **Evidence**: EunjuYang PR2752#discussion_r1847563541; Jungwon-Lee
  PR3873#discussion_r3179143671, PR3874#discussion_r3179189836,
  PR3865#discussion_r3179173114; jaemini-shin PR3847#discussion_r3159327860.
- **Strength**: 5 PRs. Upstream `nntrainer/layers/llm/*.h` are all
  `namespace nntrainer` and no file carries "custom".
- **Verify**: `git grep -n "causallm\|custom" nntrainer/layers/`.

### L6. Persistent buffers go through the context, not layer members
- **Rule**: A tensor that must live across forward calls (tables, caches,
  constants) is requested with `context.requestTensor(...)` /
  `requestWeight(...)` in `finalize()` and accessed by index; do not hold it as
  a `Tensor` member.
- **Why**: Only context-requested tensors are planned by the memory pool and
  guaranteed valid during forwarding.
- **Evidence**: jijoongmoon PR4376#discussion_r4145349889,
  PR4376#discussion_r4145368126.
- **Strength**: 1 PR, maintainer stated the reason outright;
  positional_encoding_layer.cpp:48 follows it.
- **Verify**: `meson test unittest_layers` (positional encoding golden test).

### L7. `skip_prefill` layers use the shared `is_prefill` test and skip consistently
- **Rule**: In `incremental_forwarding`, compute
  `bool is_prefill = !from || (to - from) > 1;` (only `lm_head` /
  `tie_word_embedding` keep `!from` on purpose). A skipped layer leaves its
  output unwritten, so every consumer on that path must also have
  `skip_prefill`.
- **Why**: `!from` alone misses multi-turn prefill; a non-skipping consumer
  reads stale output and corrupts the KV cache silently.
- **Evidence**: commit 1a80cfa35 (jayden0701); EunjuYang
  PR3799#discussion_r2992843240, PR3799#discussion_r2992844179.
- **Strength**: 1 PR + 1 fix commit across 11 layers. Matches
  fc_layer.cpp:338, addition_layer.cpp:57, multiply_layer.cpp:38.
- **Verify**: `meson test unittest_models` (CausalLM gemma4 multi-turn tests).

## models (nntrainer/models/)

### M1. Weight file layout: dedupe shared tensors and size by `getMemoryBytes()`
- **Rule**: Any loop over graph weights that saves, loads, or assigns file
  offsets must skip tensors already seen (dedupe on `&getVariableRef()`, as
  `visited_weights` does) and advance offsets by `getMemoryBytes()` (includes
  quantization scales), matching what `save()` writes.
- **Why**: Tied weights (TieWordEmbedding / lm_head) read or counted twice
  shift every later offset; using element bytes without scales breaks
  quantized files.
- **Evidence**: commit 628f6c1a7 (EunjuYang); EunjuYang
  PR3773#discussion_r2986438779; djeong20 PR3526#discussion_r2445351204;
  EunjuYang PR3004#discussion_r1995259888; jijoongmoon
  PR3088#discussion_r2034254436 ("align with Tensor saving").
- **Strength**: 4 PRs + 1 fix commit, maintainers.
- **Verify**: `meson test unittest_nntrainer_save_with_dtype unittest_models`.

## compiler_graph (nntrainer/compiler/, nntrainer/graph/)

### C1. ONNX op mapping follows the ONNX spec or throws
- **Rule**: When mapping an ONNX op (interpreter) or implementing its layer
  (gather/slice/concat), handle every spec case — N inputs, every axis
  including 0 and negatives, input dtypes such as INT64 — or throw a
  `std::runtime_error` naming the unsupported case. Never hard-code `axis=3`
  or two inputs, and keep the index arithmetic identical in
  forwarding/calcDerivative.
- **Why**: Silent assumptions import models that run with wrong results.
- **Evidence**: myungjoo PR3573#discussion_r2545075787 (Concat is N-ary),
  PR3573#discussion_r2545064305 (static axis=3), PR3574#discussion_r2545000438
  (INT64->FP32), PR3574#discussion_r2544364528 (gather axis cases),
  PR3338#discussion_r2508775025.
- **Strength**: 3 PRs, maintainer. Code now does this
  (onnx_interpreter.cpp:215 "only supports `concat` when there are two
  inputs", gather_layer.cpp:30).
- **Verify**: `meson test unittest_interpreter`.

### C2. Do not call `NetworkGraph::getLayerNodes()` in per-layer paths
- **Rule**: Iterate with `cbegin()/cend()` or cache the vector once;
  `getLayerNodes()` builds a new `std::vector<shared_ptr>` every call.
- **Why**: Called per layer (e.g. prefetch) it becomes an O(N^2) copy with
  refcount traffic on the inference path.
- **Evidence**: myungjoo PR3469#discussion_r2509093229 ("Performance Bug");
  songgot PR3469#discussion_r2336023336.
- **Strength**: 1 PR, maintainer stated. Verified: network_graph.cpp:673
  returns a copy.
- **Verify**: none; review.

## nntrainer_misc

### utils/ + top-level engine/context files

#### X1. Singletons: out-of-line `Global()` with `NNTRAINER_SINGLETON_API`, lazy init via `std::call_once`
- **Rule**: A process-wide singleton defines `Global()` in a `.cpp` (declared
  `static NNTRAINER_SINGLETON_API T &Global();`), never an inline header body;
  lazy members and registries are initialised through `std::call_once` /
  the existing funnels (`ensureComputeOps()`), and re-registration is a no-op.
- **Why**: Inline `Global()` is instantiated per DLL on MSVC (two thread pools,
  two engines); unguarded lazy init races.
- **Evidence**: commits d7835f6fd (jijoongmoon), dc996d2b5 (jayden0701);
  myungjoo PR3564#discussion_r2635195907, songgot PR3564#discussion_r2532338523,
  jayden0701 PR4162#discussion_r3682140099.
- **Strength**: 2 PRs + 2 fix commits; pattern in utils/singleton.h,
  thread_manager.h:158, engine, app/cl contexts.
- **Verify**: `meson test unittest_thread_manager unittest_nntrainer_appcontext`;
  real check is the Windows MSVC CI job.

#### X2. Runtime-written files go under `resolveUserDataDir()`, never the CWD, and never throw
- **Rule**: Logs, kernel caches and similar files resolve their directory with
  `resolveUserDataDir()` (util_func.h:484) with an `NNTR_*_DIR` override where
  "" / "off" disables; use non-throwing `std::filesystem` overloads and degrade
  instead of aborting.
- **Why**: CWD-relative paths crashed embedders started from unwritable
  directories and let other users plant cache binaries.
- **Evidence**: commits 484be98f4, 4de89e037 (both jijoongmoon).
- **Strength**: 2 maintainer commits, no review thread.
- **Verify**: `meson test unittest_util_func`.

#### X3. Library init failures throw; never `exit()` or ignore a return code
- **Rule**: Backend/context initialisation checks every return code and throws
  (caught by `Engine::add_default_object()`); no `exit()`/`abort()` in library
  code.
- **Why**: Ignored init results caused SIGSEGV at `Engine::Global()` on hosts
  without the SDK; `exit()` kills the embedding app.
- **Evidence**: commits 45c0fef2a (jijoongmoon), 4e845d8db (jayden0701);
  myungjoo PR3567#discussion_r2540878992; djeong20
  PR3453#discussion_r2315301142.
- **Strength**: 2 PRs + 2 fix commits.
- **Verify**: `meson test unittest_nntrainer_appcontext`.

### optimizers/, dataset/, mem_allocator, cl_*/cuda_* top-level files
No rule qualifies (single-PR comments; cl_*/cuda_* belong to the OpenCL/CUDA
units).

## Global (belongs in root AGENTS.md, not a unit file)

### G1. Every data-type branch handles all types or throws
- **Rule**: Each `switch`/`if` on `TensorDim::DataType` either implements a
  type or throws; FP16 code goes in `#ifdef ENABLE_FP16 ... #else throw
  std::invalid_argument("Error: enable-fp16 is not set"); #endif`. No empty
  `case:` / `break`, and no `getValue()` / `getData()` without the template
  type on non-FP32 tensors.
- **Why**: Empty branches return uninitialised output (HalfTensor::dot Q4_0),
  and default-float access misreads quantized tensors.
- **Evidence**: commits caa2aa12c (jijoongmoon), ecc635617 (EunjuYang);
  jijoongmoon PR3088#discussion_r2034251091; EunjuYang
  PR3751#discussion_r2856610158, PR3096#discussion_r2050372561; djeong20
  PR2804#discussion_r1865127881.
- **Strength**: 5+ PRs + 2 fix commits.
- **Verify**: build with `-Denable-fp16=false` and `true`; `meson test
  unittest_layers unittest_models`.

### G2. Write terse comments; no agent narration or plan tags
- **Rule**: Comments state what the code does or a non-obvious why in one or two
  lines. No task/plan IDs (`T10`, `T11`), no design-doc prose, no comments
  unrelated to the change.
- **Why**: Reviewers spot and strip agent-written comment walls line by line.
- **Evidence**: jaemini-shin PR4105#discussion_r3662290448,
  PR4042#discussion_r3526281406; Seunghui98 PR4106#discussion_r3663112642;
  jayden0701 PR4108#discussion_r3662887473.
- **Strength**: 4 PRs, 3 reviewers, explicitly about AI-written comments.
- **Verify**: `git diff <base> | grep -nE '\(T[0-9]+\)'`.

### G3. Format only changed lines; never clang-format `meson.build`
- **Rule**: Use `git clang-format-14 <base-sha>` rather than
  `clang-format-14 -i` on whole files; a whole-file reformat is a separate
  commit; never run clang-format on `meson.build`.
- **Why**: Whole-file formatting hides the real change in reformatted lines.
  Root AGENTS.md currently suggests `clang-format-14 -i <files>` first, which
  causes this.
- **Evidence**: EunjuYang PR3813#discussion_r2986317489,
  PR3212#discussion_r2113334712; songgot PR3560#discussion_r2521945319.
- **Strength**: 3 PRs.
- **Verify**: `git diff --stat <base>` shows only intended hunks.

### G4. Doxygen blocks that pass `doxygen-tag.sh`
- **Rule**: New files carry a `/** @file <name> @date @brief @see @author @bug */`
  header (keep `@file` in step with renames); every declaration in a header
  gets `/** @brief ... @param ... @return */` with params matching the
  signature; use `/** */`, not `///`, and lowercase tags.
- **Why**: static.check runs `doxygen-tag.sh` on changed files; reviewers also
  flag stale `@param`s.
- **Evidence**: jijoongmoon PR3088#discussion_r2034261348,
  PR3221#discussion_r2127587019; djeong20 PR3111#discussion_r2055208267;
  haehun PR4058#discussion_r3548382593; songgot PR3469#discussion_r2336038598
  (stale `@param`); Jungwon-Lee PR3874#discussion_r3179197457 (`@file` after
  rename).
- **Strength**: 6+ PRs.
- **Verify**: `bash .github/workflows/static.check.scripts/doxygen-tag.sh <list-file> 1`.

### G5. Log with `ml_log*`, not `std::cout`; no `<iostream>` in headers
- **Rule**: Use `ml_logi/ml_logw/ml_loge/ml_logd` from `nntrainer_log.h`;
  delete debug prints before committing; do not add `<iostream>` to headers.
- **Why**: `ml_log*` routes to dlog/Android log; `cout` and debug lines are
  removed in review every time.
- **Evidence**: jijoongmoon PR3088#discussion_r2034249219; piotrrak
  PR3049#discussion_r2007261208; EunjuYang PR3859#discussion_r3122075531;
  baek2sm PR3691#discussion_r2667070779; songgot PR3477#discussion_r2348297214.
- **Strength**: 5 PRs. Some old `std::cout` remain (neuralnet.cpp: 9).
- **Verify**: `git diff <base> | grep -n 'std::cout\|#include <iostream>'`.

### G6. Do not shadow class members with locals
- **Rule**: Never name a local or parameter like a member (`axis` vs
  `this->axis`).
- **Why**: It already hid a bug (slice layer never set `this->axis`); the
  maintainer said it "will be soon prohibited". `-Wshadow` is not on yet
  (meson.build: `warning_level=1`).
- **Evidence**: myungjoo PR3338#discussion_r2508759944,
  PR3574#discussion_r2544362825.
- **Strength**: 2 PRs, maintainer stated outright.
- **Verify**: build with `-Dcpp_args=-Wshadow` on changed files.

## Rejected
- Device-specific code only via registration, no backend `#ifdef` in core (jijoongmoon PR3705/3842/3900): upstream engine.cpp includes `cuda_context.h` and neuralnet.cpp has 10+ `ENABLE_OPENCL/CUDA` blocks, so the code contradicts it.
- Virtual capability hooks instead of `getType()==` checks (EunjuYang PR3773/3813): network_graph.cpp still uses type checks throughout; advisory only.
- Validate property vector sizes / empty checks (piotrrak PR2984, songgot PR3743, baek2sm PR3372): generic input validation.
- Use the POSIX-free helpers (`DynamicLibraryLoader`, `std::filesystem`) on Windows (PR2984, b4f5e21c9): already covered by root "stay cross-platform"; one PR.
- `unsigned` vs `-1` loops (2bf371e11, 70c7fa5ac, myungjoo PR3621): real bugs, but generic C++ that a competent default avoids.
- Dangling `ss.str().c_str()` in throw (7be51570a, piotrrak PR2984): generic C++.
- Do not silently promote declared input / on-disk weight dtype (caa2aa12c, 3934acf81): narrow to InputLayer/FC bias, now encoded in code.
- NCHW/NHWC format check (EunjuYang PR3212, jaemini-shin PR4096): 2 PRs, but one is "assumes NCHW" with no ask; too weak.
- Remove dead/unused code, typos, missing EOF newline, copyright line: generic or CI-enforced (newline.sh).
- Optimizer specifics (mixed precision in AdamW PR2780, MeZO naming/coupling PR3762/3813): single feature each.
- OpenCL buffer manager default 5x32 MiB allocation (myungjoo PR3625): belongs to the opencl unit.
- FSU `setWeightOffset` once / lookahead logic (PR3088, PR2846): one feature, churned code.

## Unit verdict
- **layers**: file worthwhile, 7 rules (L1-L7).
- **models**: skip as its own file, 1 rule (M1). Put it in `nntrainer/AGENTS.md` or in the layers file if a shared core file is made.
- **compiler_graph**: marginal, 2 rules (C1, C2). Put C1 in `nntrainer/compiler/AGENTS.md` only if one file per dir is required; otherwise fold into a core file.
- **nntrainer_misc / utils+engine/context**: file worthwhile (`nntrainer/utils/` or `nntrainer/AGENTS.md`), 3 rules (X1-X3).
- **nntrainer_misc / optimizers, dataset, mem_allocator, cl_/cuda_ top-level**: skip, 0 rules.
- **Global**: 6 rules (G1-G6); G1 could also live in `nntrainer/AGENTS.md` if root should stay short.
