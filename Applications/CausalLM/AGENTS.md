# AGENTS.md — Applications/CausalLM

Rules for editing anything under `Applications/CausalLM/` (models, layers,
`res/`, `api/`, `qnn/`, `Android/`, build files). They add to the root
[`AGENTS.md`](../../AGENTS.md); read that first.

## Weight files are positional

`.bin` weights load in the order layers are created in `constructModel()`,
`createAttention()` and `createMlp()`. Keep each model's
`res/<model>/weight_converter*.py` write order equal to that creation order:
when you change one side, change the other in the same commit. The shared
`Transformer::createMlp` order (up, gate, down) is fixed; a model whose file
uses another order overrides `createMlp()` in its own class.

A mismatch loads projections into the wrong layers and produces fluent garbage
with no error, and a reorder breaks every existing user `.bin`. This regressed
and was reverted twice in 2026 (87f5fea19, 45330052b).

## A new source file goes into every build list

When you add or rename a `.cpp` or a model directory, update all of these in
the same commit:

1. The in-tree meson: `models/<m>/meson.build` plus `subdir()` in
   `models/meson.build`, or `layers/meson.build`.
2. The explicit `causallm_src` list in `app_build/meson.build`. It is the
   production Android build and is kept explicit on purpose.
3. `jni/Android.mk` (the device test harness): `CAUSALLM_COMMON_INCLUDES` for
   a new directory, and the test module source lists for a new test.
4. `models/model_registry.cpp` for a new architecture.

When a model is excluded on Windows, exclude it in both (1)
(`get_option('platform') != 'windows'` in `models/meson.build`) and (4)
(`#if !defined(_WIN32)`). The lists are independent and hand-maintained; PR CI builds (2)
through the CausalLM Android job but never runs ndk-build on (3).

## `incremental_forwarding(from, to)` uses absolute positions

`from` and `to` are absolute sequence positions. Index the KV cache
absolutely. The step input and output tensors hold only rows `[from, to)`, so
index them with `pos - from` and use `to - from` as the step size. Detect
prefill with `is_prefill = !from || (to - from) > 1`, as the existing layers
do. `lm_head` and the tied embedding intentionally keep `!from` (see
1a80cfa35). A 1-row chunked-prefill tail still looks like decode.

Absolute-versus-relative offsets have been the most common CausalLM
correctness bug: wrong `lm_head` row in multi-turn, out-of-bounds V-cache
reads, stale output from `skip_prefill` layers. Cover a `from > 0` case in
tests.

## Keep every dtype and arch branch in step

Layers carry parallel FP32, `ENABLE_FP16`, `__ARM_NEON`/`__ANDROID__` and x86
paths. Apply a fix to every sibling branch, not only the one you ran. Select
kernels with the preprocessor (`#ifdef ENABLE_FP16`, arch macros), not a
runtime flag, and declare branch-only variables inside their branch.

PR CI tests CausalLM only on x86 with `-Denable-fp16=false`, and compiles the
Android build without running it. Before you push a layer change, also build
x86 with FP16:

```bash
meson setup build-fp16 -Denable-transformer=true -Denable-fp16=true
ninja -C build-fp16
```

On a device, `Applications/CausalLM/run_unittest_android.sh` runs the FP16
path of `unittest_mha_core` and `unittest_causallm_models`.

## One level of `parallel_for`

Parallelize through `nntrainer::ThreadManager::Global().parallel_for` at one
level only. Work that already parallelizes (`Tensor::dot`, gemm/gemv, ggml
interface calls) stays outside a `parallel_for` body; for MoE, keep the expert
loop serial and let the inner GEMM parallelize. Leave thread counts to the
thread manager: no `openblas_set_num_threads`, `omp_set_num_threads`, or
hard-coded counts in a layer or model.

`parallelize()` holds a non-recursive mutex, so nesting deadlocks
(`thread_manager.h`). The nesting is invisible at the call site because
`dot()` hides the inner `parallel_for`. In tests a deadlock shows up as a
timeout in the MoE cases of `unittest_causallm_models`.

## Public models only

Name only public models anywhere under `Applications/CausalLM`: code,
comments, scripts, READMEs, and `api/` headers. Describe downstream or
proprietary work neutrally ("a proprietary overlay is built downstream against
the prebuilt libcausallm.so"). `api/quick_dot_ai_api.h` is a public C ABI:
append enum values, never renumber them. Grep your diff for internal model or
product names before pushing; no CI check does this.

## `res/<model>/` holds sources only

Commit only `weight_converter*.py`, `nntr_config.json`, a README and build
scripts under `res/<model>/`. `Applications/**/res/*` is ignored, so add `!`
entries in `.gitignore` for a new model directory like the existing ones.
Files Hugging Face serves (`tokenizer.json`, `config.json`), weights, `.bin`
files, generated headers and prebuilt archives stay out of the tree.

## Honour `BATCH_SIZE`

Index inputs, outputs, gradients and cache counters per batch (`b * len`
offsets or per-batch slices), not one scalar shared across the batch. A path
that supports only batch 1 throws on `BATCH_SIZE > 1` instead of silently
computing batch 0. All test fixtures use batch 1, so check a batch-2 run by
hand (`"batch_size": 2` in `nntr_config.json`).

## Checks

```bash
meson setup build -Denable-transformer=true   # -Denable-fp16=false, as CI
ninja -C build
meson test -C build unittest_mha_core unittest_causallm_models --print-errorlogs
Applications/CausalLM/build_android.sh        # the CausalLM Android CI job
```
