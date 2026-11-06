# Agent-rule candidates: group "apps"

Inputs: unit_causallm.json (287 comments, 105 PRs), unit_apps_other.json (65
comments, 33 PRs), plus `git log refs/remotes/upstream/main --since=2026-03-01`
over `Applications/` (479 commits, about 150 fix/revert/build commits opened
or skimmed). Every code claim below was checked against upstream main
@640ac6e0e with `git show`, not the working tree. Nothing was built or run.

## causallm

Scope: one `Applications/CausalLM/AGENTS.md` covers the whole tree (see the
Unit verdict for the subdirectory question).

### C1. Weight files are positional: keep converter order equal to layer-creation order
- **Rule**: `.bin` weights load in the order layers are created in
  `constructModel()`/`createMlp()`/`createAttention()`. When you change one side
  (the C++ creation order, or `res/<model>/weight_converter.py`'s write order),
  change the other to match. Never reorder the shared `Transformer::createMlp`
  (up, gate, down). A model whose file uses a different order overrides
  `createMlp()` in its own class.
- **Why**: a mismatch loads each projection into the wrong layer. It produces
  fluent garbage in every format (FP32/Q4_0/Q6_K/QS4CX) with no error, and a
  silent reorder breaks every existing user `.bin`.
- **Evidence**: 87f5fea19 (Jaemin Shin, "Restore the legacy up, gate FFN
  weight order"), 45330052b (revert converter order to the legacy binary),
  de8f981cf (Qwen2 MLP order), b719613fe (Gemma3 Q/K/V/norm/O + FFN order);
  review: https://github.com/nntrainer/nntrainer/pull/3929#discussion_r3279826109
  (jaemini-shin), https://github.com/nntrainer/nntrainer/pull/3784#discussion_r2916523045
  (jayden0701, Q/V order vs converter).
- **Strength**: 6 PRs/commits by 4 authors, regressed and reverted twice in
  2026. The warning comment in `transformer.cpp:984` exists only on
  `createMlp`; converters and per-model overrides carry no such warning.
- **Verify**: `meson test -C build unittest_causallm_models` (the
  `*FP32MatchesHFReference*` cases), and load a `.bin` made before the change.
  Caveat: 87f5fea19's message says the Qwen2, Qwen3 and LFM2 fixtures were
  regenerated gate-first and those cases fail until the fixtures are rebuilt.
  No fixture commit has landed since. I did not run this to check.

### C2. A new source or model directory goes into every build manifest
- **Rule**: when you add or rename a `.cpp` or a model dir, update all of
  these in the same commit:
  (a) the in-tree meson: `models/<m>/meson.build` + `subdir()` in
  `models/meson.build`, or `layers/meson.build`;
  (b) the explicit `causallm_src` list in `app_build/meson.build`, which is
  the canonical Android manifest and is deliberately not globbed;
  (c) `jni/Android.mk`: `CAUSALLM_COMMON_INCLUDES` for a new dir, plus the
  test module source lists for a new test;
  (d) `models/model_registry.cpp` for a new architecture.
  Add the same `!defined(_WIN32)` guard in (a) and (d) when a model is
  excluded on Windows.
- **Why**: the lists are hand-maintained and independent. A file that only
  meson knows about stays invisible until someone builds for a device, and
  no CI job runs ndk-build on this app (0a8ac92fd).
- **Evidence**: 0a8ac92fd (jijoongmoon, "Add the missing CausalLM sources to
  the ndk module lists": an audit found 3 misses), 0b56d8658 (LFM2 never
  wired into Android), 547d2d45f (XLM-RoBERTa Android), 405cdedc9
  (bert_transformer Android), 5ed5b5c61 (rename not propagated);
  review: https://github.com/nntrainer/nntrainer/pull/3701#discussion_r2696817783
  (jijoongmoon, source missing from meson),
  https://github.com/nntrainer/nntrainer/pull/3773#discussion_r2986437964
  (EunjuYang, duplicate dep).
- **Strength**: 5 fix commits by 4 authors + 2 review PRs. The "keep this list
  explicit" convention is written by the maintainer in `app_build/meson.build`.
- **Verify**: `ninja -C build` with `-Denable-transformer=true`, then
  `Applications/CausalLM/build_android.sh` (the CausalLM Android Build CI job)
  and `run_unittest_android.sh` for test modules.

### C3. `incremental_forwarding(from, to)` contract: absolute positions vs. step tensors
- **Rule**: `from`/`to` are absolute sequence positions. Index the KV cache
  absolutely. The step input/output tensors hold only rows `[from, to)`, so
  index them with `pos - from`, with step size `to - from`; never use
  `to - 1` or `query.height()` as a row index. For prefill detection, use the
  tree convention `is_prefill = !from || (to - from) > 1`, never `!from`
  alone. Note that a 1-row chunked-prefill tail still looks like decode.
- **Why**: absolute-vs-relative offsets were the most common CausalLM
  correctness bug: wrong lm_head row in multi-turn, out-of-bounds V-cache
  reads, a full-length attention computed per decode step, and stale output
  when a `skip_prefill` layer misses a later-turn prefill.
- **Evidence**: https://github.com/nntrainer/nntrainer/pull/3482#discussion_r2348111204
  (EunjuYang, wrong out offset),
  https://github.com/nntrainer/nntrainer/pull/3814#discussion_r2986262512
  (EunjuYang, `(to - from - 1)` fix),
  https://github.com/nntrainer/nntrainer/pull/3799#discussion_r2992842902
  (EunjuYang, `from == 0` is not prefill),
  https://github.com/nntrainer/nntrainer/pull/4040#discussion_r3517164606
  (jaemini-shin, chunked-prefill edge); commits 918661016 (lmhead offset),
  a03cb732a (MHA step size), 1a80cfa35 (is_prefill unified across 11
  layers), 460a2efbf / ff1797985 (row_num / timestep off-by-one).
- **Strength**: 4 review PRs + 5 fix commits. Maintainer EunjuYang raised it 3
  times. `lm_head` and `tie_word_embedding` intentionally keep `!from`
  (1a80cfa35 explains why).
- **Verify**: `meson test -C build unittest_mha_core unittest_causallm_models`
  (multi-turn needs a `from > 0` case).

### C4. Keep every dtype/arch branch in step, and build the branches CI does not
- **Rule**: CausalLM layers carry parallel FP32 / `ENABLE_FP16` /
  `__ARM_NEON`+`__ANDROID__` / x86 paths. Put a fix in every sibling branch,
  not only the one you tested. Select kernels with the preprocessor
  (`#ifdef ENABLE_FP16`, arch macros), not a runtime bool. Declare
  branch-only variables inside their branch. Before you push, build with
  `-Denable-fp16=true` as well, plus the Android build.
- **Why**: Ubuntu CI builds and tests CausalLM only in the x86
  `-Denable-fp16=false` cell (`ubuntu_clean_meson_build.yml`). The FP16 and
  ARM paths stay broken unseen: an FP16 build "has not compiled" (4edc1673c),
  the standalone app silently compiled out FP16 (8c119f626), an OOB fix
  landed on avx2 only and needed a second PR for FP16 && ANDROID, and gemm
  test cases run the per-row arm on ARM.
- **Evidence**: https://github.com/nntrainer/nntrainer/pull/3778#discussion_r2903629803
  (Jungwon-Lee) → follow-up https://github.com/nntrainer/nntrainer/pull/3789#discussion_r2921821265
  (jaemini-shin; commits 460a2efbf → 0e50be090),
  https://github.com/nntrainer/nntrainer/pull/4000#discussion_r3526749987
  (jaemini-shin, compile-time branching),
  https://github.com/nntrainer/nntrainer/pull/3419#discussion_r2289541153
  (jijoongmoon, FP32 input + FP16 compute),
  https://github.com/nntrainer/nntrainer/pull/3933#discussion_r3273290690
  (baek2sm, bug in FP32 path),
  https://github.com/nntrainer/nntrainer/pull/4462#discussion_r4408719351
  (Seunghui98, ARM arm-selection not covered by CI); commits 5c121eec7,
  4edc1673c, 8c119f626, 3ddc908bd (FP32 and FP16 branches each had a
  different sliding-window bug).
- **Strength**: 6 review PRs (3 maintainers) + 5 fix commits.
- **Verify**: `meson setup build-fp16 -Denable-transformer=true -Denable-fp16=true && ninja -C build-fp16`;
  `Applications/CausalLM/build_android.sh`; on device,
  `Applications/CausalLM/run_unittest_android.sh` (runs `unittest_mha_core`
  on the FP16 path, with tolerance set by b541d4e4e).

### C5. One level of `parallel_for`, and no process-global thread knobs
- **Rule**: parallelize through `nntrainer::ThreadManager::Global().parallel_for`
  at one level only. Never wrap work that itself parallelizes (`Tensor::dot`,
  gemm/gemv, ggml interface calls) in a `parallel_for` body. For MoE, keep the
  outer expert loop serial and let the inner GEMM parallelize. Never call
  `openblas_set_num_threads`/`omp_set_num_threads` or hard-code a thread
  count inside a layer or model.
- **Why**: `ThreadManager::parallelize()` holds a non-recursive mutex, so
  nesting deadlocks (`thread_manager.h` @warning). Global thread counts change
  every other layer in the process.
- **Evidence**: 848dcb7e9 and 1e32a5dc5 (Jungwon-Lee, parallel_for deadlock
  in all five MoE layer variants);
  https://github.com/nntrainer/nntrainer/pull/3773#discussion_r2986437677
  (EunjuYang, global OpenBLAS thread count = 4),
  https://github.com/nntrainer/nntrainer/pull/4268#discussion_r3800184122
  (jaemini-shin, thread-manager chunk heuristic; wrap it in the kernel
  interface), https://github.com/nntrainer/nntrainer/pull/3907#discussion_r3158878199
  (jaemini-shin).
- **Strength**: 2 fix commits (5 files) + 3 review PRs. The nesting trap is not
  visible at the call site: `dot()` hides the inner `parallel_for`.
- **Verify**: `meson test -C build unittest_causallm_models --suite unittests`
  with the MoE cases (qwen3_moe, gpt_oss and their cached_slim variants).
  A deadlock shows up as a test timeout.

### C6. The public tree carries public models only
- **Rule**: never name proprietary or downstream models, product model ids,
  descriptors, enum values or file names anywhere under `Applications/CausalLM`
  (code, comments, scripts, README, `api/` headers). Describe such things
  neutrally ("a proprietary overlay is built downstream against the prebuilt
  libcausallm.so"). Append to `api/quick_dot_ai_api.h` enums; never renumber
  them.
- **Why**: the tree is staged for a public split. Leaked names have needed
  cleanup commits three times, and the public C header is an ABI.
- **Evidence**: ee3961978 (jayden0701, "remove proprietary model references
  from public tree"), a97c208ea (drop product-specific enum, ABI-safe
  because it was the last value), d6f098ae6 (jijoongmoon, "Keep downstream
  model names out of the app build comments"); PR #4092 title "(public, no
  private-model references)"; header comment in `app_build/meson.build`
  ("must never reference proprietary models or descriptors").
- **Strength**: stated outright by maintainer jijoongmoon (d6f098ae6) and
  in-tree. 3 cleanup commits. No inline review comment, because the leaks
  were caught after merge.
- **Verify**: none automated. Grep the diff for internal model or product
  names before pushing.

### C7. `res/<model>/` holds only the converter, `nntr_config.json` and README
- **Rule**: under `res/<model>/`, commit only `weight_converter*.py`,
  `nntr_config.json`, README and build scripts. Allowlist them in
  `.gitignore`: `Applications/**/res/*` is ignored, so each new model dir
  needs `!` entries like the existing ones. Do not add files that Hugging Face
  serves (`tokenizer.json`, `tokenizer_config.json`, `config.json`), weight
  or `.bin` files, generated headers (`json.hpp`), or prebuilt archives other
  than the allowlisted `lib/libtokenizers_c.a`, which is rebuilt by
  `tools/build_tokenizers_c.sh`.
- **Why**: large, redistributable or generated blobs bloat the repo and go
  stale. Reviewers reject them on sight (myungjoo: "NO!").
- **Evidence**: https://github.com/nntrainer/nntrainer/pull/3890#discussion_r3152368356
  (Jungwon-Lee, tokenizer.json) + https://github.com/nntrainer/nntrainer/pull/3890#discussion_r3134503819
  (jijoongmoon), https://github.com/nntrainer/nntrainer/pull/3976#discussion_r3412919630
  (Jungwon-Lee, config.json),
  https://github.com/nntrainer/nntrainer/pull/4246#discussion_r3810668388
  (jaemini-shin, json.hpp is generated),
  https://github.com/nntrainer/nntrainer/pull/3459#discussion_r2540917616
  (myungjoo, android tokenizer .a).
- **Strength**: 4 PRs, 4 reviewers incl. 2 maintainers. Weaker than C1–C5:
  `res/tiny-bert/config.json` and `models/gemma4/config.json` are tracked on
  main, so past practice is not clean.
- **Verify**: `git status --ignored Applications/CausalLM/res` and check
  that nothing was force-added (`git add -f`).

### C8. Honour `BATCH_SIZE`: index per batch, or reject batch > 1 explicitly
- **Rule**: in `models/*.cpp` generation code and in layers, index inputs,
  outputs, gradients and cache counters per batch (`b * len` offsets or
  per-batch slices). Don't keep one scalar for all batches (a token id, a
  cache index, a skipped-token buffer). If a path supports only batch 1,
  throw on `BATCH_SIZE > 1` rather than computing on batch 0 silently.
- **Why**: batch-1 assumptions keep coming back in multi-input, multiturn,
  skip-prefill and training work, and they give silently wrong results for
  batch > 1.
- **Evidence**: https://github.com/nntrainer/nntrainer/pull/3799#discussion_r3072181876
  (jaemini-shin), https://github.com/nntrainer/nntrainer/pull/3814#discussion_r2985297705
  (jayden0701), https://github.com/nntrainer/nntrainer/pull/3864#discussion_r3076861127
  (EunjuYang, full dy instead of per-batch slice),
  https://github.com/nntrainer/nntrainer/pull/3736#discussion_r2744601396
  (baek2sm, batch loop that does not work),
  https://github.com/nntrainer/nntrainer/pull/3750#discussion_r2839135707
  (jaemini-shin).
- **Strength**: 5 PRs, 4 reviewers (2 maintainers).
- **Verify**: none in CI (all fixtures use batch 1). Run `nntr_causallm` with
  `"batch_size": 2` in `nntr_config.json`.

## apps_other

### A1. Sample apps use only the public ccapi
- **Rule**: example applications build and drive models through
  `ml::train::Model` / `ml::train::createModel` (ccapi `model.h`). Prefer the
  symbolic `Model::compile(Tensor, ...)` graph API for new apps. Register
  custom layers with `ct_engine.registerLayerFactory("cpu", nntrainer::createLayer<X>)`.
  Never use `nntrainer::NeuralNetwork`, `trainMeZO`-style internals, or a
  `static_cast<nntrainer::AppContext *>` downcast.
- **Why**: the apps are the reference for API users. Internal types are not
  part of the installed API and break on Android prebuilt and packaged builds.
- **Evidence**: https://github.com/nntrainer/nntrainer/pull/3762#discussion_r2876740268
  and https://github.com/nntrainer/nntrainer/pull/3762#discussion_r2876743881
  (EunjuYang: "All interfaces that the NNTrainer user can utilize are defined
  in model.h"), https://github.com/nntrainer/nntrainer/pull/2930#discussion_r1980390221
  (jijoongmoon, use installed headers); commits e444a9fac (jijoongmoon,
  register through the Engine facade, no downcast) and c6080cd21 (migrate
  13 sample apps to the graph API).
- **Strength**: 2 review PRs + 2 maintainer commits that applied it
  tree-wide. Main is consistent: 0 downcasts and 0 `NeuralNetwork` uses in
  non-CausalLM apps. CausalLM still has 2 downcasts (gemma4_moe, lfm2) that
  e444a9fac left "for its own change".
- **Verify**: `git grep -n 'getRegisteredContext\|NeuralNetwork' Applications ':!Applications/CausalLM' -- '*.cpp' '*.h'`
  returns nothing. `ninja -C build test` runs the app tests.

## Global (belongs in root AGENTS.md, not a unit file)

- **G1. End every text file with a newline. This includes `.json`, `.py`, `.ini`
  and `.md`, which clang-format never touches.** The `static.check` "Incorrect
  newlines" job (`newline.sh`) fails on any changed text file without one.
  It is the most repeated nit in this group: 8 PRs, 4 reviewers.
  https://github.com/nntrainer/nntrainer/pull/3672#discussion_r2654898134,
  https://github.com/nntrainer/nntrainer/pull/3712#discussion_r2704304465,
  https://github.com/nntrainer/nntrainer/pull/3890#discussion_r3134503018,
  https://github.com/nntrainer/nntrainer/pull/3984#discussion_r3420308656
  (all jijoongmoon), https://github.com/nntrainer/nntrainer/pull/3548#discussion_r2768031438
  and https://github.com/nntrainer/nntrainer/pull/3594#discussion_r2579950454
  (baek2sm), https://github.com/nntrainer/nntrainer/pull/3976#discussion_r3387885421
  (jaemini-shin). Verify: `git diff <base> | grep '^\\ No newline'` is empty.
- **G2. Every new C/C++ file starts with an SPDX line and a doxygen block with
  `@file` (the real file name), `@date`, `@brief`, `@see`, `@author`, `@bug`.
  When you copy a file, rewrite `@file`, `@brief` and the include guard
  (`__NEW_NAME_H__`, closing `#endif /* __NEW_NAME_H__ */`).**
  `doxygen-tag.sh` enforces `@file @brief @author @bug` on changed
  `.c/.h/.cpp/.hpp`. Copy-paste headers are the common miss.
  https://github.com/nntrainer/nntrainer/pull/3672#discussion_r2654896093,
  https://github.com/nntrainer/nntrainer/pull/3690#discussion_r2667412968
  (jijoongmoon, SPDX), https://github.com/nntrainer/nntrainer/pull/3722#discussion_r2706707303,
  https://github.com/nntrainer/nntrainer/pull/3737#discussion_r2752225575
  (jaemini-shin, stale @file and guard),
  https://github.com/nntrainer/nntrainer/pull/3682#discussion_r2681261386
  (songgot); CI-fix commits 05391baa9, c4ccdf104, 86e513680, 266dc82e7.
  Verify: `bash .github/workflows/static.check.scripts/doxygen-tag.sh <list-file> 1`.
- **G3. Touch only the lines your change needs: no reformatting, no reordering,
  no deleting comments or options in untouched code.** Root AGENTS.md says
  "one commit = one topic" but not this. "Is this change necessary?" and
  "Why do you remove this?" come up in 10+ PRs from two reviewers:
  https://github.com/nntrainer/nntrainer/pull/3704#discussion_r2730642909,
  https://github.com/nntrainer/nntrainer/pull/3711#discussion_r2715362326,
  https://github.com/nntrainer/nntrainer/pull/4044#discussion_r3527548049
  (jaemini-shin), https://github.com/nntrainer/nntrainer/pull/3814#discussion_r2985864118,
  https://github.com/nntrainer/nntrainer/pull/3937#discussion_r3411381902
  (Seunghui98).

## Rejected

- Typos, naming nits, unused includes/variables, debug prints left in
  (songgot, jayden0701, jaemini-shin; 20+ comments): generic, and compilers or
  the misspelling check catch them.
- Magic numbers → named constants (myungjoo PR3496/3598, songgot): generic advice.
- MoE expert cache must be a locked LRU (re-push on hit, mutate the deque
  under `cache_mutex`, named capacity): recurring with maintainers
  (PR3481/3496/3598/4248), but it is plain correctness review on 2 files, and
  generic.
- Windows `WSTR`/wide-string prompt handling (PR3701, 3752, 3827): obsolete.
  `WSTR` is now `std::string` on both platforms (`models/causal_lm.h:33-40`).
- Register a new model in main.cpp and quantize.cpp separately (9fb294be5):
  obsolete. `models/model_registry.cpp` `registerAllModels()` is shared
  (that file is folded into C2(d)).
- Move SIMD/attention kernels out of layers into `cpu_backend` (jaemini-shin
  PR4000, 4045, 4268; done in 7c20db245): contradicted by the code.
  `mha_core.cpp` still holds about 45 intrinsic lines that maintainer commit
  4edc1673c re-added on purpose.
- Shared LLM layers live in `nntrainer/layers/llm` and register through
  AppContext; don't re-register them in the app (EunjuYang PR3847): only 1
  review PR, and one `ls` shows it.
- `forwarding()` must match `incremental_forwarding()`; inference-only layers
  return `supportBackwarding() == false` (EunjuYang PR3835, 3773): a generic
  layer contract. It belongs in an `nntrainer/layers` unit if anywhere.
- New `nntr_config.json` keys optional with feature-off default; model
  constants in config, not code (PR4045, 3919, 3773, 3496): `transformer.cpp`
  already shows the `contains() ? … : default` pattern. Derivable.
- `api/`, `qnn/`, `Android/` specifics: `api/` comments are style nits
  (guard clauses, enum naming). `qnn/` has 1 comment. `Android/` (Gradle and
  Kotlin) has 0 review comments, and its only lint fix (f66ad49ce) is generic.
- Sample-app `main()`: no nested try/catch and no throw inside main's try
  (myungjoo PR2918, 3028): myungjoo accepted it both times. Low value.
- Don't use `uint` (djeong20 PR2854/2906, "as in #2890"): main is already
  clean, it comes from 1 reviewer, and root "stay cross-platform" covers it.
- Windows meson tricks for sample apps (piotrrak PR2952, gkisalapl PR3073):
  one PR each, from 2025.

## Unit verdict

- **causallm**: file worthwhile. 8 rules (C1–C5 strong, C6
  maintainer-stated, C7 and C8 moderate). Put one
  `Applications/CausalLM/AGENTS.md` at the CausalLM root. The subdirectories do
  **not** need separate files. `layers/` and `models/` draw 70% of comments,
  but C1, C3, C4 and C5 each span both (and `res/`). C2 and C6 are tree-wide.
  `api/`, `qnn/` and `Android/` have no rule of their own. If the file grows,
  a heading per area beats more files.
- **apps_other**: skip. 1 rule (A1). Most of its 65 comments are typos or
  2025-era FSU/sample nits. If A1 is wanted, a 3-line `Applications/AGENTS.md`
  would hold it, and C-rules must not be put there.
