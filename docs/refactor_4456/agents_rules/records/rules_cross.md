# Candidate agent rules: group "cross" (test, api, ci, other, docs)

Sources: unit_test.json (135 comments / 67 PRs), unit_api.json (26/16),
unit_ci.json (31/24), unit_other.json (92/58), unit_docs.json (3/2), the
other unit files for cross-unit counts, and `git log refs/remotes/upstream/main
--since=2026-03-01`. Every rule below was checked against upstream main @ 640ac6e0e.
Note: the unit files hold comments from late 2024 onward, not only from 2026-06-19.
Comment links are shortened to `PR#discussion_rNNN`, meaning
`https://github.com/nntrainer/nntrainer/pull/PR#discussion_rNNN`.

## test

### T1. Put each test in a file whose build gate matches what the test needs
- **Rule**: Before adding a case to an existing test file, check how that file is built.
  `*_cl.cpp`, `unittest_opencl_*`, `unittest_attention_kernels_cl` are built only with
  `enable-opencl`. `unittest_nntrainer_kleidiai` and `unittest_nntrainer_ggml_arm` are
  aarch64 only. `unittest_nntrainer_cpu_backend_fp16.cpp` and
  `unittest_nntrainer_tensor_neon_fp16.cpp` are in `test/jni/Android.mk` only. CPU logic
  goes in a file that is not gated. x86-only code goes in a new file listed in meson only.
- **Why**: if the gate is wrong, the test never runs where the code ships, or it breaks the build on the other platform.
- **Evidence**: djeong20 3575#discussion_r2558237537 ("has nothing to do with the GPU, could be tested without enable-opencl"); jaemini-shin 3967#discussion_r3377350076 ("built for android only ... implement a separate test file ... add to meson.build, not Android.mk"); djeong20 3417#discussion_r2286910041 (CPU test sitting in the OpenCL block of Android.mk).
- **Strength**: 3 PRs, 2 maintainers.
- **Verify**: `git show upstream/main:test/unittest/meson.build` (the `enable-opencl`, `enable-fp16` and `cpu_family()=='aarch64'` blocks). Build with `-Denable-opencl=false` and run `meson test -C build --suite unittests`.

### T2. Register a new test source in both test lists, and build the Android one yourself
- **Rule**: Add a new test `.cpp` to `test/unittest/meson.build` (or to the subdir's `meson.build`). Also add it to `test/jni/Android.mk`, which is a separate hand-maintained list, unless the test cannot run on Android. When you touch anything that file compiles, run `tools/android_test.sh`. PR CI never builds `test/jni`: `android.yml` builds only the library and the LogisticRegression app. `test/jni/Android.mk` always passes `-DENABLE_FP16=1`.
- **Why**: a file left out of a list never runs, and an Android test break stays green in PR CI.
- **Evidence**: myungjoo 3633#discussion_r2613643033 ("You've got missing files" in layers/meson.build); skykongkong8 3189#discussion_r2113428901 ("This should be included", meson.build); djeong20 3365#discussion_r2238048482 ("might fail on the Android test ... check by running tools/android_test.sh"); EunjuYang 2752#discussion_r1847566918 ("pass all unittest under ./tools/android_test.sh"); commit a531cf219 (android_test.sh broken by a renamed meson option, found only manually).
- **Strength**: 4 PRs + 1 fix commit; 4 maintainers.
- **Verify**: `ninja -C build && meson test -C build --suite unittests`, then `ANDROID_NDK=... tools/android_test.sh` (needs a device over adb). If you cannot run it, say so.

### T3. Reach CPU kernels through `cpu_backend.h` and never include BLAS or arch headers directly
- **Rule**: In a test that builds on every platform, call kernels through `cpu_backend.h`. Use `fallback_internal.h` for reference values. Do not include `cblas.h`, `neon_impl.h`, `ggml_interface.h`, `nntr_ggml_impl.h` or x86 backend headers. The only exception is a test file whose meson entry is gated to that architecture (kleidiai, ggml_arm).
- **Why**: direct includes break `-Denable-blas=false` builds and non-matching architectures.
- **Evidence**: skykongkong8 3360#discussion_r2238917944 ("NNTrainer expects all the functions from cpu_backend.h to be called from cpu_backend.h directly ... might make a build fail"); djeong20 3407#discussion_r2283992169 ("Let's only add cpu_backend.h"); djeong20 3589#discussion_r2583678494 (cblas include vs `enable-blas`).
- **Strength**: 3 PRs, 2 maintainers. Upstream: no test includes cblas; arch headers appear only in the aarch64-gated files.
- **Verify**: `git grep -nE '#include [<"](cblas|neon_impl|ggml_interface|nntr_ggml_impl|x86_compute_backend)' -- test` should list only aarch64-gated files. Build with `-Denable-blas=false`.

### T4. Every TEST asserts something and prints nothing
- **Rule**: Each `TEST`/`TEST_F` checks its result with `EXPECT_*`/`ASSERT_*`. Do not leave `std::cout` value dumps, "BRK" markers, timing printouts or debug `print`s in tests or in `test/input_gen` golden generators.
- **Why**: a case with no assertion can never fail, and printing floods CI logs and golden output.
- **Evidence**: myungjoo 3628#discussion_r2613933882 ("A test case that can never fail? (no assert/expect?)"); dkjung 3283#discussion_r2204045221 ("no test function (e.g. EXPECT_EQ)"); djeong20 3407#discussion_r2283993161 (remove print loop); EunjuYang 3859#discussion_r3122072434 (debug prints in gen_layer_tests.py); Seunghui98 4136#discussion_r3654025045 (printing test time).
- **Strength**: 5 PRs, myungjoo and 3 other maintainers.
- **Verify**: review the diff; `git diff <base>...HEAD -- test | grep -n 'std::cout\|printf'`.

### T5. Generate golden and fixture data with the in-tree generators and ship it as a `packaging/*.tar.gz`
- **Rule**: Produce layer and model goldens with `test/input_gen/` (`gen_layer_tests.py`, `genModelTests_v2.py`). Produce CausalLM fixtures with `test/unittest/models/causallm_reference/generators/`. Put the binaries in the existing archive in `packaging/` (for example `unittest_layers.tar.gz`, `unittest_models_v2.tar.gz` for basic layer models, `causallm_reference_*.tar.gz`), which is unpacked by `unzip_target` in `test/unittest/meson.build`. Do not commit loose binaries. When a converter or weight order changes, regenerate the fixtures; do not hand-edit them.
- **Why**: goldens committed outside the archives are not present on GBS, Android or Windows test runs, and hand-edited goldens drift from the generator.
- **Evidence**: EunjuYang 2752#discussion_r1847566918 (add nnlayergolden to `packaging/unittest_layers.tar.gz`); baek2sm 3212#discussion_r2115208235 (pack into `unittest_models_v2.tar.gz`, not v3); commits d4db15c28 ("Follow the existing nntrainer convention ... tar.gz"), 82ac8d32a, 141871a58 (regenerated with the in-tree generators after the up/gate order change).
- **Strength**: 2 PRs (EunjuYang, baek2sm) + 3 commits.
- **Verify**: `meson test -C build --suite unittests` from a clean build dir, which unpacks the archives again.

### T6. Do not bend existing tests to fit a change, and test the real entry point
- **Rule**: Add new cases instead of editing or deleting existing ones to make a change pass. Drive the production path, such as `Manager::requestInputs()` or the dispatched kernel, rather than a copy of the rule inside the test. Check that the build flags actually compile the path you claim to test: aarch64 builds turn on `+dotprod`/`+i8mm` by default, so NEON emulation fallbacks are not built.
- **Why**: tests rewritten next to the code, or tests of a re-transcribed rule, pass while the real code is broken.
- **Evidence**: jaemini-shin 3967#discussion_r3377337292 ("Please do not try to modify existing test codes"); jaemini-shin 4458#discussion_r4398008523 (test compares against "a transcription of the same rule, so it can't catch wiring mistakes"); jaemini-shin 4299#discussion_r3891767447 (test always builds with `+dotprod`, so the fallback is untested); Seunghui98 3688#discussion_r2768010538 ("Why do you remove this unit test?").
- **Strength**: 4 PRs, mainly one maintainer (jaemini-shin). Medium.
- **Verify**: `git diff --stat <base>...HEAD -- test/` should show no deleted or rewritten existing assertions without a reason given in the commit message.

## api

### A1. Never renumber or rename an existing public enum value
- **Rule**: In `api/` (`nntrainer-api-common.h`, `ccapi/include/*.h`, `capi`), keep the names and numbers of existing enumerators. Only append. Fixing a genuinely duplicated value is an API/ABI break: say so in the commit, and it needs a version bump and a release note.
- **Why**: renaming breaks how older applications compile, and renumbering breaks older binaries (Tizen ships these headers).
- **Evidence**: myungjoo 3573#discussion_r2544049459 ("Keep the API forward compatibility. Do not rename NEG to NEGATIVE ... Do not change number for pre-existing enums"); myungjoo 3573#discussion_r2544057792 ("do not change the enum values. It breaks applications ... clearly state at the release note"); commit a97c208ea ("Value 15 was the last entry, so no other enum value is renumbered (ABI-safe)").
- **Strength**: maintainer (myungjoo) stated it outright, 2 PRs (#3573, #3570).
- **Verify**: `git diff <base>...HEAD -- api/ | grep -E '^-.*= *[0-9]+'` should be empty.

### A2. Define a new public layer type in `nntrainer-api-common.h` and alias it in `LayerType`
- **Rule**: Add `ML_TRAIN_LAYER_TYPE_<X> = <next free number>` with a `(Since N.0)` doc tag in `api/nntrainer-api-common.h`, then `LAYER_<X> = ML_TRAIN_LAYER_TYPE_<X>` in `api/ccapi/include/layer.h`. Internal-only types (not exposed through the C API) go after `LAYER_UNKNOWN` (999), in the 1000+ range.
- **Why**: avoids duplicate numbers between the C and C++ APIs, and keeps `LAYER_UNKNOWN` usable as the validation boundary.
- **Evidence**: myungjoo 3570#discussion_r2544374099 ("Move to api/nntrainer-api-common.h along with other ML_TRAIN_LAYER_*"); myungjoo 3570#discussion_r2540992450; songgot 3570#discussion_r2541104360 (UNKNOWN as boundary, internal 1000+); songgot 3573#discussion_r2548115127; djeong20 2752#discussion_r1868613839.
- **Strength**: 3 PRs; maintainer convention (myungjoo). Matches upstream layout (`LAYER_TIME_DIST` and other internal types after `LAYER_UNKNOWN`).
- **Verify**: `git show upstream/main:api/ccapi/include/layer.h | grep -n LAYER_UNKNOWN`.

### A3. Extend an existing API before adding a new public method
- **Rule**: Before adding a method to `ccapi` (`model.h`, `layer.h`), see whether an existing call can carry the feature, for example a new `ModelFormat` value for `save()`/`load()`. If not, see whether it can stay internal (`layer_devel.h`, model-internal). Add public API only when every user needs it.
- **Why**: public ccapi surface is permanent (see A1) and is shipped in the dev packages.
- **Evidence**: EunjuYang 3200#discussion_r2106525626 ("using `format` in save and load instead of making a new API"); EunjuYang 3200#discussion_r2106519505 ("create these operations in layer_devel.h only"); DonghakPark 3200#discussion_r2106561945 ("create a new API only when it is necessary in all cases"); jijoongmoon 3213#discussion_r2119877105 ("I wonder do we need this?").
- **Strength**: 2 PRs, 3 maintainers. Medium.
- **Verify**: none mechanical.

## ci

### C1. Add to an existing workflow's matrix, and copy a sibling workflow's skeleton
- **Rule**: To cover a new option combination, add a leg to the matrix of the existing workflow (as `android.yml` `cl_options` and the `meson_options` of `ubuntu_clean_meson_build.yml` do) rather than adding a new workflow file. If a new PR build workflow is really needed, copy a sibling: `name: "Build Test - <Platform> <tool>"`, the `concurrency` block with `cancel-in-progress: true`, the `./.github/actions/check-rebuild` step, and `if: env.rebuild == '1'` on every heavy step.
- **Why**: duplicate workflows add CI cost. Without the rebuild guard, doc-only PRs run full builds, and without `concurrency`, stale runs keep going after a new push.
- **Evidence**: djeong20 3417#discussion_r2286692303 ("this new workflow is not necessary. The existing android workflow file can be edited ... matrix"); djeong20 3448#discussion_r2315330391 (matrix option); DonghakPark 3199#discussion_r2108042406 and SeoHyungjun 3124#discussion_r2053298475 (name pattern); commits 5ad6032a7, 8173b0d26, c2bef4e04 ("Like the other build workflows it goes through check-rebuild"), e811bb427 (FSU added as a matrix leg).
- **Strength**: 4 PRs + 4 commits.
- **Verify**: `grep -L check-rebuild .github/workflows/*.yml` for PR build workflows; `actionlint` if installed.

### C2. Keep the required-CausalLM-test list in step with the tests
- **Rule**: The `-Denable-fp16=false` leg of `ubuntu_clean_meson_build.yml` fails when `unittest_causallm_models` or `unittest_mha_core` is missing from `meson introspect --tests`. If you rename, move or gate those targets, update the `required_tests` set in the same PR. Add a CausalLM test target there if it must not silently drop out. CausalLM (`enable-transformer=true`) is configured only in that leg.
- **Why**: this is the only test-presence gate in CI (see Root corrections: `check_count` does not count tests).
- **Evidence**: commit d07e979fa (dlwlzzero, reviewed upstream); workflow step "Check CausalLM model tests are enabled".
- **Strength**: 1 commit + config. Verified in code; it is a mechanical gate that an agent cannot infer from the test tree.
- **Verify**: `meson introspect build --tests | python3 -c "import json,sys;print(sorted(t['name'] for t in json.load(sys.stdin)))"`.

## other (meson files, packaging, debian, jni, tools, root files)

### O1. Update both packaging manifests whenever meson installs (or stops installing) a file
- **Rule**: When you add, remove or rename an installed header or library (`nntrainer_headers`, `install_headers`, `install: true`), change `debian/nntrainer-dev.install` (or the matching `debian/*.install`) and the `%files` lists in `packaging/nntrainer.spec` in the same commit. When you drop a dependency, also delete its `debian/<dep>.install`, `packaging/<dep>.tar.gz` and its spec `Source`/`%files` lines.
- **Why**: pdebuild runs `dh_missing --fail-missing` and GBS fails on "Installed (but unpackaged) file(s)". Both fail the other way too, when a listed file is not installed.
- **Evidence**: commits e99dc33f0 (driver_version.h missing from both manifests broke GBS and pdebuild), 2c199b8b0 ("an installed header with no manifest entry fails the Debian ... and Tizen RPM jobs, and a manifest entry with nothing installed fails them the other way round"), 3c331fb26; myungjoo 3513#discussion_r2381046988 (delete `debian/ruy.install`); djeong20 3429#discussion_r2322284399 (delete `packaging/ggml.tar.gz`); djeong20 2930#discussion_r1964844269 and 2930#discussion_r1968857778 (headers in `nntrainer-dev.install`).
- **Strength**: 3 PRs (myungjoo, djeong20) + 3 fix commits.
- **Verify**: "Build Test - Ubuntu Pdebuild" and "Build Test - Tizen/GBS" CI. Locally: `meson install -C build --destdir /tmp/x` and diff the installed header list against `debian/nntrainer-dev.install`.

### O2. Do not change `meson_options.txt` defaults in a feature PR, and default new backend options to off
- **Rule**: Leave existing defaults alone unless changing the default is the topic of the commit; do not flip `enable-test` or `enable-logging` to get a local build through. New backend or vendor options (`enable-qnn`, `enable-htp`, `enable-cuda`, `enable-biqgemm` and the like) default to `false`. Path options get no machine-specific default.
- **Why**: a changed default silently changes every platform build and package.
- **Evidence**: EunjuYang 3441#discussion_r2300218623 ("Please revert these"); EunjuYang 3115#discussion_r2041717267 ("Is it intended change?") and 3115#discussion_r2041716376; EunjuYang 2840#discussion_r1897517036 (`enable-qnn` default false); 3367#discussion_r2240568962 ("Why disabling tests and logging?"); SeoHyungjun 3088#discussion_r2025930375 (hard-coded biqgemm path).
- **Strength**: 5 PRs; EunjuYang 3 of them. Upstream: every accelerator option defaults to false.
- **Verify**: `git diff <base>...HEAD -- meson_options.txt` shows only added lines.

### O3. Scope every compiler flag and define to the narrowest meson condition
- **Rule**: Put platform- or ISA-specific flags under the exact condition: `get_option('platform') == 'android'`, `host_machine.cpu_family()` (`x86` is 32-bit, `x86_64` is 64-bit), or the compiler id. Do not use `add_project_arguments` for something one backend needs, because it reaches every target, the tests and the subprojects. Vendor-SDK options (HexKL/QNN) mean Qualcomm only, not every ARM target. Tizen targets OpenCL 2.0. Android always configures with `-Denable-fp16=true` (`tools/package_android.sh`), so every `arm-arch` choice must work with fp16 on.
- **Why**: a flag set too broadly broke x86 CI, a 32-bit build, the Tizen OpenCL build and the armv8-a Android build.
- **Evidence**: djeong20 3176#discussion_r2096748552 ("should only be added when the platform is Android"); myungjoo 3146#discussion_r2067957341 ("Don't add -march=x86-64 for x86"); djeong20 3255#discussion_r2160574811 (Tizen has no OpenCL 3.0); jaemini-shin 4162#discussion_r3680862532 ("This will affect through whole project") and 4162#discussion_r3680796723 ("Only Qualcomm device"); dlwlzzero 4439#discussion_r4384918496 (package_android forces fp16); commit 94812c749 (`-march=native` conflicted with `-mavx2 -mfma` on CI).
- **Strength**: 5 PRs, myungjoo, djeong20 and jaemini-shin, + 1 fix commit.
- **Verify**: configure the platforms you touch: `meson setup b -Dplatform=android ...` via `tools/package_android.sh [--arm-arch=...]`; Ubuntu gcc and clang CI legs.

### O4. Prefer the platform package for a third-party dependency, and pin any wrap fallback
- **Rule**: Get a new dependency with `dependency('<pkg>', required: false)` (or `fallback:`) so distro builds link the platform-supplied library. Use a `subprojects/<dep>.wrap` only as the fallback, pinned to a fixed revision, with no recursive clone, and with the wrap file under `subprojects/`.
- **Why**: pdebuild (Ubuntu) and gbs/obs (Tizen) cannot reach external repositories at build time.
- **Evidence**: myungjoo 2883#discussion_r1924810560 ("pdebuild ... and gbs/obs ... does NOT allow referencing external repo ... restrict the usage of external repos to the case that really needs it"); dkjung 3112#discussion_r2041255109 ("fixing to a specific version"); myungjoo 2922#discussion_r1960734714 ("You don't need recursive clone"); djeong20 2952#discussion_r1958969805 (move `*.wrap` under `subprojects/`); piotrrak 2952#discussion_r1960860193 / 2952#discussion_r1960862051 (`required: false` / `fallback`).
- **Strength**: 4 PRs; myungjoo stated it outright.
- **Verify**: pdebuild and GBS CI (both run with no network).

## docs

No rule qualifies: 3 comments in 2 PRs (a stale example file reference and wording nits). The cross-unit "English only" rule is under Global.

## Global (belongs in root AGENTS.md, not a unit file)

### G1. Give every C/C++ file the standard header, and document every function with `/** */`
- **Rule**: First line `// SPDX-License-Identifier: Apache-2.0`, then a `/** ... */` block with `Copyright`, `@file`, `@date`, `@brief`, `@see`, `@author`, `@bug`. `@file @brief @author @bug` are required in the leading comment. Every multi-line comment starts with `/**`. Doxygen tags are lowercase. Classes and functions in headers carry `@brief`, plus `@param`/`@retval` where applicable.
- **Why**: `static.check.yml` runs `doxygen-tag.sh <list> 1`, which fails on a missing header tag, a `/*` multi-line comment or an uppercase tag. Reviewers flag missing `@brief`/`@param` constantly.
- **Evidence**: jijoongmoon 3739#discussion_r2840879157 ("please use SPDX"); jaemini-shin 4086#discussion_r3612899647 ("Multi line comments for doxygen tag must be started with `/**`"); skykongkong8 3307#discussion_r2212557063; jaemini-shin 4105#discussion_r3662372216; jijoongmoon 3221#discussion_r2127587019; commits f89779e28, 05391baa9 ("add missing doxygen tags").
- **Strength**: 10+ PRs across units, 5 maintainers. 840 of 877 C/C++ files under `nntrainer/ test/ api/` carry SPDX.
- **Verify**: `git diff --name-only <base>...HEAD > /tmp/l && bash .github/workflows/static.check.scripts/doxygen-tag.sh /tmp/l 1`.

### G2. End every text file with one newline and no trailing whitespace, including `.github`
- **Rule**: This applies to every file type (yml, sh, json, txt, CHANGES, .gitignore, meson), not only C/C++. Run `git diff --check <base>...HEAD`. `cpp_linter.yml` runs that check but excludes `.github/` and `Applications/CausalLM/third_party/`, so check workflow files by hand. Check the final newline with `tail -c1 <file> | xxd`.
- **Why**: the most frequent nit in the data. `newline.sh` inspects only the last commit's patch, so it misses earlier commits in a PR.
- **Evidence**: jijoongmoon 3739#discussion_r2840880917, 3759#discussion_r2915178593; EunjuYang 3616#discussion_r2608886249, 3132#discussion_r2057488999; baek2sm 3548#discussion_r2768031438; skykongkong8 3299#discussion_r2191609491; songgot 3605#discussion_r2590952174, 3558#discussion_r2535844961 (trailing spaces in yml); commit 760f7be5b (added `git diff --check`).
- **Strength**: 13 PRs (newline) + 2 (yml whitespace), 6+ maintainers.
- **Verify**: `git diff --check <base>...HEAD` (no output means it passes).

### G3. Never commit a personal or absolute path
- **Rule**: Do not put `/home/<user>/...`, `~/Downloads/...` or a local SDK path in any committed file: configs (`nntr_config.json`), shell scripts, meson option defaults, tests. Use paths relative to the model or source dir, or an environment variable override with no personal default.
- **Why**: such a file breaks on every other machine. CI does not catch it: `static.check.scripts/hardcoded-path.sh` tests an unset `$VIOLATION` and compares the literal string `"failed"`, so it can never fail.
- **Evidence**: Seunghui98 3979#discussion_r3433159552 ("change to a relative path, not your local path"); Seunghui98 4236#discussion_r3747786147 (`$HOME/Downloads/hexkl_addon`); SeoHyungjun 3088#discussion_r2025930375; myungjoo 2833#discussion_r1891258293 (use `prefix`/`includedir`, cross-compile); commit 1624c58bb (removed `/home/jwon/...` from 11 fixture configs).
- **Strength**: 4 PRs + 1 fix commit; myungjoo among the reviewers.
- **Verify**: `git diff <base>...HEAD | grep -nE '^\+.*(/home/|/Users/|Downloads/)'`.

### G4. Write comments, strings, scripts and docs in English
- **Evidence**: DonghakPark 3254#discussion_r2151144002 ("Please translate to English", test); EunjuYang 3583#discussion_r2597473423 ("Please write it in English", .github js).
- **Strength**: 2 PRs, 2 maintainers. Short rule. Agents working for Korean-speaking users are at real risk of breaking it.
- **Verify**: `git diff <base>...HEAD | grep -nP '^\+.*[\x{AC00}-\x{D7A3}]'`.

### G5. Gate fp16 code on `ENABLE_FP16` and do not assume `_FP16` is a native float
- **Rule**: Guard `_FP16` code with `#ifdef ENABLE_FP16`, which follows the meson `enable-fp16` option. Do not use ISA or platform macros (`__AVX2__`, `__ARM_NEON`) for this. `_FP16` can be `__fp16`, `_Float16` or the uint16-backed `nntrainer::Half` wrapper (MSVC, `-Dfp16-impl=wrapper`), so do not rely on implicit float promotion or on `constexpr` arithmetic. Public headers must still compile when `ENABLE_FP16` is undefined.
- **Why**: the wrapper leg (`ubuntu_clean_meson_build` `-Denable-fp16=true -Dfp16-impl=wrapper`, `windows_msvc_fp16.yml`) and fp16-off builds break otherwise.
- **Evidence**: jaemini-shin 3820#discussion_r3205746483 and 3820#discussion_r3426942013 (use `#if defined(ENABLE_FP16)`, not an AVX2 guard); jijoongmoon 3823#discussion_r3321277870 (api header without ENABLE_FP16); djeong20 3456#discussion_r2485601791 ("`_FP16` can be represented as uint16_t on Windows"); jaemini-shin 4000#discussion_r3526749987; commits 304bab086, da1fd28b5, d611de499.
- **Strength**: 4 PRs + 3 commits; jijoongmoon, jaemini-shin and djeong20.
- **Verify**: build `-Denable-fp16=false`, `-Denable-fp16=true` and `-Denable-fp16=true -Dfp16-impl=wrapper`, and run the unittests.

### G6. Write `(std::min)(a, b)` / `(std::max)(a, b)` in code that Windows builds can reach
- **Rule**: In code reachable from `windows.h` (anything that includes OpenCL headers, public api headers, `numeric_limits<...>::min/max`), wrap the callee in parentheses instead of relying on `NOMINMAX`.
- **Why**: the windows.h `min`/`max` macros break the MSVC build.
- **Evidence**: commits 1a0c06dcd and d611de499 (jijoongmoon, 2026-08/09).
- **Strength**: 2 fix commits by a maintainer. Weak by count, but not derivable and costly (Windows CI only).
- **Verify**: "Build Test - Windows Meson" / "Windows Meson fp16 (Half wrapper)" CI.

### G7. Keep unrelated edits and submodule pointer moves out of the PR
- **Rule**: Before you commit, check `git diff --cached --stat --submodule`. Do not commit a moved submodule pointer (`nntrainer-windows-resource`, `subprojects/*`, `Applications/CausalLM/xgrammar`), reformatting of untouched lines, or stray option changes, unless they are the topic of the commit. This refines the root "one commit = one topic" rule.
- **Evidence**: myungjoo 3184#discussion_r2098080424 ("Having unrelated changes in a PR will confuse you in the future"); djeong20 3611#discussion_r2600696878 (outdated `nntrainer-windows-resource` submodule committed); EunjuYang 3813#discussion_r2986317489 ("indentation changes are unrelated ... revert"); EunjuYang 3441#discussion_r2300218623.
- **Strength**: 4 PRs, 3 maintainers.

### G8. CI coverage gaps an agent must cover locally (build-system scope)
- **Rule**: PR CI does not run the following, so run them or state that you did not:
  1. OpenCL code is compiled (`ubuntu_opencl_build.yml`, werror) but its tests are never run.
  2. The `test/jni` Android tests are never built (see T2).
  3. GBS runs unit tests on x86_64 only (`unit_test 0` for armv7l/aarch64).
  4. FSU runs only in the dedicated Ubuntu leg.
  5. CausalLM tests run only in the `-Denable-fp16=false` leg.

  The project default is `werror=true` with `warning_level=1`, so an unused variable fails the build.
- **Evidence**: commits c2bef4e04 ("an OpenCL-only build break can sit in the tree with every check green"), e811bb427, ba859f100 (OpenCL tests did not compile on main); EunjuYang 3812#discussion_r2986365202 (unused vars "may emit warnings"); `gbs_build.yml` matrix.
- **Verify**: `meson setup build -Denable-opencl=true && ninja -C build && meson test -C build --suite unittests` on a machine with an OpenCL device; `tools/android_test.sh`.

## Root AGENTS.md corrections

1. **`check_count` does not count tests.** `check_count.yml` ("Check Review") runs on `pull_request_review: submitted` and counts unique APPROVED reviewers. `labeler.yml` then sets the "Need Review" label. The line "If you add or remove tests, expect `check_count` to react — update counts accordingly" is wrong and should be deleted. The only test-presence gate is the `required_tests` set (`unittest_causallm_models`, `unittest_mha_core`) in `ubuntu_clean_meson_build.yml` (C2). The table entry "review/count checks ... run on every pull request" is also wrong for this check.
2. **`pr-desc` neither validates nor manages the PR body.** It runs only when the PR carries the `ai-describe` label. It then posts a gpt-4o-mini summary as a PR comment ("never edit PR body"). Nothing in CI checks the template. Keep the template rule, but drop "The `pr-desc` workflow manages/validates the PR description."
3. **minja is not in `subprojects/`.** It is vendored at `Applications/CausalLM/third_party/minja` (with `nlohmann`), and `cpp_linter` excludes that directory. `subprojects/` holds git submodules (iniparser, ruy, OpenBLAS, googletest, benchmark), ONNXConverter and `*.wrap` files. Add `Applications/CausalLM/third_party/` and the `Applications/CausalLM/xgrammar` submodule to the "do not edit" list.
4. **The format gate also covers whitespace.** `cpp_linter.yml` runs `git diff --check base...HEAD` over every changed file, excluding `.github/` and `Applications/CausalLM/third_party/`, before the clang-format-14 check. Add `git diff --check <base-sha>...HEAD` to Quick commands (G2).
5. **Undocumented static-check requirements.** "Static checkers and verifiers" is described only as "static analysis / verifiers". In fact it enforces the doxygen file header (`@file @brief @author @bug`, `/**` comments, lowercase tags; G1), a newline at EOF, no executable bit on sources, shellcheck, pylint, cppcheck, flawfinder, a 30 MiB file size limit and Signed-off-by. It runs only for PRs into `main`. Agents cannot guess G1, so name it.
6. **Not every gate runs on every PR.** Build jobs go through `.github/actions/check-rebuild` and skip PRs that change only `*.md/png/webp/css/html`. GBS, pdebuild and Android also rebuild for `packaging/`, `debian/` and `jni/` respectively. The build matrix list omits "Ubuntu Meson (OpenCL)" (compile only, werror, no tests) and "Windows Meson fp16 (Half wrapper)". It also omits that the Ubuntu Meson job has four legs (fp16 off, on, wrapper, FSU) and that CausalLM is configured only in the fp16-off leg.
7. **The test commands are incomplete for Android.** `ninja -C build test` covers only meson tests. Android tests are a separate hand-maintained list (`test/jni/Android.mk`) built only by `tools/android_test.sh`, never in PR CI. Add that line (T2). CI itself runs `meson test -C build --suite unittests --print-errorlogs`.
8. **"Stay cross-platform" has no concrete pointers.** The two traps found in the evidence are G5 (`_FP16` may be the `nntrainer::Half` wrapper) and G6 (windows.h min/max). Packaging (O1) is the most common build break and is absent from the root file.
9. **Upstream bug, not an AGENTS.md wording issue.** `hardcoded-path.sh` can never fail (G3). Worth a separate upstream fix; until then the root file should not imply CI catches personal paths.

## Rejected

- Negative test cases and the `_p`/`_n` name suffix (3633, 3645, 2793): generic "write more tests". The suffix is used in 1095 of 1660 named tests but in almost none of the 2026 files (e.g. unittest_ddtree 0/22, kv_cache_manager 0/20), so the code contradicts it as a current rule.
- Float comparison style (bitwise memcpy 3758, check every element 3685, MSE comparison 3575): generic advice, 1 PR each.
- Cover ubuntu 22.04 and 24.04 / windows-2022 (3157, 3199): contradicted by a7ea056e7 "Reduce PR build jobs" (PR matrix is 22.04 only).
- `.gitignore` order for `Applications/CausalLM/res/<model>/*` with `!` exceptions (3976, 3996): one reviewer, CausalLM-specific; belongs to the causallm group if anywhere.
- Explicit dims in `nntrainer_test_util.h` helpers, `enum class` and `DEFAULT` in `common.h`, pure virtual for new `Model` methods, default arguments on virtuals: 1 PR each.
- Typos and wording nits (docs, ci, test): not rules.
- Merge per-arch `tools/cross/*.json` files, run Python once in `package_android.sh` (3615): one-off refactor comments.
- Doc and tooling version notes (meson >= 0.55 in getting-started, node version for the visualizer): one-off, and visible in `meson.build`.

## Unit verdict

| Unit | Verdict | Rules |
|---|---|---|
| test | file worthwhile | 6 (T1–T6; T6 medium) |
| api | file worthwhile | 3 (A1–A3; A3 medium) |
| ci (.github) | file worthwhile, small | 2 (C1, C2); could fold into root instead |
| other | worthwhile, split by location: O1 → `packaging/` + `debian/` (or root); O2–O4 → root AGENTS.md (they govern root `meson.build`/`meson_options.txt`) | 4 (O1–O4) |
| docs | skip | 0 |
| Global | add to root | 8 (G1–G8) + 9 corrections |
