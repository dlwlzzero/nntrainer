# AGENTS.md — Guide for AI coding agents working on nntrainer

This file is the entry point for AI coding agents (Claude Code, etc.) working in
this repository. Read it before building, testing, committing, or opening a PR.

Detailed rules live in `docs/` — this file links to them rather than copying, so
the source of truth stays in one place.

## 📍 Directory rules (read before editing there)

Some directories carry their own `AGENTS.md` with rules that apply only there.
Before you edit a file under one of these paths, read that file too:

| Editing under | Also read |
|---|---|
| `Applications/CausalLM/` | [`Applications/CausalLM/AGENTS.md`](Applications/CausalLM/AGENTS.md) |

## ⚡ Non-negotiable rules (do these before every commit / PR)

- **Sign off every commit:** `git commit -s` (DCO). A `Signed-off-by:` line is
  required; commits without it fail the review check.
- **Mark agent-authored commits:** if an AI agent wrote the commit, add a
  `Co-authored-by:` trailer identifying the agent, e.g.
  `Co-authored-by: Claude <noreply@anthropic.com>`.
- **Format only the lines you changed, with clang-format 14:**
  `git clang-format-14 <base-sha>`, using the repo's [`.clang-format`](https://github.com/nntrainer/nntrainer/blob/main/.clang-format).
  CI (`cpp_linter`) checks only the changed lines, with version **14**.
  Running `clang-format-14 -i` on a whole file reformats untouched lines, and
  reviewers reject that churn. Do not clang-format `meson.build`.
- **Commit message format:** `[<component>] <short description>`.
  Examples: `[CausalLM] ...`, `[test] ...`, `[kv_cache_manager] ...`.
- **One commit = one topic.** Keep each commit focused; provide enough background.
  Touch only the lines your change needs: no reformatting, reordering, or
  comment deletion in untouched code, and no moved submodule pointer unless it
  is the topic. Check with `git diff --cached --stat --submodule`.
- **Do not edit vendored code:** `subprojects/` (submodules such as iniparser,
  ruy, OpenBLAS, googletest, benchmark, plus `*.wrap` files),
  `Applications/CausalLM/third_party/` (minja, nlohmann), and the
  `Applications/CausalLM/xgrammar` and `nntrainer-windows-resource`
  submodules. Changes there do not belong in nntrainer PRs.
- **Stay cross-platform.** This builds on Android, Tizen, Windows, and Yocto in
  addition to Ubuntu. Avoid Linux-only headers/APIs unless properly guarded.
  See "Cross-platform traps" below.

## 🧱 Rules for every change

- **File hygiene:** every text file you touch (C/C++, `meson.build`, `.json`,
  `.py`, `.yml`, `.sh`, `.md`, `.gitignore`) ends with exactly one newline and
  has no trailing whitespace. CI runs `git diff --check` but skips `.github/`,
  so check workflow files yourself.
- **File header:** every new C/C++ file starts with
  `// SPDX-License-Identifier: Apache-2.0`, then one `/** ... */` block with
  `Copyright`, `@file` (the real file name), `@date`, `@brief`, `@see`,
  `@author`, `@bug`. Multi-line comments start with `/**`; classes and
  functions in headers carry `@brief`. When you copy a file, rewrite `@file`,
  `@brief` and the include guard. Use include guards, not `#pragma once`.
- **Data types:** every `switch`/`if` on a tensor data type implements each
  type or throws; no empty `case`/`break`. Put FP16 code under
  `#ifdef ENABLE_FP16` with a throw in the `#else`.
- **Installed files:** when meson starts or stops installing a header or
  library, update `debian/nntrainer-dev.install` (or the matching
  `debian/*.install`) and the `%files` lists in `packaging/nntrainer.spec` in
  the same commit. Otherwise the pdebuild and Tizen GBS jobs fail.
- **Build options:** leave `meson_options.txt` defaults alone unless changing
  them is the topic of the commit. New backend or vendor options default to
  `false`. Scope each compiler flag or define to the narrowest condition
  (compiler id, `host_machine.cpu_family()`, `get_option('platform')`); avoid
  `add_project_arguments` for something one backend needs.
- **Paths:** commit only paths relative to the source or model dir, or an
  environment variable with no personal default. No `/home/<user>/...` or
  local SDK paths in code, configs, scripts, or tests. The CI hardcoded-path
  check does not catch these today, so reviewers do.
- **Logging:** use `ml_logi/ml_logw/ml_loge/ml_logd` from `nntrainer_log.h`,
  not `std::cout`; keep `<iostream>` out of headers.
- **Size arithmetic:** cast to `size_t` before multiplying sizes (CodeQL
  `cpp/integer-multiplication-cast-to-long`).

### Cross-platform traps

- `_FP16` may be `__fp16`, `_Float16`, or the uint16-backed `nntrainer::Half`
  wrapper (MSVC, `-Dfp16-impl=wrapper`). Write explicit conversions; implicit
  float promotion and `constexpr` arithmetic on `_FP16` break the wrapper
  build. Public headers must compile with `ENABLE_FP16` undefined.
- In code that `windows.h` can reach (OpenCL headers, public API headers),
  write `(std::min)(a, b)` / `(std::max)(a, b)`.
- Android builds (`tools/package_android.sh`) configure `-Denable-fp16=true`
  by default; Tizen GBS also builds 32-bit `armv7l`.

## 🔧 Quick commands

```bash
# First time only: pull submodules (required to build)
git submodule sync && git submodule update --init --depth 1

# Configure + build (from repo root)
meson build -Denable-transformer=true
ninja -C build

# Run unit tests (gtest based), as CI does
meson test -C build --suite unittests --print-errorlogs

# Android unit tests: a separate list in test/jni/Android.mk, never built by PR CI
tools/android_test.sh

# Format and whitespace, matching CI (changed lines only, against the PR base)
git clang-format-14 <base-sha>
git diff --check <base-sha>...HEAD
```

See [docs/getting-started.md](https://github.com/nntrainer/nntrainer/blob/main/docs/getting-started.md) for full build
dependencies and platform-specific builds (Tizen/GBS, Android, Windows).

## 📝 PR description rules (agents get this wrong most often)

Write the PR body using [`.github/PULL_REQUEST_TEMPLATE.md`](https://github.com/nntrainer/nntrainer/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
exactly:

- **One `<details>` block per commit**, each containing the commit message, a
  **Self evaluation** (Build test / Run test), and a `Signed-off-by:` line.
- A **Summary** section, also ending with `Signed-off-by:`.

No CI job checks the PR body; reviewers do. The `pr-desc` workflow only posts
an AI summary comment when the PR carries the `ai-describe` label.

## ✅ CI gates — the definition of "done"

A PR is mergeable only when these pass:

| Check | What it verifies |
|-------|------------------|
| `C++ Format Checker` (cpp_linter) | `git diff --check` (whitespace, final newline) on all changed files except `.github/` and `Applications/CausalLM/third_party/`, then clang-format 14 on changed C/C++ lines |
| `Static checkers and verifiers` (static.check) | PRs into `main`: doxygen file tags, newline at EOF, no executable bit on sources, file size ≤ 30 MiB, pylint, shellcheck, cppcheck, flawfinder, `Signed-off-by:` |
| `Check Review` (check_count) | number of approving reviewers; runs on review submission, not on code changes |
| `CodeQL Advanced` | security/code scanning |
| Build matrix | Ubuntu Meson (gcc & clang; fp16 off / on / Half wrapper / FSU legs), Ubuntu OpenCL (compile only), Android NDK, CausalLM Android, Tizen/GBS, Ubuntu Pdebuild, Windows (Meson / fp16 Half wrapper / on-ARM / cross-compile), Yocto |

Build jobs skip PRs that change only `*.md`, `*.png`, `*.webp`, `*.css` or
`*.html` files.

**What PR CI does not cover**, so run it yourself or say in the PR that you
did not:

- CausalLM is configured (`-Denable-transformer=true`) and tested only in the
  Ubuntu `-Denable-fp16=false` leg. That leg fails if `unittest_causallm_models`
  or `unittest_mha_core` disappears from `meson introspect --tests`
  (`required_tests` in `ubuntu_clean_meson_build.yml`).
- Android unit tests (`test/jni/Android.mk`, `tools/android_test.sh`).
- OpenCL kernels at runtime: no runner has a GPU.

## 🗂️ Repo map

- `api/` — public API definitions (`capi` = C-API, `ccapi` = C++-API)
- `nntrainer/` — all core NNTrainer code
- `Applications/` — examples and apps (e.g. CausalLM)
- `test/` — gtest unit tests, grouped into subdirectories
- `nnstreamer/` — NNStreamer sub-filter code for NNTrainer
- `jni/` — Android/Java build scripts
- `tools/` — developmental tools and scripts
- `packaging/` — Tizen RPM build scripts
- `debian/` — Debian/Ubuntu packaging
- `subprojects/` — vendored dependencies (**do not edit**)

## 📚 Deep dive (read only when needed)

- Coding convention → [docs/coding-convention.md](https://github.com/nntrainer/nntrainer/blob/main/docs/coding-convention.md)
- Contributing & merge process, sign-off details → [docs/contributing.md](https://github.com/nntrainer/nntrainer/blob/main/docs/contributing.md)
- Build & dependencies → [docs/getting-started.md](https://github.com/nntrainer/nntrainer/blob/main/docs/getting-started.md)
- Running test cases → [docs/how-to-use-testcases.md](https://github.com/nntrainer/nntrainer/blob/main/docs/how-to-use-testcases.md)
