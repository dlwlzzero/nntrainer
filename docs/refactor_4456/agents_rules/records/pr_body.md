## Dependency of the PR

None.

## Commits to be reviewed in this PR

<details><summary>[docs] AGENTS.md: correct the CI facts and add rules for every change</summary><br />

Several statements in AGENTS.md did not match the workflows, so agents
acted on them wrongly:

- check_count counts approving reviewers on review submission; it does
  not count tests, so "update counts" when tests change was wrong.
- pr-desc never validates the PR body; it only posts an AI summary
  comment when the PR has the ai-describe label.
- minja lives in Applications/CausalLM/third_party/, not subprojects/.
- cpp_linter also runs git diff --check (whitespace, final newline) on
  every changed file except .github/ and CausalLM third_party.
- clang-format-14 -i reformats whole files; reviewers reject that churn,
  so recommend git clang-format-14 <base> only.

Add the rules that reviewers raised across many PRs (file hygiene,
doxygen file header, data-type branches, packaging manifests, meson
options, personal paths, logging, size casts), the concrete
cross-platform traps (_FP16 Half wrapper, windows.h min/max), the
static.check contents, and what PR CI does not cover (CausalLM only in
the fp16-off leg, Android unit tests, OpenCL at runtime).

Each rule comes from upstream review comments or fix commits from
2024-11 to 2026-11 and was checked against the workflows on main.

**Self evaluation:**
1. Build test: [ ]Passed [ ]Failed [X]Skipped (documentation only)
2. Run test: [ ]Passed [ ]Failed [X]Skipped (documentation only; `git diff --check` clean)

Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>

</details>

<details><summary>[CausalLM] Add AGENTS.md with rules for agents editing the app</summary><br />

Add Applications/CausalLM/AGENTS.md and a directory-rules table in the
root AGENTS.md that points to it. The file holds the rules that recur in
CausalLM reviews and fix commits:

- weight files are positional: converter order equals layer-creation
  order (regressed and reverted twice in 2026)
- a new source goes into every build list (in-tree meson,
  app_build/meson.build, jni/Android.mk, model_registry.cpp)
- incremental_forwarding(from, to) uses absolute positions, and the
  shared is_prefill test
- keep every dtype/arch branch in step; CI tests CausalLM only on x86
  with fp16 off
- one level of parallel_for (nesting deadlocks the thread manager)
- public models only; res/<model>/ holds sources only
- honour BATCH_SIZE or reject batch > 1

**Self evaluation:**
1. Build test: [ ]Passed [ ]Failed [X]Skipped (documentation only)
2. Run test: [ ]Passed [ ]Failed [X]Skipped (documentation only; `git diff --check` clean)

Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>

</details>

### Summary

- Fix statements in the root `AGENTS.md` that contradict the workflows (`check_count`, `pr-desc`, minja location, `cpp_linter` whitespace check, whole-file clang-format), and add the rules reviewers raise across many PRs.
- Add per-directory rules, starting with `Applications/CausalLM/AGENTS.md`, linked from a table in the root file. If the format works for reviewers, follow-up PRs add `nntrainer/tensor`, `cpu_backend`, `opencl`, `layers`, `nntrainer`, `test` and `api`.
- Every rule comes from upstream inline review comments (2024-11 to 2026-11) or fix/revert commits, and each code reference was checked against `main`.

Also found, not fixed here: `.github/workflows/static.check.scripts/hardcoded-path.sh` can never fail (`$VIOLATION` is never set and `"failed" = "1"` compares two literals).

Signed-off-by: dlwlzzero <dlwlzzero@gmail.com>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
