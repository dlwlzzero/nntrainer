# Brief: harvest agent rules for nntrainer from review evidence

Goal: we will add per-directory `AGENTS.md` files to upstream nntrainer
(github.com/nntrainer/nntrainer, main @ 640ac6e0e, local ref
`refs/remotes/upstream/main` in /home/j2z0-lee/nntrainer). Each file holds only
the rules an AI coding agent must follow when editing that directory, beyond the
global rules already in the root `AGENTS.md`. Your job is the evidence step: turn
real maintainer review feedback into candidate rules. You write no repo files.

Inputs (all in /tmp/claude-1002/-home-j2z0-lee-nntrainer/13ec36c8-5404-4632-93c6-510155e2dbbc/scratchpad/ev/):
- `unit_<name>.json`: human reviewer top-level inline review comments on upstream
  PRs, 2026-06-19 .. 2026-11-05, excluding bots and PR-author self-comments.
  Fields: pr, title, merged, reviewer, path, line, body, url.
- `AGENTS.root.md`: the existing root AGENTS.md (global rules; do not repeat them).
- `coding-convention.md`: docs/coding-convention.md (do not repeat it either).
- Source of truth for code: `git -C /home/j2z0-lee/nntrainer show refs/remotes/upstream/main:<path>`
  and `git -C /home/j2z0-lee/nntrainer log refs/remotes/upstream/main --since=2026-03-01 -- <paths>`.
  Do NOT read the working tree (it is a fork branch with unrelated changes).

Method:
1. Read every comment in your unit files. Cluster them by the underlying rule.
2. Also scan `git log refs/remotes/upstream/main --since=2026-03-01` subjects for
   your paths: fix / revert / "build break" commits reveal rules nobody wrote down.
   Open a few to confirm the cause.
3. Verify each candidate against upstream main code when it names a function,
   macro, file, or build option. Drop it if the code contradicts it.

A candidate rule qualifies only if ALL hold:
- Recurring: raised in >= 2 different PRs, OR stated as a convention by a
  maintainer (myungjoo, jijoongmoon, EunjuYang, djeong20, skykongkong8, baek2sm,
  DonghakPark, dkjung, jaemini-shin, songgot and similar regular reviewers).
- Not derivable by reading the code or a config file in one lookup.
- Changes agent behaviour versus a competent default (generic advice like
  "handle errors" or "write tests" fails).
- Generic upstream nntrainer, not one fork's experiment.

Output: write `/tmp/.../scratchpad/ev/rules_<group>.md` (full path given in your
task) with:
- `## <unit>` per unit, then per candidate rule:
  - **Rule** phrased positively as what to do, one or two lines.
  - **Why** in one line (the failure it prevents).
  - **Evidence**: 2-5 comment URLs or commit SHAs, with the reviewer name.
  - **Strength**: count of distinct PRs; note if a maintainer stated it outright.
  - **Verify**: the test target / command that checks it, if any.
- `## Global (belongs in root AGENTS.md, not a unit file)`: rules you found that
  span units.
- `## Rejected`: one line each for frequent themes you dropped and why.
- `## Unit verdict`: per unit, "file worthwhile" or "skip", with rule count.

Be relentless on evidence and stingy on rules: 5 strong rules beat 20 weak ones.
Report back in under 200 words: path of your output file, rule counts per unit,
and the three strongest rules.
