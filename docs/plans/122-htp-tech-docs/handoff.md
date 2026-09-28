# #122 handoff: the docs/htp HTML doc set

State as of **2026-09-28**. Phase 1 is done and published; the Results rework and the page TOC (decisions 20–21) are in. Phase 2 is next.

## Where things are

| what | where |
|---|---|
| the doc set (the source) | branch `htp/122-tech-docs`, `docs/htp/*.html` + `docs/htp/img/wh-tile.svg` |
| this handoff, specs, tools | `docs/plans/122-htp-tech-docs/` on the same branch |
| published site | branch `htp_html`, `docs/htp/`; GitHub Pages (legacy build, `htp_html:/docs`) serves it at https://dlwlzzero.github.io/nntrainer/htp/ |
| issue | #122. Its body holds the agreed structure; keep it in sync |
| facts | `origin/htp_moe`: `docs/htp_moe/BENCHMARK.md` (numbers), `docs/htp_moe/LEDGER.md` (rules, verdicts), `docs/measurements/*.md` (sittings) |

Commits on `htp/122-tech-docs`:

- `7d14d9fa`: the Markdown chapters
- `8a7c8f8a`: HTML with redrawn SVG diagrams
- `c13a06a0`: phase 1
- the commit that adds this handoff

Publish commit on `htp_html`: `e0d8f8fb`.

The branch is based on `9fb1a3bc`, an old `htp_moe`. It only adds files under `docs/htp/` and `docs/plans/122-htp-tech-docs/`, so merging it into `htp_moe` later is conflict-free.

## Decisions made with the user (2026-09-27, do not re-ask)

1. **Structure.**
   - Four story pages come first:
     1. **Results** (`index.html`)
     2. **Communication** (`communication.html`)
     3. **Software** (`software.html`)
     4. **Optimizations** (`optimizations.html`)
   - Reference chapters follow: Weight format · HMX matmul · MoE kernel · Build & test · Measurement · Lessons · Roadmap.
   - `architecture.html` gets split into Communication and Software, then deleted.
   - `foundation.html` gets split into Communication and a smaller "Build & test" page.
   - `hmx-matmul`, `weight-format` and `moe-ffn` stay as deep dives. Where they overlap the story pages, link instead of repeating.
2. **Results is the index page.** It shows the MoE model's performance at a glance.
3. **Baseline = the current default path** (the VTCM feed). Optimizations tells how the code got there from upstream PR #4327.
4. **Snapshot = latest `htp_moe`.** Phase 1 used `a7c66dec`.
5. **Optimizations scope.** It covers the upstream PR #4327 prefill work plus the `htp_moe` decode work. Dropped attempts go in a short table only, linked to Lessons.
6. **Optimizations format, per change:** bottleneck → change (before/after figure, with the difference highlighted) → gain (same-sitting A/B tok/s and ms/token) → its place on a cumulative tok/s staircase at the top of the page. Cross-sitting bars show direction only; say so.
7. **Results metrics (all):**
   - tiles
   - NPU vs CPU prefill/decode
   - distance to the ≥ 50 tok/s goal
   - per-token time split
   - gen 64/512/1024
   - accuracy (text identical, bit-identical tests)
   - progress
   - test conditions
8. **Reader:** a SW engineer who knows ML inference but not Hexagon. Define HVX, HMX, VTCM, FastRPC, PD, ION and DMA on first use. Results should also read for management.
9. **Language:** English.
10. **Communication sections:** actors and memories → life of a session → life of one MoE call → the three ways weights reach the compute units → synchronization rules. Default path only.
11. **Software:**
    - A UML-style class diagram: ARM C++ on one side, the IDL boundary in the middle, DSP C structs and modules on the other.
    - A call-path sequence (`Lfm2MoELayer` → `HtpComputeOps::invokeMoeLayer` → stub → skel → `hexkl_mm_u8i4_moe_layer_run` → worker pool).
    - A code-map table. Group the ~40 IDL methods into 6 families; don't draw them all.
12. **The #119 per-token entry** (`NNTR_HTP_FORWARD`, off by default) appears **only** in Software (dashed border, tag "behind `NNTR_HTP_FORWARD`") and in Roadmap. It is not drawn on Communication.
13. **Hosting:** GitHub Pages via `htp_html`. Publish at each review gate.
14. **JS:** small inline JS is allowed (step-through, toggles). Every page must be fully readable with JS off.
15. **Merged work only** on the Results page.
16. **Order:**
    1. Results + Communication → publish → review (done).
    2. Software + Optimizations → publish → review.
    3. Fold the reference chapters.
    4. PR into `htp_moe`.
17. **Public safety** (the repo and Pages are PUBLIC):
    - **No device serials.** Say "Galaxy S25 Ultra (Snapdragon 8 Elite, Hexagon v79)". Both phones are the same model.
    - Give the unit-to-unit spread **only as a range**: NPU DMA 31–37 GB/s, MoE floor 13.0–15.5 ms. No "unit 1 / unit 2" or "phone 1 / phone 2" labels.
    - **No workstation paths**; use `$W/…`.
    - Binary md5s may stay.
    - Check with: `grep -nE 'R3CY|205Z|10WM|WM83|ZMND|/local/mnt|/home/|phone [12]' docs/htp/*.html` → must print nothing.
18. **The byte floor.** Use the doc set's count, not BENCHMARK's:
    - MoE: 484 MB at 31–37 GB/s = 13.0–15.5 ms.
    - CPU: 408 MB at ≈ 50 GB/s ≈ 8.2 ms.
    - Serial floor ≈ 21–24 ms ≈ 42–47 tok/s, so 50 tok/s is not reachable by bytes with today's serial split.
    - BENCHMARK's cycle-16 budget still says ≈ 6.5 ms (246 MB). Results carries a one-line note on the difference (lessons.html D1).

19. **Phase-2 proposals accepted (2026-09-28):**
    - Snapshot = latest `htp_moe` (`9f074a5e` at the time).
    - The Results headline stays #120 A until a new sitting.
    - PR #121's #120 B A/B goes into Optimizations.
    - #81/#82 go into Software, dashed, "landed, unwired, #130".
20. **Every page gets an "On this page" list** built by `nav.py` from the `h2`/`h3` ids. On a wide screen (≥ 72em) it is a fixed rail on the right, with a small scroll-spy script. On a narrow screen it is an inline list. Never hand-edit it; rerun `nav.py`.
21. **Results reads as one story, for management first:** four tiles → numbered sections (NPU vs CPU · time split · how we got here · accuracy · by gen · conditions). **No PR numbers, sitting ids or commit hashes in the body, the charts or the captions.** They live only in the collapsed "Sources and snapshot" box at the bottom. The goal-band chart was dropped; the split chart already shows the floor against 20 ms. Apply the same rule to the other story pages.
    - No summary box above the sections (user, 2026-09-28). Text uses the full column width, the same as the figures; do not cap `p`/`li` width.

## Phase 1: what was done

- **Markdown → HTML.**
  - The 9 md chapters became HTML (`README.md` → `index.html`), and the md files were removed; **the HTML is the source now**.
  - All 34 Mermaid diagrams were redrawn by hand as inline SVG (palette: blue = NPU/DSP, orange = CPU, tan = memory).
  - `img/wh-tile.svg` was restyled and its clipped caption fixed.
- **Results page** (5 charts). Its numbers come from BENCHMARK @ `8af1a347`:
  - #120 A: 35.97 / 35.96 / 35.17 tok/s decode; prefill means 487.6 / 459.1 / 448.8.
  - #94 s2 A: the CPU reference.
  - The cycle-16 budget: 27.81 ms = MoE 15.7 + transport 1.8 + outside ≈ 10.3.
  - The progress bars 18.31 → 23.80 → 27.09 → 27.89 → 35.96 come from different sittings and are captioned as such.
- **Communication page** (10 SVGs).
  - It has a 12-step JS step-through; the JS was syntax-checked only, **not run in a browser yet**.
  - The feed timeline (fig 6) is schematic: its per-expert compute split of ≈ 40/20 µs comes from moe-ffn §6 and was not measured.
  - Symbols were verified at `a7c66dec`.
- **The 8 other chapters were made public-safe:** serials, per-phone labels and paths were removed. The lessons DMA-anchor chart was redrawn in one colour.
- **One shared nav bar** on every page (`nav.py`).

## Open: raise these with the user at the start of phase 2

Questions the user has not answered yet:

1. **Review of phase 1:** ask for feedback on the two published pages.
2. *(Answered 2026-09-28, decision 19.)* **`htp_moe` moved** after phase 1 was written. It is at `dbaf24b5` as of 2026-09-27. Merged: PR #121 (the upstream sync, `b7d46b57`), #124, #125 (#82 M=1 small ops, kernels unwired), #126 (#81 M=1 attention + KV cache, unwired), #127 (#84 in-process host E2E build, `-Dhtp-inproc`), #128 (#89 generation(last 64) tok/s), and #129. #130 (wiring #82/#81 into the per-token table) is filed. Proposed answers, which need the user's OK:
   - Snapshot → the latest `htp_moe`.
   - The Results headline stays #120 A until a new sitting. BENCHMARK's rule is that the next A carries the sync and is read against #120 A.
   - PR #121's #120 B A/B goes into Optimizations. B vs A: prefill +12.4 / +10.2 / +6.2 %; decode +4.7 / +0.9 / −0.2 %; text identical; M>1 dsp 16 986 → 14 808 µs.
   - #81/#82 kernels go into Software, dashed, as "landed, unwired, #130", like #119.
3. **Serials elsewhere in the public repo:** `docs/htp_moe/BENCHMARK.md`, `LEDGER.md`, `docs/measurements/*`, and this branch's older commits still name the phones. Only the Pages doc set was cleaned. Ask whether to clean those too.

## Phase 2: what to build

- **`software.html`** per decision 11. Start from `codemap.md` and re-verify it against the new snapshot (#81/#82/#84 added files; `hvx_add_f32.c` `close()` changed).
  - ARM classes: `HtpContext`, `HtpBackend`, `ComputeOps`/`CpuComputeOps`/`HtpComputeOps` (with `ArenaChunk`, `ArenaEntry`, `StagingPool`), `HtpRpcBuffer`/`HtpRpcMemApi`, `HtpProfile`, `Lfm2MoeCausalLM`, `Lfm2MoELayer`.
  - DSP side: `nntr_hvx_session`, `hexkl_weight_u8i4` table, the MoE kernel + layout, `hexkl_dma_ring`, `hvx_worker_pool`, `hexkl_graph`.
  - Plus the call-path sequence and the code map.
- **`optimizations.html`** per decisions 5–6. Sources: `codemap.md` §"Optimizations landed", LEDGER, BENCHMARK, the handoffs. Changes in order:
  1. The upstream PR's prefill work (one call per MoE layer, the weight format/ION arena, prefill tuning).
  2. The transport fix (#88, PR #103: size-class staging + poll QoS).
  3. The HVX GEMV decode (#80/#101, PRs #86/#108).
  4. The prefetch lead + one-row loop (#105/#113, PR #115).
  5. The VTCM feed (#117, PR #118).
  6. The upstream sync (PR #121, if the user agrees).
  7. The per-token entry (#119, landed, not measured).
- Replace the "(in progress)" text in `index.html`'s Progress section with a link to `optimizations.html`.
- Extend `ORDER`/`CORE` in `nav.py` with `software` and `optimizations`, then run it.
- **Before publishing:** run `python3 docs/plans/122-htp-tech-docs/check.py docs/htp` and get `OK`. It parses every SVG, looks for duplicate ids, broken links and anchors, and runs the public-safety pattern.

## Tools in this folder

- `diagram-spec.md`: the figure rules (palette, sizes, markers, preview with `qlmanage` on macOS, XML check). Give it to anyone drawing a figure.
- `codemap.md`: the code map gathered at `a7c66dec` (classes, IDL groups, shared memory, sync, DSP modules, switches, landed optimizations).
- `template-head.html`: the page head plus CSS. A new page starts from it, keeping the `<!--NAV-->` placeholder.
- `nav.py`: rewrites the nav bar and prev/next pager on every page. Run it as `python3 docs/plans/122-htp-tech-docs/nav.py docs/htp`.
- `check.py`: the pre-publish check (SVG XML, duplicate ids, links/anchors, public safety).

The per-figure generator scripts and the Markdown converter of phase 1 were throwaway and are not kept; edit the HTML/SVG directly.

## How to publish

```sh
git fetch origin
git worktree add ../nntrainer-pages origin/htp_html -b htp_html-pub   # once per machine
cd ../nntrainer-pages && git pull --ff-only origin htp_html 2>/dev/null || git reset --hard origin/htp_html
mkdir -p docs/htp/img
cp <docs-worktree>/docs/htp/*.html docs/htp/ && cp <docs-worktree>/docs/htp/img/*.svg docs/htp/img/
git add docs/htp && git commit -s -m "[html] Publish the HTP backend docs (#122 …)" && git push origin HEAD:htp_html
gh api repos/dlwlzzero/nntrainer/pages/builds/latest -q '.status+" "+.commit'   # wait for "built"
```

If a page is deleted in the source (e.g. `architecture.html` in phase 3), delete it in `htp_html` too.

## Environment notes

- GitHub MCP writes return 403 on this repo; use the `gh` CLI (e.g. `gh issue edit 122 --body-file …`).
- Commit with `-s` and end with `Co-Authored-By: Claude …`.
