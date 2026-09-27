# Diagram spec for docs/htp (hand-drawn inline SVG)

Every figure in `docs/htp/*.html` is a hand-written inline SVG inside one
`<figure>…<svg>…</svg><figcaption>…</figcaption></figure>`. There is no
Mermaid, no chart library and no CDN. This spec keeps new figures in the
same family as the existing ones.

## Content rules

- Every number comes from a source (BENCHMARK.md, LEDGER.md, a measurement handoff, the code). Do not invent numbers.
  Keep labels short and move long explanations into the `<figcaption>`.
- Draw the mechanism: which processor/memory each step is on, what data moves, arrow labels.
- Pick the clearest form:
  - a share of a whole → a single horizontal 100 % stacked bar or a sorted horizontal bar chart with values
    printed at the bar ends.
  - a series → a proper bar/line chart with value labels on bars, y-axis ticks, gridlines,
    and any target line drawn dashed and labelled.
  - a schedule → a timeline with a labelled time axis; lanes grouped by section.
  - calls between parties → lifelines as vertical lines with headers; numbered messages as
    horizontal arrows with labels; keep the numbering if the prose refers to step numbers.
  - a flow → a laid-out box diagram on a grid; subgraphs as lightly filled regions with
    a title in the top-left corner.

## Visual rules (all diagrams must look like one family)

- `<svg viewBox="0 0 W H" role="img" aria-label="…" xmlns="http://www.w3.org/2000/svg"
  font-family="-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-size="13">`.
  W at most 880 (page column is ~830 px). Do not set width/height attributes (CSS scales it).
  H fitted to content — no big empty bands.
- No `<script>`, `<style>`, `<foreignObject>`, no external refs. Marker ids must be unique across
  the whole site: prefix them with the file key, e.g. `id="arch1-arrow"`.
- Text 12–13 px (titles 14 px bold, small notes 11 px). Nothing below 11 px. Every label must fit
  inside its box — size boxes to text (approx 7 px per char at 13 px). Multi-line labels are
  separate `<text>`/`<tspan x=… dy="1.3em">` lines.
- Align to a grid: shared baselines, equal gaps, straight orthogonal or straight-line arrows;
  avoid arrows crossing text or boxes.
- Palette (literal hex, page is light-theme only):
  - ink `#0b0b0b` (main text), ink2 `#52514e` (secondary text, arrows), muted `#898781`,
    grid lines `#e1e0d9`, box fill `#fcfcfb`, box stroke `#c3c2b7`
  - **NPU / DSP / HTP side**: stroke `#2a78d6`, fill `#e9f1fb`
  - **CPU / ARM side**: stroke `#eb6834`, fill `#fdeee6`
  - **memory (DDR, VTCM, ION)**: stroke `#8a7f5c`, fill `#f4efe1`
  - good `#006300`, warn `#b35c00`, bad `#d03b3b` — only for verdicts/targets.
  - Chart series: primary `#2a78d6`, second `#eb6834`; target lines dashed `#52514e`.
  Use the CPU/NPU/memory colours consistently whenever a box belongs to one of them;
  neutral boxes use the box fill/stroke.
- Boxes: `rx="6"`, stroke-width 1.2. Group regions: `rx="10"`, fill a very light tint of the
  group colour (or `#f6f5f1` for neutral), stroke same hue, no dashes unless meaningful.
- Arrows: stroke `#52514e`, width 1.3, `marker-end` with a small filled triangle marker.
  Arrow labels 11–12 px, `#52514e`, placed beside the line on a small `#f9f9f7` background
  rect if they overlap anything.
- `<figcaption>`: one or two sentences stating what the figure shows (the claim), plus any
  explanation that was squeezed out of the labels.

## Verify each figure visually

Render a preview and look at it before finishing:

```sh
# extract figure N of a page into a standalone .svg and render a PNG thumbnail (macOS)
python3 - page.html N <<'PY' > /tmp/fig.svg
import re,sys; t=open(sys.argv[1]).read(); print(re.findall(r'<svg.*?</svg>',t,re.S)[int(sys.argv[2])-1])
PY
qlmanage -t -s 1200 -o /tmp /tmp/fig.svg >/dev/null 2>&1   # then look at /tmp/fig.svg.png
```

(The thumbnail is a square canvas; the drawing sits at the top-left.) Fix overlaps, clipped
text, text outside boxes, and crossing arrows, then re-render. Check the SVG is well-formed:
`python3 -c "import xml.dom.minidom,sys;xml.dom.minidom.parse(sys.argv[1])" /tmp/fig.svg`.

