#!/usr/bin/env python3
"""Write one shared nav bar and prev/next pager into every docs/htp page."""
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
CORE = [
    ("index", "Results"),
    ("communication", "Communication"),
]
REF = [
    ("architecture", "Architecture"),
    ("foundation", "Foundation"),
    ("weight-format", "Weight format"),
    ("hmx-matmul", "HMX matmul"),
    ("moe-ffn", "MoE FFN"),
    ("measurement", "Measurement"),
    ("lessons", "Lessons"),
    ("roadmap", "Roadmap"),
]
ORDER = CORE + REF
NAV_CSS = "nav.top .sep{color:var(--muted);margin:0 .9em 0 0}"


def nav(here):
    def link(key, label):
        cls = ' class="here"' if key == here else ""
        return f'<a href="{key}.html"{cls}>{label}</a>'
    return ('<nav class="top">' + "".join(link(*c) for c in CORE)
            + '<span class="sep">|</span>'
            + "".join(link(*r) for r in REF) + "</nav>")


def pager(i):
    prev = ORDER[i - 1] if i > 0 else None
    nxt = ORDER[i + 1] if i + 1 < len(ORDER) else None
    left = f'← <a href="{prev[0]}.html">{prev[1]}</a>' if prev else ""
    right = f'<a href="{nxt[0]}.html">{nxt[1]}</a> →' if nxt else ""
    return f'<div class="pager"><span>{left}</span><span>{right}</span></div>'


for i, (key, _) in enumerate(ORDER):
    p = D / f"{key}.html"
    t = p.read_text()
    t = re.sub(r'<nav class="top">.*?</nav>|<!--NAV-->', lambda m: nav(key), t, count=1, flags=re.S)
    t = re.sub(r'<div class="pager">.*?</div>', lambda m: pager(i), t, count=1, flags=re.S)
    if NAV_CSS not in t:
        t = t.replace("</style>", NAV_CSS + "\n</style>", 1)
    p.write_text(t)
    print("nav:", p.name)
