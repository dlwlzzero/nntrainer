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
# The "On this page" list: inline on narrow screens, a fixed right rail on wide ones.
TOC_CSS = ("/*toc*/.toc ul{list-style:none;padding-left:0}.toc li.l3{margin-left:1.1em;font-size:.93em}"
           ".toc a{text-decoration:none}.toc a.on{color:var(--ink);font-weight:600}"
           "@media (min-width:72em){main{margin-left:max(1em,calc(50% - 36em));margin-right:0}"
           "nav.toc{position:fixed;top:3.2em;left:calc(50% + 17em);width:17em;max-height:calc(100vh - 4.5em);"
           "overflow-y:auto;margin:0}nav.toc summary{pointer-events:none;list-style:none}"
           "nav.toc summary::-webkit-details-marker{display:none}}/*end toc*/")
# Highlights the section in view; the list works without it.
SPY = ('<script id="tocspy">(()=>{const a=[...document.querySelectorAll("nav.toc a")],'
       'm=new Map(a.map(x=>[decodeURIComponent(x.hash.slice(1)),x]));'
       'const o=new IntersectionObserver(es=>{for(const e of es)if(e.isIntersecting){'
       'a.forEach(x=>x.classList.remove("on"));m.get(e.target.id)?.classList.add("on")}},'
       '{rootMargin:"0px 0px -70% 0px"});'
       'm.forEach((_,id)=>{const h=document.getElementById(id);h&&o.observe(h)})})()</script>')


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


def toc(t):
    li = []
    for lvl, hid, body in re.findall(r'<h([23]) id="([^"]+)"[^>]*>(.*?)</h\1>', t, re.S):
        text = re.sub(r'<[^>]+>', '', re.sub(r'<a class="hd".*?</a>', '', body)).strip()
        li.append(f'<li class="l{lvl}"><a href="#{hid}">{text}</a></li>')
    return ('<nav class="toc"><details open><summary>On this page</summary><ul>'
            + "".join(li) + '</ul></details></nav>')


for i, (key, _) in enumerate(ORDER):
    p = D / f"{key}.html"
    t = p.read_text()
    t = re.sub(r'<nav class="top">.*?</nav>|<!--NAV-->', lambda m: nav(key), t, count=1, flags=re.S)
    t = re.sub(r'<div class="pager">.*?</div>', lambda m: pager(i), t, count=1, flags=re.S)
    if re.search(r'<nav class="toc">.*?</nav>|<!--TOC-->', t, re.S):
        t = re.sub(r'<nav class="toc">.*?</nav>|<!--TOC-->', lambda m: toc(t), t, count=1, flags=re.S)
    else:
        t = t.replace("</h1>", "</h1>\n" + toc(t), 1)
    t = re.sub(r'/\*toc\*/.*?/\*end toc\*/\n?', '', t, flags=re.S)
    t = t.replace("</style>", TOC_CSS + "\n</style>", 1)
    t = re.sub(r'<script id="tocspy">.*?</script>\n?', '', t, flags=re.S)
    t = t.replace("</body>", SPY + "\n</body>", 1)
    if NAV_CSS not in t:
        t = t.replace("</style>", NAV_CSS + "\n</style>", 1)
    p.write_text(t)
    print("nav:", p.name)
