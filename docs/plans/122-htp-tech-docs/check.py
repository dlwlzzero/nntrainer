#!/usr/bin/env python3
"""Pre-publish checks for docs/htp: SVG well-formedness, duplicate ids, links, public safety."""
import glob
import re
import sys
import xml.dom.minidom
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/htp")
UNSAFE = re.compile(r"R3CY|205Z|10WM|WM83|ZMND|/local/mnt|/home/|phone [12]")
bad = 0
ids = {}
for f in sorted(glob.glob(str(D / "*.html"))):
    t = Path(f).read_text()
    name = Path(f).name
    for i, s in enumerate(re.findall(r"<svg.*?</svg>", t, re.S), 1):
        try:
            xml.dom.minidom.parseString(s)
        except Exception as e:
            print(f"SVG {name} #{i}: {e}")
            bad += 1
    found = re.findall(r'\bid="([^"]+)"', t)
    for d in sorted({i for i in found if found.count(i) > 1}):
        print(f"DUP-ID {name}: {d}")
        bad += 1
    ids[name] = set(found)
    for n, line in enumerate(t.splitlines(), 1):
        if UNSAFE.search(line):
            print(f"UNSAFE {name}:{n}: {UNSAFE.search(line).group(0)}")
            bad += 1
for name in ids:
    t = (D / name).read_text()
    for h in re.findall(r'href="([^"]+)"', t):
        if h.startswith(("http:", "https:", "mailto:")):
            continue
        path, _, anchor = h.partition("#")
        target = path or name
        if target.endswith(".html") and target not in ids:
            print(f"BROKEN {name}: {h}")
            bad += 1
        elif anchor and target in ids and anchor not in ids[target]:
            print(f"ANCHOR {name}: {h}")
            bad += 1
        elif not target.endswith(".html") and not (D / target).exists():
            print(f"MISSING {name}: {h}")
            bad += 1
print("OK" if not bad else f"{bad} problem(s)")
sys.exit(1 if bad else 0)
