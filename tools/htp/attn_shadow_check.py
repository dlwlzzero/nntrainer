#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
@file   attn_shadow_check.py
@brief  dev/attn-shadow-170 only (#170, never merged): G4 of plan 170 on
        NNTR_ATTN_SHADOW / NNTR_LOGIT_SHADOW dumps.

    attn_shadow_check.py <ref dir (A's logits)> <run dir>...

Per run dir: the q heads (64 values each) of every tag-3 record whose HTP
attention output equals the CPU's fp16 attention of the same row bit for
bit, the layer ordinals and positions seen, the records whose HTP output
is all zero (a hook that wrote nothing), and the decode steps whose
logits.bin equals the reference's bit for bit.
"""
import os
import struct
import sys

import numpy as np

HD = 64
V = 128000  # LFM2.5 vocab: one decode step of logits.bin


def records(path):
    with open(path, "rb") as f:
        while True:
            h = f.read(20)
            if len(h) < 20:
                return
            tag, pos, ordinal, n_row, n_out = struct.unpack("<5I", h)
            f.read(4 * n_row)
            c = np.frombuffer(f.read(4 * n_out), "<u4")
            o = np.frombuffer(f.read(4 * n_out), "<u4")
            yield tag, pos, ordinal, c, o


def logits(d):
    p = os.path.join(d, "logits.bin")
    return np.fromfile(p, "<u4") if os.path.exists(p) else np.zeros(0, "<u4")


ref = logits(sys.argv[1])
bad = 0
for d in sys.argv[2:]:
    n = ok = zero = recs = 0
    ords, poss = set(), set()
    p = os.path.join(d, "attn.bin")
    for tag, pos, ordinal, c, o in (records(p) if os.path.exists(p) else []):
        recs += 1
        ords.add(ordinal)
        poss.add(pos)
        zero += int(not o.any())
        for h in range(len(c) // HD):
            n += 1
            ok += int(np.array_equal(c[HD * h:HD * h + HD], o[HD * h:HD * h + HD]))
    lg = logits(d)
    steps = len(lg) // V if len(lg) % V == 0 else -1
    same = 0
    if steps > 0 and len(ref) == len(lg):
        same = sum(int(np.array_equal(lg[i * V:(i + 1) * V],
                                      ref[i * V:(i + 1) * V]))
                   for i in range(steps))
    print(f"ATTN SHADOW {os.path.basename(d.rstrip('/'))} tag3_heads={ok}/{n} "
          f"records={recs} layers={len(ords)} positions={len(poss)} "
          f"zero_records={zero} logits_equal_steps={same}/{steps}")
    bad += (ok != n) or zero or same != steps
sys.exit(1 if bad else 0)
