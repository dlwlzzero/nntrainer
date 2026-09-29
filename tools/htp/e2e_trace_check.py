#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
@file   e2e_trace_check.py
@brief  dev/e2e-shadow-132 only (never merged): the first (step, layer,
        side) where the two-session E2E token leaves the CPU path.

    e2e_trace_check.py <S0 moe_rows.bin> <Ev e2e_trace.bin>

S0 (switch off, NNTR_MOE_ROW_TRACE): per decode MoE layer call, u32 pos,
n, then f32 input[n], output[n]. Ev (NNTR_HTP_E2E=1, NNTR_E2E_TRACE): per
token, u32 step, pos, count, then count f32 = per MoE layer the row S2
posts (the MoE input: every S2 op before it) and the row S1 returns (the
MoE output). Both runs are forced on the same ids. Prints, per step, the
layers whose input / output differ bit for bit, and the first one:
an input first names S2's ops since the previous MoE (norms, conv or
attention, FC, ADD), an output with an equal input names S1 (router,
MoE).
"""
import struct
import sys

import numpy as np


def cpu_rows(path):
    by_pos = {}
    with open(path, "rb") as f:
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            pos, n = struct.unpack("<2I", h)
            x = np.frombuffer(f.read(4 * n), "<u4")
            y = np.frombuffer(f.read(4 * n), "<u4")
            by_pos.setdefault(pos, []).append((x, y))
    return by_pos


def e2e_rows(path):
    out = []
    with open(path, "rb") as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            step, pos, cnt = struct.unpack("<3I", h)
            out.append((step, pos, np.frombuffer(f.read(4 * cnt), "<u4")))
    return out


cpu = cpu_rows(sys.argv[1])
first = None
steps = 0
for step, pos, flat in e2e_rows(sys.argv[2]):
    layers = cpu.get(pos)
    if layers is None:
        print(f"step {step} pos {pos}: no CPU rows")
        continue
    n = len(layers[0][0])
    rows = flat.reshape(-1, 2, n)
    steps += 1
    bad = []
    for l, ((cx, cy), (ex, ey)) in enumerate(zip(layers, rows)):
        if not np.array_equal(cx, ex):
            bad.append(f"L{l}.in({int((cx != ex).sum())})")
        if not np.array_equal(cy, ey):
            bad.append(f"L{l}.out({int((cy != ey).sum())})")
    if len(layers) != len(rows):
        bad.append(f"layers cpu={len(layers)} e2e={len(rows)}")
    if bad:
        print(f"step {step} pos {pos}: " + " ".join(bad[:12]))
        if first is None:
            first = (step, pos, bad[0])
print(f"E2E TRACE steps={steps} first_diff=" +
      (f"step{first[0]}/pos{first[1]}/{first[2]}" if first else "-"))
sys.exit(1 if first else 0)
