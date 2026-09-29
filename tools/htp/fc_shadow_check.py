#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
@file   fc_shadow_check.py
@brief  dev/fc-shadow only (#132 PR 2, never merged): G2 of plan 132 on
        NNTR_FC_SHADOW dumps.

    fc_shadow_check.py <shadow.bin>...

Records (htp_compute_ops.cpp, dev/fc-shadow): u32 tag, step, n_in, n_out,
then f32 in[n_in], cpu[n_out], dsp[n_out]. Tag 3: one FC call or lm_head
slice (the DSP's fc_q4m1_f32 on the same input and weight), tag 4: one
residual ADD (the DSP's add_f32 of the same two rows), tag 5: one router
(logits | expert ids sorted | their weights; the DSP's router_topk_det_f32);
dev/e2e-shadow-132: tag 6 one dense-FFN SwiGLU (in = gate | up; the DSP's
swiglu_cpu_f32, DENSE_FFN's), tag 7 one greedy pick (in = the logits; the
CPU's first maximum against the DSP's argmax_f32, LM_HEAD's), tag 8 one
decode conv + gate (in = a | b | c; the CPU's NEON decode from its own copy
of the state against the resident CONV1D_GATE's output). Every record
must be bit-equal; fc, add, router and argmax must each have one. Prints per tag equal / total, the step
range, per-step counts, and the first differing records.
"""
import collections
import struct
import sys

import numpy as np

NAMES = {3: "fc", 4: "add", 5: "router", 6: "swiglu", 7: "argmax", 8: "conv"}
TAGS = (3, 4, 5, 6, 7, 8)
REQUIRED = (3, 4, 5, 7)


def records(path):
    with open(path, "rb") as f:
        while True:
            h = f.read(16)
            if len(h) < 16:
                return
            tag, step, n_in, n_out = struct.unpack("<4I", h)
            x = np.frombuffer(f.read(4 * n_in), "<u4")
            c = np.frombuffer(f.read(4 * n_out), "<u4")
            o = np.frombuffer(f.read(4 * n_out), "<u4")
            yield tag, step, n_in, n_out, x, c, o


bad = 0
for path in sys.argv[1:]:
    n = collections.Counter()
    ok = collections.Counter()
    per_step = collections.Counter()
    shapes = collections.Counter()
    first_bad = []
    steps = set()
    first_step = {}
    for tag, step, n_in, n_out, x, c, o in records(path):
        n[tag] += 1
        eq = np.array_equal(c, o)
        ok[tag] += int(eq)
        per_step[(tag, step)] += 1
        steps.add(step)
        if tag == 3:
            shapes[(n_in, n_out)] += 1
        if not eq and first_step.get(tag) is None:
            first_step[tag] = step
        if not eq and len(first_bad) < 8:
            i = int(np.nonzero(c != o)[0][0])
            first_bad.append(f"  tag={tag} step={step} n_in={n_in} n_out={n_out} "
                             f"elems_bad={int((c != o).sum())} first i={i} "
                             f"cpu={c[i]:08x} dsp={o[i]:08x}")
    counts = " ".join(f"{NAMES.get(t, t)}={ok[t]}/{n[t]}" for t in TAGS)
    print(f"FC SHADOW {path} {counts} steps={min(steps) if steps else '-'}.."
          f"{max(steps) if steps else '-'}")
    for s in sorted(steps):
        print(f"  step {s}: " + " ".join(
            f"{NAMES[t]}={per_step[(t, s)]}" for t in TAGS))
    print("  fc shapes (K, N): " + " ".join(
        f"{k}x{v}" for k, v in sorted(shapes.items())))
    print("  first differing step per tag: " + (" ".join(
        f"{NAMES[t]}@{first_step[t]}" for t in sorted(first_step)) or "-"))
    for line in first_bad:
        print(line)
    bad += any(ok[t] != n[t] for t in n) or any(n[t] == 0 for t in REQUIRED)
sys.exit(1 if bad else 0)
