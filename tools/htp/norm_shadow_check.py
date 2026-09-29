#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
@file   norm_shadow_check.py
@brief  dev/norm-shadow only (#164, never merged): G2 of plan 164 on
        NNTR_NORM_SHADOW / NNTR_LOGIT_SHADOW dumps.

    norm_shadow_check.py <ref d_f_A dir> <run dir>...

Per run dir: tag 0 (RMSNORM on the HTP) records whose HTP output equals
the CPU's bit for bit, tag 1 (q|k norm) heads whose DSP output (the
rmsnorm_det_f32 entry) equals the CPU's, tag 2 (RMSNORM on the CPU)
records whose layer output equals the fresh recompute, and the decode
steps whose logits.bin equals the reference's bit for bit.
"""
import os
import struct
import sys

import numpy as np


def records(path):
    with open(path, "rb") as f:
        while True:
            h = f.read(16)
            if len(h) < 16:
                return
            tag, pos, n_in, n_out = struct.unpack("<4I", h)
            x = np.frombuffer(f.read(4 * n_in), "<u4")
            c = np.frombuffer(f.read(4 * n_out), "<u4")
            o = np.frombuffer(f.read(4 * n_out), "<u4")
            yield tag, pos, x, c, o


def logits(d):
    p = os.path.join(d, "logits.bin")
    return np.fromfile(p, "<u4") if os.path.exists(p) else np.zeros(0, "<u4")


V = 128000  # LFM2.5 vocab: one decode step of logits.bin
ref = logits(sys.argv[1])
bad = 0
for d in sys.argv[2:]:
    n = [0, 0, 0]
    ok = [0, 0, 0]
    zero1 = 0
    for tag, pos, x, c, o in records(os.path.join(d, "norm.bin")):
        if tag == 1:
            zero1 += int(not o.any())
            for h in range(len(c) // 64):
                n[1] += 1
                ok[1] += int(np.array_equal(c[64 * h:64 * h + 64], o[64 * h:64 * h + 64]))
        else:
            n[tag] += 1
            ok[tag] += int(np.array_equal(c, o))
    lg = logits(d)
    steps = len(lg) // V if len(lg) % V == 0 else -1
    same = 0
    if steps > 0 and len(ref) == len(lg):
        same = sum(int(np.array_equal(lg[i * V:(i + 1) * V],
                                      ref[i * V:(i + 1) * V]))
                   for i in range(steps))
    print(f"NORM SHADOW {os.path.basename(d.rstrip('/'))} tag0={ok[0]}/{n[0]} "
          f"tag1_heads={ok[1]}/{n[1]} tag1_zero_records={zero1} "
          f"tag2={ok[2]}/{n[2]} logits_equal_steps={same}/{steps}")
    bad += (ok != n) or zero1 or same != steps
sys.exit(1 if bad else 0)
