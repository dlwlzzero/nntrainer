#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
@file   norm_gammas.py
@brief  dev/norm-shadow only (#164, never merged): the true RMSNorm and q/k
        norm gammas of hf/model.safetensors (bf16 -> f32) in decode-call
        order, for m1_ops_host_check --replay and RmsNormCpuOrder.Replay.

    norm_gammas.py <model.safetensors> <out_dir>

gamma_rms.f32: layers.{i}.operator_norm, layers.{i}.ffn_norm for every
layer, then embedding_norm. gamma_qk.f32: q_layernorm | k_layernorm of each
attention layer in layer order.
"""
import json
import struct
import sys

import numpy as np

path, out = sys.argv[1], sys.argv[2]
with open(path, "rb") as fh:
    hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
    base = fh.tell()

    def get(name):
        t = hdr[name]
        assert t["dtype"] == "BF16", name
        a, b = t["data_offsets"]
        fh.seek(base + a)
        u = np.frombuffer(fh.read(b - a), dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32)

    n = 1 + max(int(k.split(".")[2]) for k in hdr if k.startswith("model.layers."))
    rms = [get(f"model.layers.{i}.{w}.weight")
           for i in range(n) for w in ("operator_norm", "ffn_norm")]
    rms.append(get("model.embedding_norm.weight"))
    qk = [get(f"model.layers.{i}.self_attn.{w}.weight") for i in range(n)
          if f"model.layers.{i}.self_attn.q_layernorm.weight" in hdr
          for w in ("q_layernorm", "k_layernorm")]
np.concatenate(rms).tofile(f"{out}/gamma_rms.f32")
np.concatenate(qk).tofile(f"{out}/gamma_qk.f32")
print(f"gamma_rms.f32 {len(rms)} x {rms[0].size}, gamma_qk.f32 {len(qk)} x {qk[0].size}")
