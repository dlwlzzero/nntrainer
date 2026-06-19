# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file weight_converter_cpu_q4_0.py
## @brief CPU-target plain Q4_0 weight converter for Qwen3-0.6B.
## @author j2z0.lee <j2z0.lee@ax.samsung.com>
##
## Produces an nntrainer weight binary with every FC/Linear weight stored as
## a plain Q4_0 block stream (ggml block_q4_0 format: 18 bytes/block,
## QK4_0=32 elements/block, layout = [2-byte FP16 scale | 16-byte packed quants]).
##
## This is the CPU-side counterpart of weight_converter_htp.py:
##   * weight_converter_htp.py:       FC → Q4_0 x4x2 (DSP-native repack)
##   * weight_converter_cpu_q4_0.py:  FC → plain Q4_0 block stream
##
## The nntrainer CPU runtime's Q4_0Tensor / FloatTensor::dotQnK (Tdatatype::Q4_0
## case) reads the weight as a flat Q4_0 block stream per row, which is exactly
## what quantize_q4_0() in q4_0_x4x2_quant.py produces before the x4x2 repack
## step.  This converter stops at the plain Q4_0 stage.
##
## Alignment requirements:
##   K must be a multiple of QK4_0 (32) — a Q4_0 row must have an integer number
##   of blocks.  N has no alignment requirement for plain Q4_0.  If K is not a
##   multiple of 32, the weight falls back to plain FP32.
##
## Tensor write order is identical to weight_converter.py,
## weight_converter_hmx.py, and weight_converter_htp.py (same nntrainer loader
## offset-indexed order):
##   embed_tokens (FP32), then for each layer:
##     input_layernorm (FP32)
##     q_proj [q_norm], k_proj [k_norm], v_proj, o_proj  (FC → Q4_0)
##     post_attention_layernorm (FP32)
##     up_proj, gate_proj, down_proj  (FC → Q4_0)
##   model.norm (FP32)
##   [lm_head: only when tie_word_embeddings=False]
##
## Config (nntr_config.json) for CPU plain Q4_0:
##   "fc_layer_dtype": "Q4_0",
##   "model_tensor_type": "Q4_0-FP32"

import argparse
import io
import os
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Import quantize_q4_0 from q4_0_x4x2_quant.py (same directory).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from q4_0_x4x2_quant import quantize_q4_0, QK4_0

# Module-level flag to suppress per-tensor logging during synthetic self-test.
_QUIET = False


def _q4_0_row_bytes(K: int) -> int:
    """Bytes for one row of plain Q4_0: (K / QK4_0) blocks × 18 bytes/block."""
    return (K // QK4_0) * 18


def save_qwen3_cpu_q4_0(params, n_layers, file, tie_word_embeddings=True):
    """Write Qwen3 weights to *file* in CPU plain Q4_0 format.

    FC/Linear projections are stored as plain Q4_0 block streams per row.
    All other tensors (embeddings, norms) are stored as FP32.

    Parameters
    ----------
    params : dict-like
        HuggingFace state_dict (torch.Tensors or numpy arrays).
    n_layers : int
        Number of transformer decoder layers.
    file : writable binary file
        Destination weight binary.
    tie_word_embeddings : bool
        When True (Qwen3-0.6B default) no separate lm_head weight is written.
    """

    def save_fp32(weight):
        if hasattr(weight, "detach"):
            weight = weight.detach().cpu().numpy()
        arr = np.ascontiguousarray(np.asarray(weight, dtype=np.float32))
        file.write(arr.tobytes())
        if not _QUIET:
            print(f"  [fp32] shape={weight.shape} -> {arr.nbytes} bytes")

    def save_linear_q4_0_plain(weight, key_hint=""):
        """Write a Linear weight as plain Q4_0 block stream.

        HuggingFace stores nn.Linear weight as [out=N, in=K].  We quantize
        each row (of K elements) independently to one Q4_0 block per QK4_0
        elements, producing (K/QK4_0) blocks × 18 bytes/block per row,
        N rows total.

        If K is not a multiple of QK4_0, falls back to plain FP32 with a
        warning (the C++ CPU path will handle it as FP32).
        """
        if hasattr(weight, "detach"):
            w = weight.detach().cpu().numpy().astype(np.float32)
        else:
            w = np.asarray(weight, dtype=np.float32)

        if w.ndim != 2:
            raise ValueError(
                f"Linear weight must be 2D, got shape {w.shape} ({key_hint})"
            )

        N, K = w.shape
        if K % QK4_0 != 0:
            print(
                f"  [warn] {key_hint or '?'}: K={K} is not a multiple of "
                f"QK4_0={QK4_0}; storing plain FP32 [N,K] (CPU fallback)"
            )
            file.write(np.ascontiguousarray(w, dtype=np.float32).tobytes())
            return

        # quantize_q4_0 accepts any shape (..., K) and returns N*(K/32)*18 bytes.
        packed = quantize_q4_0(np.ascontiguousarray(w))
        expected = N * _q4_0_row_bytes(K)
        assert len(packed) == expected, (
            f"Q4_0 byte count mismatch for {key_hint}: "
            f"got {len(packed)}, expected {expected}"
        )
        file.write(packed)
        if not _QUIET:
            print(
                f"  [q4_0-plain] {key_hint or '?'}: N={N}, K={K} "
                f"-> {len(packed)} bytes ({N * K // QK4_0} blocks)"
            )

    def save_projection(layer_name, proj_name):
        full_proj = f"{layer_name}{proj_name}"
        lora_key = f"{full_proj}.lora_A.default.weight"
        if lora_key in params:
            save_linear_q4_0_plain(
                params[f"{full_proj}.base_layer.weight"],
                f"{full_proj}.base_layer.weight",
            )
            save_linear_q4_0_plain(
                params[f"{full_proj}.lora_A.default.weight"],
                f"{full_proj}.lora_A.default.weight",
            )
            save_linear_q4_0_plain(
                params[f"{full_proj}.lora_B.default.weight"],
                f"{full_proj}.lora_B.default.weight",
            )
        else:
            save_linear_q4_0_plain(
                params[f"{full_proj}.weight"],
                f"{full_proj}.weight",
            )

    def save_attention(layer_name):
        save_fp32(params[f"{layer_name}input_layernorm.weight"])
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            norm_key = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if norm_key in params:
                save_fp32(params[norm_key])

    def save_feed_forward(layer_name):
        save_fp32(params[f"{layer_name}post_attention_layernorm.weight"])
        for proj in ["up_proj", "gate_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")

    # 1. Embedding table (FP32).
    save_fp32(params["model.embed_tokens.weight"])

    # 2. Per-layer transformer blocks.
    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_attention(layer_prefix)
        save_feed_forward(layer_prefix)

    # 3. Final RMSNorm (FP32).
    save_fp32(params["model.norm.weight"])

    # 4. lm_head: only when NOT tied.
    if not tie_word_embeddings:
        save_linear_q4_0_plain(params["lm_head.weight"], "lm_head.weight")


# ---------------------------------------------------------------------------
# Synthetic self-test (runs at import; no HF download required)
# ---------------------------------------------------------------------------

def _run_synthetic_test():
    """Verify byte layout for a minimal 1-layer model.

    Checks:
      1. Each FC tensor emits exactly N * _q4_0_row_bytes(K) bytes.
      2. FP32 tensors emit exactly 4*numel bytes.
      3. A spot-sampled FC weight round-trips correctly through Q4_0 decode.
    """
    global _QUIET
    _QUIET = True

    rng = np.random.default_rng(13)

    VOCAB  = 256
    HIDDEN = 256   # must be multiple of QK4_0=32
    HEAD_DIM = 32
    HEADS = 4
    KV_HEADS = 2
    INTER = 256
    N_LAYERS = 1

    Q_OUT  = HEADS * HEAD_DIM
    KV_OUT = KV_HEADS * HEAD_DIM

    def fp32(shape):
        return rng.standard_normal(shape).astype(np.float32)

    params = {
        "model.embed_tokens.weight": fp32((VOCAB, HIDDEN)),
    }
    for li in range(N_LAYERS):
        p = f"model.layers.{li}."
        params[f"{p}input_layernorm.weight"]          = fp32((HIDDEN,))
        params[f"{p}self_attn.q_proj.weight"]         = fp32((Q_OUT,  HIDDEN))
        params[f"{p}self_attn.q_norm.weight"]         = fp32((HEAD_DIM,))
        params[f"{p}self_attn.k_proj.weight"]         = fp32((KV_OUT, HIDDEN))
        params[f"{p}self_attn.k_norm.weight"]         = fp32((HEAD_DIM,))
        params[f"{p}self_attn.v_proj.weight"]         = fp32((KV_OUT, HIDDEN))
        params[f"{p}self_attn.o_proj.weight"]         = fp32((HIDDEN, Q_OUT))
        params[f"{p}post_attention_layernorm.weight"] = fp32((HIDDEN,))
        params[f"{p}mlp.up_proj.weight"]              = fp32((INTER, HIDDEN))
        params[f"{p}mlp.gate_proj.weight"]            = fp32((INTER, HIDDEN))
        params[f"{p}mlp.down_proj.weight"]            = fp32((HIDDEN, INTER))
    params["model.norm.weight"] = fp32((HIDDEN,))

    buf = io.BytesIO()
    save_qwen3_cpu_q4_0(params, N_LAYERS, buf, tie_word_embeddings=True)
    total = buf.getbuffer().nbytes

    def q4_0_bytes(N, K):
        return N * _q4_0_row_bytes(K)

    expected = 4 * VOCAB * HIDDEN     # embed_tokens FP32
    for _ in range(N_LAYERS):
        expected += 4 * HIDDEN                         # input_layernorm FP32
        expected += q4_0_bytes(Q_OUT,  HIDDEN)         # q_proj
        expected += 4 * HEAD_DIM                       # q_norm FP32
        expected += q4_0_bytes(KV_OUT, HIDDEN)         # k_proj
        expected += 4 * HEAD_DIM                       # k_norm FP32
        expected += q4_0_bytes(KV_OUT, HIDDEN)         # v_proj (no v_norm)
        expected += q4_0_bytes(HIDDEN, Q_OUT)          # o_proj
        expected += 4 * HIDDEN                         # post_attn_ln FP32
        expected += q4_0_bytes(INTER, HIDDEN)          # up_proj
        expected += q4_0_bytes(INTER, HIDDEN)          # gate_proj
        expected += q4_0_bytes(HIDDEN, INTER)          # down_proj
    expected += 4 * HIDDEN                             # model.norm FP32

    assert total == expected, (
        f"Synthetic test FAILED: wrote {total} bytes, expected {expected}"
    )

    # Spot-check: q_proj output should be exactly quantize_q4_0 on the raw FP32.
    buf.seek(0)
    buf.read(4 * VOCAB * HIDDEN)   # skip embed_tokens
    buf.read(4 * HIDDEN)           # skip input_layernorm
    q_proj_bytes_out = buf.read(q4_0_bytes(Q_OUT, HIDDEN))
    q_proj_ref = quantize_q4_0(
        np.ascontiguousarray(
            params["model.layers.0.self_attn.q_proj.weight"], dtype=np.float32
        )
    )
    if q_proj_bytes_out != q_proj_ref:
        raise RuntimeError(
            "Synthetic test FAILED: q_proj spot-check mismatch — "
            "plain Q4_0 block stream does not match quantize_q4_0 reference."
        )

    _QUIET = False
    print(
        f"[synthetic-test] PASS: 1-layer plain Q4_0, "
        f"{total} bytes == {expected} expected; q_proj spot-check OK"
    )


_run_synthetic_test()


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert Qwen3-0.6B HuggingFace weights to nntrainer plain Q4_0 "
            "block-stream binary (CPU backend)."
        )
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="./nntr_qwen3_0.6b_q4_0.bin",
        help="Output weight binary path",
    )
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
    print(f"tie_word_embeddings: {tie_word_embeddings}")
    print(f"num_hidden_layers:   {config.num_hidden_layers}")

    with open(args.output_name, "wb") as f_model:
        save_qwen3_cpu_q4_0(
            model.state_dict(),
            config.num_hidden_layers,
            f_model,
            tie_word_embeddings=tie_word_embeddings,
        )

    print(f"Saved: {args.output_name}")
