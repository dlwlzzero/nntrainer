# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file weight_converter_htp.py
## @brief HTP-dedicated weight conversion script for Qwen3-0.6B (Q4_0 x4x2)
## @author j2z0.lee <j2z0.lee@ax.samsung.com>
##
## This converter produces an nntrainer weight binary pre-formatted for the
## HTP DSP matmul kernel `htp_ops_mat_mul_af32_pwqk0_of32`, which expects
## every FC/Linear weight as a Q4_0 x4x2 row-strided buffer (see
## q4_0_x4x2_quant.py for the layout spec).  The runtime consumer is
## nntrainer/tensor/float_tensor.cpp FloatTensor::dotQnK, Tdatatype::Q4_0_X4X2
## case — that path memcpy's the weight directly into RPC shared memory with no
## further conversion, mirroring the FP16 HMX path (dotFloat32Float16).
##
## What is quantized vs kept FP32:
##   * FC/Linear projections (q/k/v/o_proj, mlp up/gate/down_proj) → Q4_0 x4x2
##     via quantize_and_repack_q4_0_x4x2.  HF stores weight as [N=out, K=in];
##     the x4x2 quantizer expects exactly [N, K], so we feed the HF weight
##     as-is (no transpose — the HMX converter transposed to [K, N] for its
##     tile-permute, but x4x2 convention is opposite).
##   * Embeddings, RMSNorm/LayerNorm weights (input_layernorm,
##     post_attention_layernorm, q_norm, k_norm, model.norm) → plain FP32.
##
## Non-aligned fallback:
##   If a Linear weight has K%256 != 0 or N%32 != 0 (e.g. LoRA adapter dims),
##   quantize_and_repack_q4_0_x4x2 raises ValueError.  We catch it, emit a
##   warning, and store the weight as plain FP32 [N, K] row-major.  The C++
##   side falls back to the CPU dispatch path for such tensors.
##
## Tensor order:
##   Identical to weight_converter.py::save_qwen3_for_nntrainer so that the
##   same nntrainer loader can consume this binary (Tdatatype differs per
##   tensor, but the order is fixed).  Qwen3-0.6B is tied
##   (tie_word_embeddings=True), so no separate lm_head weight is written.
##
## Differences from weight_converter_hmx.py:
##   * save_linear_fp16 replaced by save_linear_x4x2 (Q4_0 x4x2 vs FP16 tiles)
##   * No transpose before quantize (x4x2 takes [N,K]; HMX took [K,N])
##   * Fallback stores FP32 (not FP16) for non-aligned tensors
##   * lm_head is NOT written when tie_word_embeddings=True (0.6B default)

import argparse
import sys
import os

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Import the Task-1 quantizer from the same directory as this script.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from q4_0_x4x2_quant import quantize_and_repack_q4_0_x4x2, x4x2_row_stride


def save_qwen3_for_nntrainer_htp(params, n_layers, file, tie_word_embeddings=True):
    """Convert and save weights in Q4_0 x4x2 format for the HTP DSP backend.

    Parameters
    ----------
    params : dict[str, torch.Tensor | np.ndarray]
        HuggingFace state_dict (or any mapping with the same key names).
    n_layers : int
        Number of transformer decoder layers (config.num_hidden_layers).
    file : writable binary file object
        Destination weight binary.
    tie_word_embeddings : bool
        When True (Qwen3-0.6B default) lm_head shares embed_tokens; no
        separate lm_head weight is written.
    """

    def save_fp32(weight):
        """Write a tensor as plain FP32 row-major."""
        if hasattr(weight, "detach"):
            weight = weight.detach().cpu().numpy()
        arr = np.ascontiguousarray(np.asarray(weight, dtype=np.float32))
        file.write(arr.tobytes())

    def save_linear_x4x2(weight, key_hint=""):
        """Write a Linear weight as Q4_0 x4x2 bytes, with FP32 fallback.

        Parameters
        ----------
        weight : array-like, shape [N, K]  (HF nn.Linear convention: out, in)
        key_hint : str
            State-dict key name, used only for warning messages.
        """
        if hasattr(weight, "detach"):
            w = weight.detach().cpu().numpy().astype(np.float32)
        else:
            w = np.asarray(weight, dtype=np.float32)

        if w.ndim != 2:
            raise ValueError(
                f"Linear weight must be 2D, got shape {w.shape} ({key_hint})"
            )

        N, K = w.shape  # HF: [out=N, in=K] — fed as-is to x4x2 quantizer.
        try:
            packed = quantize_and_repack_q4_0_x4x2(np.ascontiguousarray(w))
            file.write(packed)
            expected = N * x4x2_row_stride(K)
            assert len(packed) == expected, (
                f"x4x2 byte count mismatch for {key_hint}: "
                f"got {len(packed)}, expected {expected}"
            )
            print(
                f"  [x4x2] {key_hint or '?'}: N={N}, K={K} -> {len(packed)} bytes"
            )
        except ValueError as exc:
            print(
                f"  [warn] {key_hint or '?'}: shape [{N},{K}] not x4x2-aligned "
                f"({exc}); storing plain FP32 [N,K] (CPU fallback at runtime)"
            )
            file.write(np.ascontiguousarray(w, dtype=np.float32).tobytes())

    def save_projection(layer_name, proj_name):
        """Save one Linear projection (base weight, or base+LoRA A/B)."""
        full_proj = f"{layer_name}{proj_name}"
        lora_key = f"{full_proj}.lora_A.default.weight"
        if lora_key in params:
            save_linear_x4x2(
                params[f"{full_proj}.base_layer.weight"],
                f"{full_proj}.base_layer.weight",
            )
            save_linear_x4x2(
                params[f"{full_proj}.lora_A.default.weight"],
                f"{full_proj}.lora_A.default.weight",
            )
            save_linear_x4x2(
                params[f"{full_proj}.lora_B.default.weight"],
                f"{full_proj}.lora_B.default.weight",
            )
        else:
            save_linear_x4x2(
                params[f"{full_proj}.weight"],
                f"{full_proj}.weight",
            )

    def save_attention(layer_name):
        """Save attention block weights for one transformer layer.

        Order matches weight_converter.py::save_attention:
          input_layernorm (FP32)
          q_proj [+ q_norm], k_proj [+ k_norm], v_proj [+ v_norm], o_proj [+ o_norm]
        """
        save_fp32(params[f"{layer_name}input_layernorm.weight"])

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            # Qwen3 per-head QK norms (FP32, consumed by RMSNorm kernel).
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                print(f"  [fp32] {proj_norm_name}")
                save_fp32(params[proj_norm_name])

    def save_feed_forward(layer_name):
        """Save MLP block weights for one transformer layer.

        Order matches weight_converter.py::save_feed_forward:
          post_attention_layernorm (FP32), up_proj, gate_proj, down_proj
        """
        save_fp32(params[f"{layer_name}post_attention_layernorm.weight"])

        for proj in ["up_proj", "gate_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")

    # --- Top-level serialization order (matches weight_converter.py) ----------
    # 1. Embedding table (FP32, CPU-side lookup).
    save_fp32(params["model.embed_tokens.weight"])

    # 2. Per-layer transformer blocks.
    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_attention(layer_prefix)
        save_feed_forward(layer_prefix)

    # 3. Final RMSNorm (FP32).
    save_fp32(params["model.norm.weight"])

    # 4. lm_head: only written when NOT tied.
    if not tie_word_embeddings:
        save_linear_x4x2(params["lm_head.weight"], "lm_head.weight")


# ---------------------------------------------------------------------------
# Synthetic logic test (runs at import; no HF download required)
# ---------------------------------------------------------------------------

def _run_synthetic_test():
    """Verify byte lengths and tensor count/order against weight_converter.py.

    Builds a minimal 1-layer fake state_dict with 32-aligned dims and checks:
      1. Each FC tensor emits exactly N * x4x2_row_stride(K) bytes.
      2. FP32 tensors emit exactly 4 * numel bytes.
      3. Total tensor count matches the reference order for a 1-layer model
         with tied embeddings and with q_norm/k_norm present.
    """
    import io

    rng = np.random.default_rng(42)

    # Qwen3-0.6B-like dims (shrunk for speed; must satisfy N%32==0, K%256==0).
    VOCAB  = 512   # embed_tokens rows (arbitrary, stays FP32)
    HIDDEN = 1024  # K for most projections (must be multiple of 256)
    HEADS  = 16
    KV_HEADS = 8
    HEAD_DIM = 64  # 1024 / 16
    INTER  = 2816  # intermediate_size for MLP (must be multiple of 256 here)
    N_LAYERS = 1

    # q_proj: [num_heads*head_dim, hidden] = [1024, 1024]
    # k_proj: [num_kv_heads*head_dim, hidden] = [512, 1024]
    # v_proj: [512, 1024]
    # o_proj: [hidden, num_heads*head_dim] = [1024, 1024]
    Q_OUT = HEADS * HEAD_DIM            # 1024
    KV_OUT = KV_HEADS * HEAD_DIM        # 512
    MLP_OUT = INTER                     # 2816

    def fp32(shape):
        return rng.standard_normal(shape).astype(np.float32)

    # Build synthetic state_dict (numpy arrays — accepted by save_* helpers).
    params = {}
    params["model.embed_tokens.weight"] = fp32((VOCAB, HIDDEN))

    for li in range(N_LAYERS):
        p = f"model.layers.{li}."
        params[f"{p}input_layernorm.weight"]        = fp32((HIDDEN,))
        params[f"{p}self_attn.q_proj.weight"]       = fp32((Q_OUT,  HIDDEN))
        params[f"{p}self_attn.q_norm.weight"]       = fp32((HEAD_DIM,))
        params[f"{p}self_attn.k_proj.weight"]       = fp32((KV_OUT, HIDDEN))
        params[f"{p}self_attn.k_norm.weight"]       = fp32((HEAD_DIM,))
        params[f"{p}self_attn.v_proj.weight"]       = fp32((KV_OUT, HIDDEN))
        # v_norm is not in Qwen3 (only q_norm and k_norm exist); omit.
        params[f"{p}self_attn.o_proj.weight"]       = fp32((HIDDEN, Q_OUT))
        params[f"{p}post_attention_layernorm.weight"] = fp32((HIDDEN,))
        params[f"{p}mlp.up_proj.weight"]            = fp32((MLP_OUT, HIDDEN))
        params[f"{p}mlp.gate_proj.weight"]          = fp32((MLP_OUT, HIDDEN))
        params[f"{p}mlp.down_proj.weight"]          = fp32((HIDDEN,  MLP_OUT))

    params["model.norm.weight"] = fp32((HIDDEN,))
    # tie_word_embeddings=True → no lm_head entry needed.

    buf = io.BytesIO()
    save_qwen3_for_nntrainer_htp(params, N_LAYERS, buf, tie_word_embeddings=True)
    buf.seek(0)

    total = buf.getbuffer().nbytes

    # ---- Reference byte-count calculation (mirrors weight_converter.py order) ----
    # embed_tokens: FP32
    expected_bytes = 4 * VOCAB * HIDDEN

    for li in range(N_LAYERS):
        # input_layernorm (FP32)
        expected_bytes += 4 * HIDDEN
        # q_proj x4x2
        expected_bytes += Q_OUT * x4x2_row_stride(HIDDEN)
        # q_norm FP32
        expected_bytes += 4 * HEAD_DIM
        # k_proj x4x2
        expected_bytes += KV_OUT * x4x2_row_stride(HIDDEN)
        # k_norm FP32
        expected_bytes += 4 * HEAD_DIM
        # v_proj x4x2 (no v_norm in Qwen3)
        expected_bytes += KV_OUT * x4x2_row_stride(HIDDEN)
        # o_proj x4x2
        expected_bytes += HIDDEN * x4x2_row_stride(Q_OUT)
        # post_attention_layernorm FP32
        expected_bytes += 4 * HIDDEN
        # up_proj, gate_proj, down_proj x4x2
        expected_bytes += MLP_OUT * x4x2_row_stride(HIDDEN)   # up
        expected_bytes += MLP_OUT * x4x2_row_stride(HIDDEN)   # gate
        expected_bytes += HIDDEN  * x4x2_row_stride(MLP_OUT)  # down

    # model.norm FP32
    expected_bytes += 4 * HIDDEN
    # no lm_head (tied)

    assert total == expected_bytes, (
        f"Synthetic test FAILED: wrote {total} bytes, expected {expected_bytes}"
    )
    print(
        f"[synthetic-test] PASS: 1-layer model, "
        f"{total} bytes == {expected_bytes} expected"
    )


_run_synthetic_test()


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert Qwen3-0.6B HuggingFace weights to nntrainer Q4_0 x4x2 "
            "binary (HTP DSP backend)."
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
        default="./nntr_qwen3_0.6b_q4_0_x4x2.bin",
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
        save_qwen3_for_nntrainer_htp(
            model.state_dict(),
            config.num_hidden_layers,
            f_model,
            tie_word_embeddings=tie_word_embeddings,
        )

    print(f"Saved: {args.output_name}")
