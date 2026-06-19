# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file weight_converter_cpu_fp16.py
## @brief CPU-target plain FP16 weight converter for Qwen3-0.6B.
## @author j2z0.lee <j2z0.lee@ax.samsung.com>
##
## Produces an nntrainer weight binary with every FC/Linear weight stored as
## plain row-major FP16 [N, K] (HuggingFace convention: N=out_features,
## K=in_features).  The nntrainer CPU runtime's FloatTensor::dotFloat path
## (Tdatatype::FP16 case) reads the weight directly as [N, K] FP16 without
## any tile-permute.
##
## This file is the CPU-side counterpart of weight_converter_hmx.py:
##   * weight_converter_hmx.py:     FC weights → HMX 32×32 tile-permuted FP16
##                                   (consumed by dotFloat32Float16 HTP path)
##   * weight_converter_cpu_fp16.py: FC weights → plain [N, K] FP16
##                                   (consumed by dotFloat32Float16 CPU path)
##
## Key difference from HMX converter:
##   * `permute_weight_to_fp16_tiles` is NOT called.  The HF [N, K] tensor is
##     cast to float16 and written to file as-is.  No transpose is needed:
##     the CPU path expects weight in [N, K] row-major, which is exactly the
##     HF nn.Linear storage convention.
##
## Tensor write order is identical to weight_converter.py and
## weight_converter_hmx.py so the same nntrainer loader can consume this bin
## (Tdatatype per tensor differs, but the offset-indexed order is fixed):
##   embed_tokens (FP32), then for each layer:
##     input_layernorm (FP32)
##     q_proj [q_norm], k_proj [k_norm], v_proj, o_proj  (FC → FP16)
##     post_attention_layernorm (FP32)
##     up_proj, gate_proj, down_proj  (FC → FP16)
##   model.norm (FP32)
##   [lm_head: only when tie_word_embeddings=False]
##
## Config (nntr_config.json) for CPU plain FP16:
##   "fc_layer_dtype": "FP16",
##   "model_tensor_type": "FP16-FP32"

import argparse
import io

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Module-level flag to suppress per-tensor logging during synthetic self-test.
_QUIET = False


def save_qwen3_cpu_fp16(params, n_layers, file, tie_word_embeddings=True):
    """Write Qwen3 weights to *file* in CPU plain FP16 format.

    FC/Linear projections are stored as plain row-major FP16 [N, K].
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
        """Write tensor as plain FP32 row-major."""
        if hasattr(weight, "detach"):
            weight = weight.detach().cpu().numpy()
        arr = np.ascontiguousarray(np.asarray(weight, dtype=np.float32))
        file.write(arr.tobytes())
        if not _QUIET:
            print(f"  [fp32] shape={weight.shape} -> {arr.nbytes} bytes")

    def save_linear_fp16_plain(weight, key_hint=""):
        """Write a Linear weight as plain row-major FP16 [N, K].

        HuggingFace stores nn.Linear weight as [out=N, in=K].  The CPU
        dotFloat path expects exactly this [N, K] layout — no transpose,
        no tile-permute.
        """
        if hasattr(weight, "detach"):
            w = weight.detach().cpu().numpy()
        else:
            w = np.asarray(weight)

        if w.ndim != 2:
            raise ValueError(
                f"Linear weight must be 2D, got shape {w.shape} ({key_hint})"
            )

        w_fp16 = np.ascontiguousarray(w.astype(np.float16))
        file.write(w_fp16.tobytes())
        if not _QUIET:
            N, K = w_fp16.shape
            print(
                f"  [fp16-plain] {key_hint or '?'}: N={N}, K={K} "
                f"-> {w_fp16.nbytes} bytes"
            )

    def save_projection(layer_name, proj_name):
        """Save one Linear projection (base weight, or base+LoRA A/B)."""
        full_proj = f"{layer_name}{proj_name}"
        lora_key = f"{full_proj}.lora_A.default.weight"
        if lora_key in params:
            save_linear_fp16_plain(
                params[f"{full_proj}.base_layer.weight"],
                f"{full_proj}.base_layer.weight",
            )
            save_linear_fp16_plain(
                params[f"{full_proj}.lora_A.default.weight"],
                f"{full_proj}.lora_A.default.weight",
            )
            save_linear_fp16_plain(
                params[f"{full_proj}.lora_B.default.weight"],
                f"{full_proj}.lora_B.default.weight",
            )
        else:
            save_linear_fp16_plain(
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
        save_linear_fp16_plain(params["lm_head.weight"], "lm_head.weight")


# ---------------------------------------------------------------------------
# Synthetic self-test (runs at import; no HF download required)
# ---------------------------------------------------------------------------

def _run_synthetic_test():
    """Verify byte layout against the HMX converter for a minimal 1-layer model.

    Checks:
      1. Each FC tensor emits exactly N*K*2 bytes (plain FP16).
      2. FP32 tensors emit exactly 4*numel bytes.
      3. A spot-sampled FC weight is bitwise-identical to w.astype(np.float16).
    """
    global _QUIET
    _QUIET = True

    rng = np.random.default_rng(7)

    VOCAB  = 256
    HIDDEN = 64   # small, no 32-alignment requirement for plain FP16
    HEAD_DIM = 16
    HEADS = 4
    KV_HEADS = 2
    INTER = 128
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
    save_qwen3_cpu_fp16(params, N_LAYERS, buf, tie_word_embeddings=True)
    total = buf.getbuffer().nbytes

    # Compute expected byte total.
    expected = 4 * VOCAB * HIDDEN  # embed_tokens FP32
    for _ in range(N_LAYERS):
        expected += 4 * HIDDEN                    # input_layernorm FP32
        expected += Q_OUT  * HIDDEN * 2           # q_proj FP16
        expected += 4 * HEAD_DIM                  # q_norm FP32
        expected += KV_OUT * HIDDEN * 2           # k_proj FP16
        expected += 4 * HEAD_DIM                  # k_norm FP32
        expected += KV_OUT * HIDDEN * 2           # v_proj FP16 (no v_norm)
        expected += HIDDEN * Q_OUT  * 2           # o_proj FP16
        expected += 4 * HIDDEN                    # post_attention_layernorm FP32
        expected += INTER  * HIDDEN * 2           # up_proj FP16
        expected += INTER  * HIDDEN * 2           # gate_proj FP16
        expected += HIDDEN * INTER  * 2           # down_proj FP16
    expected += 4 * HIDDEN                        # model.norm FP32

    assert total == expected, (
        f"Synthetic test FAILED: wrote {total} bytes, expected {expected}"
    )

    # Spot-check: the first FC tensor (q_proj for layer 0) must be stored as
    # plain FP16 [N, K] without any permutation.
    buf.seek(0)
    buf.read(4 * VOCAB * HIDDEN)     # skip embed_tokens
    buf.read(4 * HIDDEN)             # skip input_layernorm
    q_proj_bytes = buf.read(Q_OUT * HIDDEN * 2)
    q_fp16_ref = params["model.layers.0.self_attn.q_proj.weight"].astype(np.float16)
    q_fp16_out = np.frombuffer(q_proj_bytes, dtype=np.float16).reshape(Q_OUT, HIDDEN)
    if not np.array_equal(q_fp16_out, q_fp16_ref):
        raise RuntimeError(
            "Synthetic test FAILED: q_proj spot-check mismatch — "
            "plain FP16 [N,K] bit layout does not match expected."
        )

    _QUIET = False
    print(
        f"[synthetic-test] PASS: 1-layer plain FP16, "
        f"{total} bytes == {expected} expected; q_proj spot-check OK"
    )


_run_synthetic_test()


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert Qwen3-0.6B HuggingFace weights to nntrainer plain FP16 "
            "binary (CPU backend, row-major [N, K])."
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
        default="./nntr_qwen3_0.6b_fp16.bin",
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
        save_qwen3_cpu_fp16(
            model.state_dict(),
            config.num_hidden_layers,
            f_model,
            tie_word_embeddings=tie_word_embeddings,
        )

    print(f"Saved: {args.output_name}")
