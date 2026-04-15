## @file weight_converter_hmx.py
## @brief HMX-friendly weight conversion script for qwen3 model (HTP target)
## @author Eunju Yang <ej.yang@samsung.com>
##
## This converter produces an nntrainer weight binary pre-formatted for the
## HTP HMX matmul kernel `hmx_mat_mul_af32_pwf16_of32`, which expects every
## Linear weight as an HMX 32x32 tile-permuted FP16 buffer
## (total bytes = N * K * sizeof(fp16); N = out_features, K = in_features;
## both axes must be multiples of 32). Runtime consumer:
## nntrainer/tensor/float_tensor.cpp FloatTensor::dotFloat32Float16 HTP path
## — that path memcpy's the weight directly into RPC shared memory with no
## further conversion.
##
## Tile layout (matches test/unittest/unittest_htp_kernels.cpp:38-51):
##   for i in [0, K), j in [0, N):
##     i0 = i // 32, i1 = i % 32
##     j0 = j // 32, j1 = j % 32
##     tile_idx = j0 * (K / 32) + i0
##     dst[tile_idx * 1024 + (i1 & ~1) * 32 + j1 * 2 + (i1 & 1)] = fp16(w[i, j])
##
## Differences from weight_converter.py:
##   * Linear weights are saved as FP16 (np.float16), not FP32.
##   * Linear weights are NOT row-major [K, N]; instead they are tile-permuted
##     into the layout above (no .permute(1, 0); HuggingFace [out, in] = [N, K]
##     is fed straight through the permute helper).
##   * Embeddings, RMSNorm weights (input/post-attention layernorm, Qwen3
##     q_norm/k_norm, final model.norm), and biases stay FP32 — they are
##     either consumed on CPU or by the HVX RMSNorm kernel which requires
##     FP32.
##   * lm_head.weight is treated as a Linear projection.
##   * If a Linear weight has any axis that is not a multiple of 32 (e.g.
##     LoRA adapter dims), the tile-permute is skipped and the weight is
##     stored as plain row-major [N, K] FP16. The C++ side falls back to
##     CPU dispatch in that case (the 32-align guard in float_tensor.cpp
##     dotFloat32Float16 handles this).

import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def permute_weight_to_fp16_tiles(w_f32):
    """Convert a [K, N] FP32 weight into the HMX 32x32 tile-permuted FP16
    layout consumed by `hmx_mat_mul_af32_pwf16_of32`.

    Bit-for-bit equivalent to the C reference at
    test/unittest/unittest_htp_kernels.cpp:38-51:

        for i in [0, K):
          for j in [0, N):
            i0, i1 = divmod(i, 32)
            j0, j1 = divmod(j, 32)
            tile_idx = j0 * (K / 32) + i0
            dst[tile_idx * 1024 + (i1 & ~1) * 32 + j1 * 2 + (i1 & 1)] = ...

    Implemented as a pure reshape + transpose (no Python loop):
      src = w_f16.reshape(K/32, 16, 2, N/32, 32)
                   axes:  i0   p   q   j0    j1
      dst = src.transpose(j0, i0, p, j1, q) -> (N/32, K/32, 16, 32, 2)

    The dst's flat C-order position of element src[i0, p, q, j0, j1] is
        j0 * 32 * K + i0 * 1024 + p * 64 + j1 * 2 + q
    which matches the C reference (with p = (i % 32) // 2, q = i % 2).

    Total bytes returned: K * N * sizeof(fp16). No padding.
    """
    if w_f32.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape {w_f32.shape}")
    k, n = w_f32.shape
    if k % 32 != 0 or n % 32 != 0:
        raise ValueError(
            f"K={k}, N={n} must both be multiples of 32 for HMX tile permute"
        )
    w_f16 = w_f32.astype(np.float16, copy=False)
    src = w_f16.reshape(k // 32, 16, 2, n // 32, 32)
    dst = src.transpose(3, 0, 1, 4, 2)
    return np.ascontiguousarray(dst)


def _self_test_permute():
    """Element-by-element check of permute_weight_to_fp16_tiles against the
    slow C-reference loop on a small fixture. Run once at module import to
    catch any regression in the vectorised path.
    """
    rng = np.random.default_rng(0)
    k, n = 64, 96
    w = rng.standard_normal((k, n), dtype=np.float32)
    fast = permute_weight_to_fp16_tiles(w).reshape(-1)
    slow = np.zeros(k * n, dtype=np.float16)
    w16 = w.astype(np.float16)
    for i in range(k):
        for j in range(n):
            i0, i1 = divmod(i, 32)
            j0, j1 = divmod(j, 32)
            tile_idx = j0 * (k // 32) + i0
            slow[tile_idx * 1024 + (i1 & ~1) * 32 + j1 * 2 + (i1 & 1)] = w16[i, j]
    if not np.array_equal(fast, slow):
        raise RuntimeError(
            "permute_weight_to_fp16_tiles: vectorised output does not match "
            "C-reference loop. Refusing to write a corrupt weight binary."
        )


_self_test_permute()


def save_qwen3_for_nntrainer_hmx(params, n_layers, file):
    """Convert and save weights in HMX-ready format for the HTP backend."""

    def save_fp32(weight):
        np.array(weight, dtype=np.float32).tofile(file)

    def save_linear_fp16(weight):
        # HuggingFace nn.Linear stores weight as [out, in] = [N, K]. The
        # pwf16 kernel's reference (test/unittest/unittest_htp_kernels.cpp:
        # 67-142) feeds the permute helper a [K, N] row-major source — i.e.
        # transposed relative to HF — and then computes
        #   C[i,j] = sum_l A[i,l] * W[l,j].
        # So we transpose HF [N, K] -> [K, N] before tile-permuting.
        w = np.array(weight, dtype=np.float32)
        if w.ndim != 2:
            raise ValueError(
                f"linear weight must be 2D, got shape {w.shape}"
            )
        n_hf, k_hf = w.shape  # HF: out, in
        w_kn = np.ascontiguousarray(w.T)  # [K, N]
        if k_hf % 32 == 0 and n_hf % 32 == 0:
            permute_weight_to_fp16_tiles(w_kn).tofile(file)
        else:
            # Non-32-aligned (e.g. small LoRA adapter dims): store plain
            # FP16 in HF [N, K] row-major. The C++ side's 32-align guard in
            # FloatTensor::dotFloat32Float16 falls through to the CPU path
            # for these tensors, which consumes the FC layer's standard
            # [N, K] FP16 buffer (same layout the previous wf16-only build
            # used end-to-end).
            print(
                f"  [warn] HF shape {w.shape} (K={k_hf}, N={n_hf}) not "
                f"32-aligned; storing plain FP16 [N, K] (CPU fallback at "
                f"runtime)"
            )
            w.astype(np.float16).tofile(file)

    def save_projection(layer_name, proj_name):
        """Helper for Linear-like projections (handles base/LoRA split)."""
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"
        if lora_key in params:
            save_linear_fp16(params[f"{layer_name}{proj_name}.base_layer.weight"])
            save_linear_fp16(params[f"{layer_name}{proj_name}.lora_A.default.weight"])
            save_linear_fp16(params[f"{layer_name}{proj_name}.lora_B.default.weight"])
        else:
            save_linear_fp16(params[f"{layer_name}{proj_name}.weight"])

    def save_attention(layer_name):
        """Save attention block weights for one transformer layer."""
        save_fp32(params[f"{layer_name}input_layernorm.weight"])

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")
            # Qwen3 QK-norm: q_norm / k_norm are FP32 (consumed by RMSNorm).
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                print(proj_norm_name)
                save_fp32(params[proj_norm_name])

    def save_feed_forward(layer_name):
        """Save MLP block weights for one transformer layer."""
        save_fp32(params[f"{layer_name}post_attention_layernorm.weight"])

        for proj in ["up_proj", "gate_proj", "down_proj"]:
            save_projection(layer_name, f"mlp.{proj}")

    # Embedding table (FP32, CPU-side lookup).
    save_fp32(params["model.embed_tokens.weight"])

    # Per-layer blocks.
    for layer_idx in range(n_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_attention(layer_prefix)
        save_feed_forward(layer_prefix)

    # Final RMSNorm (FP32) and lm_head Linear (FP16, no permute).
    save_fp32(params["model.norm.weight"])
    save_linear_fp16(params["lm_head.weight"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./Qwen3-0.6B")
    parser.add_argument(
        "--output_name", type=str, default="./nntr_qwen3_0_6b_hmx_w16a32.bin"
    )
    args = parser.parse_args()

    model_path = args.model_path
    output_name = args.output_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="float", trust_remote_code=True
    )
    model.eval()

    with open(output_name, "wb") as f_model:
        save_qwen3_for_nntrainer_hmx(
            model.state_dict(), config.num_hidden_layers, f_model
        )
