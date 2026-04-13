## @file weight_converter_hmx.py
## @brief HMX-friendly weight conversion script for qwen3 model (HTP target)
## @author Eunju Yang <ej.yang@samsung.com>
##
## This converter produces an nntrainer weight binary pre-formatted for the
## HTP HMX matmul kernel `hmx_mat_mul_af32_wf16_of32`, which expects every
## Linear weight as a row-major [N x K] FP16 buffer (N = out_features,
## K = in_features). Runtime consumer: nntrainer/tensor/float_tensor.cpp
## FloatTensor::dotFloat32Float16 HTP path (lines ~981-1030) — that path
## memcpy's the weight directly into RPC shared memory with no conversion.
##
## Differences from weight_converter.py:
##   * Linear weights are saved as FP16 (np.float16), not FP32.
##   * Linear weights are NOT permuted: HuggingFace stores nn.Linear.weight
##     as [out, in] = [N, K] natively, which already matches the HMX kernel
##     contract. The legacy converter applied .permute(1, 0) to produce
##     [K, N] for the CPU path; we skip that.
##   * Embeddings, RMSNorm weights (input/post-attention layernorm, Qwen3
##     q_norm/k_norm, final model.norm), and biases stay FP32 — they are
##     either consumed on CPU or by the HVX RMSNorm kernel which requires
##     FP32.
##   * lm_head.weight is treated as a Linear projection: saved as [N, K]
##     FP16 (again no permute).

import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def save_qwen3_for_nntrainer_hmx(params, n_layers, file):
    """Convert and save weights in HMX-ready format for the HTP backend."""

    def save_fp32(weight):
        np.array(weight, dtype=np.float32).tofile(file)

    def save_linear_fp16(weight):
        # No permute: HuggingFace layout [out, in] = [N, K] already matches
        # the HMX kernel's expected [N x K] row-major FP16 layout.
        np.array(weight, dtype=np.float16).tofile(file)

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
