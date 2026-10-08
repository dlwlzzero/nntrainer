// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gemma4_moe_causallm.cpp
 * @brief  Gemma4 MoE causal language model implementation.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs
 */

#include <app_context.h>
#include <engine.h>
#include <gemma4_moe_causallm.h>
#include <gemma4_moe_layer.h>
#include <layer_context.h>
#include <lfm2_moe_layer.h>
#include <llm_util.hpp>
#include <model.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>

#ifdef ENABLE_HEXKL
#include <compute_ops.h>
#include <htp_graph_desc.h>
#endif

namespace causallm {

void Gemma4MoECausalLM::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  Gemma4CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);

  NNTR_THROW_IF(!cfg.contains("num_experts") || cfg["num_experts"].is_null() ||
                  !cfg.contains("top_k_experts") ||
                  cfg["top_k_experts"].is_null() ||
                  !cfg.contains("moe_intermediate_size") ||
                  cfg["moe_intermediate_size"].is_null(),
                std::invalid_argument)
    << "[Gemma4MoE] num_experts, top_k_experts, and moe_intermediate_size "
       "must be provided";

  num_experts = cfg["num_experts"].get<unsigned int>();
  top_k_experts = cfg["top_k_experts"].get<unsigned int>();
  moe_intermediate_size = cfg["moe_intermediate_size"].get<unsigned int>();
  moe_cache_size =
    nntr_cfg.contains("moe_cache_size") && !nntr_cfg["moe_cache_size"].is_null()
      ? nntr_cfg["moe_cache_size"].get<unsigned int>()
      : 0;

  NNTR_THROW_IF(num_experts == 0 || top_k_experts == 0 ||
                  top_k_experts > num_experts || moe_intermediate_size == 0,
                std::invalid_argument)
    << "[Gemma4MoE] invalid expert configuration";
  NNTR_THROW_IF(NUM_KV_SHARED_LAYERS > 0, std::invalid_argument)
    << "[Gemma4MoE] shared KV layers are not supported";

  // [plan 201 S4] moe_engine=htp: the lfm2_moe layer with the softmax
  // router, QS4CX_WH experts and the expert pool; its experts and the
  // dense FFN are GeGLU on the HTP (the session's MoE flag, sent with the
  // first MoE call)
  moe_engine = nntr_cfg.value("moe_engine", std::string("cpu"));
  moe_layer_dtype = nntr_cfg.value("moe_layer_dtype", FC_LAYER_DTYPE);
  // The two expert layouts: nntr_quantize_stream fuses gate | up for
  // QS4CX_WH and QS2CX_WH (the HTP layer's), every other dtype keeps them
  // apart (gemma4_moe's)
  NNTR_THROW_IF((moe_engine == "htp") != (moe_layer_dtype == "QS4CX_WH" ||
                                          moe_layer_dtype == "QS2CX_WH"),
                std::invalid_argument)
    << "[Gemma4MoE] moe_engine=htp takes QS4CX_WH or QS2CX_WH experts and "
       "only they do "
       "(moe_engine="
    << moe_engine << ", moe_layer_dtype=" << moe_layer_dtype << ")";
#ifdef ENABLE_HEXKL
  if (moe_engine == "htp")
    nntrainer::get_htp_ops()->set_moe_geglu(true);
  // [plan 201 S4] NNTR_HTP_E2E=1: the whole decode token on the HTP, one
  // call a token; the list is built after load (load_weight)
  const char *e2e_env = std::getenv("NNTR_HTP_E2E");
  htp_e2e =
    e2e_env != nullptr && std::atoi(e2e_env) != 0 && moe_engine == "htp";
  NNTR_THROW_IF(htp_e2e && (HIDDEN_SIZE_PER_LAYER_INPUT > 0 ||
                            ATTN_LOGIT_SOFTCAPPING > 0.0f),
                std::invalid_argument)
    << "[Gemma4MoE] NNTR_HTP_E2E=1: the decode list has no per-layer input "
       "(PLE) and no attention logit soft-cap";
#endif
}

void Gemma4MoECausalLM::load_weight(const std::string &weight_path) {
  Gemma4CausalLM::load_weight(weight_path);
#ifdef ENABLE_HEXKL
  if (!htp_e2e)
    return;
  // The weights by their own names, "<layer>:<weight>" (#4415's fused
  // layers hold several each: qkv's gammas and projections, dense_ffn's
  // gate / up / down and its two gammas, residual_add's gamma and the
  // block's scalar), never by index.
  std::map<std::string, nntrainer::Tensor *> w;
  model->forEachLayer(
    [&w](ml::train::Layer &, nntrainer::RunLayerContext &rc, void *) {
      for (auto *t : rc.getWeights())
        w[t->getName()] = &t->getVariableRef();
    },
    nullptr);
  auto weight = [&w](const std::string &name) -> nntrainer::Tensor & {
    auto it = w.find(name);
    if (it == w.end())
      throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: no weight " + name);
    return *it->second;
  };
  auto f32 = [&weight](const std::string &name, size_t n) -> const float * {
    nntrainer::Tensor &t = weight(name);
    if (t.getDataType() != ml::train::TensorDim::DataType::FP32 ||
        t.size() != n)
      throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: " + name +
                               " is not " + std::to_string(n) + " FP32 values");
    return t.getData<float>();
  };

  // The list, now that layer_scalar (a checkpoint weight) is in memory.
  const uint32_t n_layers = static_cast<uint32_t>(NUM_LAYERS);
  std::vector<uint8_t> is_full(n_layers);
  std::vector<float> scalar(n_layers);
  for (uint32_t l = 0; l < n_layers; ++l) {
    is_full[l] = !isSlidingAttentionLayer(static_cast<int>(l));
    scalar[l] = f32(
      "layer" + std::to_string(l) + "_post_ffn_norm:scalar_multiplier", 1)[0];
    // the list's ADD stores 0.0f as "no multiplier" (#221)
    if (scalar[l] == 0.0f)
      throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: layer " +
                               std::to_string(l) +
                               "'s layer_scalar is 0, which the decode list "
                               "cannot carry");
  }
  htp_graph_gemma_shape shape{};
  shape.n_layers = n_layers;
  shape.hidden = static_cast<uint32_t>(DIM);
  shape.inter_dense = static_cast<uint32_t>(INTERMEDIATE_SIZE);
  shape.inter_moe = moe_intermediate_size;
  shape.n_experts = num_experts;
  shape.top_k = top_k_experts;
  shape.n_heads = static_cast<uint32_t>(NUM_HEADS);
  shape.n_kv = static_cast<uint32_t>(NUM_KEY_VALUE_HEADS);
  shape.head_dim = static_cast<uint32_t>(HEAD_DIM);
  shape.n_kv_full = ATTENTION_K_EQ_V
                      ? NUM_GLOBAL_KEY_VALUE_HEADS
                      : static_cast<uint32_t>(NUM_KEY_VALUE_HEADS);
  shape.head_dim_full = GLOBAL_HEAD_DIM;
  shape.k_eq_v = ATTENTION_K_EQ_V ? 1u : 0u;
  shape.window = SLIDING_WINDOW;
  shape.vocab = NUM_VOCAB;
  shape.max_seq = MAX_SEQ_LEN;
  shape.eps = NORM_EPS;
  // a tied head folds the final norm and the softcap (#4415's
  // FOLD_OUTPUT_NORM), so the DSP caps; an untied one keeps the CPU's
  // logit_softcapping layer after the head, which caps the logits
  shape.softcap = TIE_WORD_EMBEDDINGS ? FINAL_LOGIT_SOFTCAPPING : 0.0f;
  std::vector<uint32_t> words(htp_graph_words_for(n_layers, HTP_GRAPH_MAX_OPS));
  const uint32_t n_words = htp_graph_gemma_build(
    words.data(), static_cast<uint32_t>(words.size()), &shape, is_full.data(),
    scalar.data(), HTP_GRAPH_KINDS_ALL);
  if (n_words == 0u)
    throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: the decode op list "
                             "does not fit HTP_GRAPH_MAX_OPS");
  words.resize(n_words);
  auto *ops = nntrainer::get_htp_ops();
  if (!ops->set_decode_graph_desc(words))
    throw std::runtime_error(
      "[Gemma4MoE] NNTR_HTP_E2E: the HTP backend has no per-token entry");

  // The f32 parameters by name. The RMSNORMs of a layer in the builder's
  // order (htp_graph_gemma_build): input, post-attention, the MoE branch's
  // pre / post, the dense branch's pre / post, post-FFN; the tail's final.
  // #4415's graph holds four of them in fused layers: the input norm in
  // qkv, the dense branch's two in dense_ffn, post-attention and post-FFN
  // in the residual_adds.
  static const char *const kNorms[7] = {
    "_qkv:in_norm_gamma",    "_post_attention_norm:gamma",
    "_pre_ffn_norm_2:gamma", "_post_ffn_norm_2:gamma",
    "_ffn:in_norm_gamma",    "_ffn:out_norm_gamma",
    "_post_ffn_norm:gamma"};
  const uint32_t n_ops = words[3], H = shape.hidden, E = num_experts;
  auto hand = [ops](uint32_t op, uint32_t which, const float *d, size_t n) {
    if (!ops->set_decode_graph_param(op, which, d, static_cast<unsigned>(n)))
      throw std::runtime_error(
        "[Gemma4MoE] NNTR_HTP_E2E: the backend took no parameter");
  };
  htp_params.clear();
  uint32_t layer = HTP_GRAPH_NO_OP, norm = 0;
  for (uint32_t i = 0; i < n_ops; ++i) {
    const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
    if (op->layer != layer) {
      layer = op->layer;
      norm = 0;
    }
    const std::string p = "layer" + std::to_string(layer);
    if (op->kind == HTP_OP_RMSNORM) {
      if (layer < n_layers && norm == 7u)
        throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: layer " + p +
                                 " has more RMSNORM ops than names");
      const std::string name = layer < n_layers ? p + kNorms[norm++]
                               : TIE_WORD_EMBEDDINGS
                                 ? std::string("output_of_causallm:gamma")
                                 : std::string("output_norm:gamma");
      hand(i, HTP_GRAPH_PARAM_GAMMA, f32(name, H), H);
    } else if (op->kind == HTP_OP_QK_NORM) {
      const uint32_t hd = op->head_dim;
      std::vector<float> g(2u * hd);
      std::copy_n(f32(p + "_qkv:q_norm_gamma", hd), hd, g.begin());
      std::copy_n(f32(p + "_qkv:k_norm_gamma", hd), hd, g.begin() + hd);
      htp_params.push_back(std::move(g));
      hand(i, HTP_GRAPH_PARAM_GAMMA, htp_params.back().data(), 2u * hd);
    } else if (op->kind == HTP_OP_ROUTER_TOPK) {
      // lfm2_moe (softmax): router [H][E], router_scale [H], per-expert
      // scale [E]; ROUTER_BIAS is g | per-expert scale with g the layer's
      // own router_scale / sqrt(H) (Lfm2MoELayer::route)
      const std::string m = p + "_sparse_moe:";
      hand(i, HTP_GRAPH_PARAM_ROUTER_W, f32(m + "router", size_t(H) * E),
           size_t(H) * E);
      const float *rs = f32(m + "router_scale", H),
                  *pes = f32(m + "router_per_expert_scale", E);
      const float hs = 1.0f / std::sqrt(static_cast<float>(H));
      std::vector<float> b(H + E);
      for (uint32_t f = 0; f < H; ++f)
        b[f] = rs[f] * hs;
      std::copy_n(pes, E, b.begin() + H);
      htp_params.push_back(std::move(b));
      hand(i, HTP_GRAPH_PARAM_ROUTER_BIAS, htp_params.back().data(), H + E);
    }
  }

  // The Q4_0 weights of the FC / DENSE_FFN / LM_HEAD ops, in list order:
  // q | k (| v), o; up, gate, down; the tied table.
  auto q4 = [&weight, ops](const std::string &name, bool tied) {
    nntrainer::Tensor &t = weight(name);
    if (t.getDataType() != ml::train::TensorDim::DataType::Q4_0)
      throw std::runtime_error("[Gemma4MoE] NNTR_HTP_E2E: " + name +
                               " is not Q4_0 (the resident FC kinds' type)");
    const unsigned K = tied ? t.width() : t.height();
    const unsigned N = tied ? t.height() : t.width();
    if (!ops->add_decode_graph_q4_0(t.getData<char>(), K, N, tied))
      throw std::runtime_error(
        "[Gemma4MoE] NNTR_HTP_E2E: the backend took no Q4_0 weight");
  };
  layer = HTP_GRAPH_NO_OP;
  uint32_t fc = 0;
  for (uint32_t i = 0; i < n_ops; ++i) {
    const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
    if (op->layer != layer) {
      layer = op->layer;
      fc = 0;
    }
    const std::string p = "layer" + std::to_string(layer);
    if (op->kind == HTP_OP_FC && fc++ == 0) {
      q4(p + "_qkv:qweight", false);
      q4(p + "_qkv:kweight", false);
      if (w.count(p + "_qkv:vweight")) // absent under k = v
        q4(p + "_qkv:vweight", false);
    } else if (op->kind == HTP_OP_FC) {
      q4(p + "_attention_out:weight", false);
    } else if (op->kind == HTP_OP_DENSE_FFN) {
      // the list's order up, gate, down; the file's is gate-first
      q4(p + "_ffn:up", false);
      q4(p + "_ffn:gate", false);
      q4(p + "_ffn:down", false);
    } else if (op->kind == HTP_OP_LM_HEAD) {
      q4(TIE_WORD_EMBEDDINGS ? "embedding0:Embedding"
                             : "output_of_causallm:weight",
         TIE_WORD_EMBEDDINGS);
    }
  }
  std::fprintf(stderr,
               "[HTP] gemma: list n_ops=%u layers=%u params=%zu by name\n",
               n_ops, n_layers, htp_params.size());
#endif
}

void Gemma4MoECausalLM::repack_weight() {
  Gemma4CausalLM::repack_weight();
#ifdef ENABLE_HEXKL
  // [plan 201 S4] the E2E FC arena, after the experts' registration above
  if (htp_e2e)
    nntrainer::get_htp_ops()->finish_decode_graph_q4_0();
#endif
}

Tensor Gemma4MoECausalLM::createMoe(const int layer_id, Tensor post_attention) {
  LayerHandle pre_sparse_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_pre_ffn_norm_2"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  Tensor sparse_input = pre_sparse_norm(post_attention);
  // [plan 201 S4] moe_engine=htp: lfm2_moe with the softmax router (the
  // experts' file layout is the fused gate | up of nntr_quantize_stream's
  // QS4CX_WH writer); else #4296's gemma4_moe
  const std::string moe_name =
    "layer" + std::to_string(layer_id) + "_sparse_moe";
  LayerHandle sparse_moe(
    moe_engine == "htp"
      ? createLayer(
          "lfm2_moe",
          {withKey("name", moe_name),
           withKey("unit", std::to_string(moe_intermediate_size)),
           withKey("num_experts", std::to_string(num_experts)),
           withKey("num_experts_per_token", std::to_string(top_k_experts)),
           withKey("moe_activation", "tanh_gelu"),
           withKey("moe_router", "softmax"),
           withKey("epsilon", std::to_string(NORM_EPS)),
           withKey("weight_dtype", moe_layer_dtype),
           withKey("engine", moe_engine)})
      : createLayer(
          "gemma4_moe",
          {withKey("name", moe_name),
           withKey("unit", std::to_string(moe_intermediate_size)),
           withKey("num_experts", std::to_string(num_experts)),
           withKey("num_experts_per_token", std::to_string(top_k_experts)),
           withKey("moe_cache_size", std::to_string(moe_cache_size)),
           withKey("moe_activation", "tanh_gelu"),
           withKey("epsilon", std::to_string(NORM_EPS)),
           withKey("weight_dtype", moe_layer_dtype)}));
  Tensor sparse_output = sparse_moe({sparse_input, post_attention});

  LayerHandle post_sparse_norm(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_post_ffn_norm_2"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));
  return post_sparse_norm(sparse_output);
}

void Gemma4MoECausalLM::registerCustomLayers() {
  Gemma4CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  try {
    app_context->registerFactory(nntrainer::createLayer<Gemma4MoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
  // [plan 201 S4] the HTP MoE layer (moe_engine=htp)
  try {
    app_context->registerFactory(nntrainer::createLayer<Lfm2MoELayer>);
  } catch (std::invalid_argument &e) {
    (void)e; // already registered by another model of this process
  }
}

} // namespace causallm
