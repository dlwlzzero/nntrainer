// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
 *
 * @file   lfm2_moe_causallm.cpp
 * @date   06 July 2026
 * @brief  This defines the LFM2-8B-A1B Mixture-of-Experts causal LM.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <lfm2_moe_causallm.h>
#include <lfm2_moe_layer.h>

#include <algorithm>
#include <app_context.h>
#include <cstdio>
#include <cstdlib>
#include <engine.h>
#include <functional>
#include <layer_node.h>
#include <llm_util.hpp>
#include <map>
#include <model.h>

#ifdef ENABLE_HEXKL
#include <compute_ops.h>
#include <htp_graph_desc.h>
#endif

namespace causallm {

void Lfm2MoeCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                      json &nntr_cfg) {
  // Parse the LFM2 backbone parameters (dims, layer_types, conv, ...).
  Lfm2CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);

  // MoE-specific parameters.
  try {
    NUM_EXPERTS = cfg["num_experts"];
    NUM_EXPERTS_PER_TOK = cfg["num_experts_per_tok"];
    MOE_INTERMEDIATE_SIZE = cfg["moe_intermediate_size"];
  } catch (const std::exception &e) {
    throw std::runtime_error(
      "Lfm2Moe: num_experts, num_experts_per_tok and moe_intermediate_size "
      "must be specified in the config file");
  }
  // Layers [0, num_dense_layers) keep the dense SwiGLU FFN. Optional (default
  // 0).
  NUM_DENSE_LAYERS = cfg.value("num_dense_layers", 0);

  // MoE expert FFN weight dtype. Defaults to FC_LAYER_DTYPE (set by
  // Lfm2CausalLM::setupParameters above) so an unquantized or fc_dtype==
  // moe_dtype model is unaffected; nntr_quantize's --moe_dtype writes
  // moe_layer_dtype when the two are meant to differ (see quantize.cpp's
  // buildLayerDtypeMap docstring).
  MOE_LAYER_DTYPE = nntr_cfg.value("moe_layer_dtype", FC_LAYER_DTYPE);

  // MoE FFN engine. Defaults to "cpu" so an existing run is unaffected
  // unless nntr_config.json opts in; MOE_HTP_LAYERS (empty = every MoE
  // layer) bounds a real checkpoint's device run to what the HTP weight
  // registry can hold at once -- see the two fields' docs in the header.
  MOE_ENGINE = nntr_cfg.value("moe_engine", std::string("cpu"));
  MOE_HTP_LAYERS =
    parseLayerIdList(nntr_cfg.value("moe_htp_layers", std::string("")));

#ifdef ENABLE_HEXKL
  // [#85] NNTR_HTP_FORWARD=1: describe this model's decode step to the HTP
  // backend once, so its MoE calls at M == 1 can go through the per-token
  // entry. The words are the header's LFM2 builder over this config; the
  // backend validates them with the same validator the skel runs. Off by
  // default, and only meaningful with the MoE FFN on the HTP.
  const char *fwd = std::getenv("NNTR_HTP_FORWARD");
  if (fwd != nullptr && std::atoi(fwd) != 0 && MOE_ENGINE == "htp") {
    htp_graph_lfm2_shape shape;
    shape.n_layers = static_cast<uint32_t>(NUM_LAYERS);
    shape.n_dense_layers = NUM_DENSE_LAYERS;
    shape.hidden = static_cast<uint32_t>(DIM);
    shape.inter_dense = static_cast<uint32_t>(INTERMEDIATE_SIZE);
    shape.inter_moe = MOE_INTERMEDIATE_SIZE;
    shape.n_experts = NUM_EXPERTS;
    shape.top_k = NUM_EXPERTS_PER_TOK;
    shape.n_heads = static_cast<uint32_t>(NUM_HEADS);
    shape.n_kv_heads = static_cast<uint32_t>(NUM_KEY_VALUE_HEADS);
    shape.head_dim = static_cast<uint32_t>(HEAD_DIM);
    shape.vocab = NUM_VOCAB;
    shape.max_seq = MAX_SEQ_LEN;
    shape.eps = NORM_EPS;
    std::vector<uint8_t> attn(layer_types_.size());
    for (size_t l = 0; l < layer_types_.size(); ++l)
      attn[l] = layer_types_[l] != "conv";
    // [#130] NNTR_HTP_FORWARD_KINDS (comma-separated kind names) chooses
    // which kinds carry the resident bit; the default is #130's six kinds,
    // so its recipes keep their meaning -- #132's ADD and ROUTER_TOPK are
    // opt-in (...,ADD,ROUTER_TOPK). The bit is the one source of truth for
    // both sides (plan 130 section 3.2).
    const char *kinds = std::getenv("NNTR_HTP_FORWARD_KINDS");
    const uint32_t mask =
      kinds != nullptr
        ? htp_graph_kinds_parse(kinds)
        : (HTP_GRAPH_KIND_BIT(HTP_OP_MOE) | HTP_GRAPH_KIND_BIT(HTP_OP_RMSNORM) |
           HTP_GRAPH_KIND_BIT(HTP_OP_QK_NORM) |
           HTP_GRAPH_KIND_BIT(HTP_OP_ROPE) |
           HTP_GRAPH_KIND_BIT(HTP_OP_CONV1D_GATE) |
           HTP_GRAPH_KIND_BIT(HTP_OP_ATTN_M1));
    if (mask == 0u)
      throw std::runtime_error(
        std::string("Lfm2Moe: NNTR_HTP_FORWARD_KINDS: unknown kind in '") +
        kinds + "'");
    std::vector<uint32_t> words(
      htp_graph_words_for(shape.n_layers, HTP_GRAPH_MAX_OPS));
    const uint32_t n =
      htp_graph_lfm2_build(words.data(), static_cast<uint32_t>(words.size()),
                           &shape, attn.data(), mask);
    if (n == 0u)
      throw std::runtime_error("Lfm2Moe: NNTR_HTP_FORWARD: the decode op list "
                               "does not fit HTP_GRAPH_MAX_OPS");
    words.resize(n);
    if (!nntrainer::get_htp_ops()->set_decode_graph_desc(words))
      throw std::runtime_error(
        "Lfm2Moe: NNTR_HTP_FORWARD: the HTP backend has no per-token entry");
  }
#endif
}

Tensor Lfm2MoeCausalLM::createMoeLayer(const int layer_id, Tensor input) {
  const std::string engine =
    (MOE_HTP_LAYERS.empty() || MOE_HTP_LAYERS.count(layer_id)) ? MOE_ENGINE
                                                               : "cpu";
  LayerHandle moe(createLayer(
    "lfm2_moe",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", MOE_INTERMEDIATE_SIZE),
     withKey("num_experts", NUM_EXPERTS),
     withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
     withKey("moe_activation", "swish"),
     withKey("weight_dtype", MOE_LAYER_DTYPE), withKey("engine", engine)}));
  return moe(input);
}

Tensor Lfm2MoeCausalLM::createMlp(const int layer_id, int dim, int hidden_dim,
                                  Tensor input) {
  // Dense SwiGLU FFN for the first NUM_DENSE_LAYERS layers.
  if (layer_id < static_cast<int>(NUM_DENSE_LAYERS))
    return Transformer::createMlp(layer_id, dim, hidden_dim, input);

  // MoE FFN for the remaining layers.
  return createMoeLayer(layer_id, input);
}

void Lfm2MoeCausalLM::load_weight(const std::string &weight_path) {
  Lfm2CausalLM::load_weight(weight_path);

  // [#162] The prefetch overlap (plan 162 section 2, lever (b)): layer
  // N's MoE decode call is a CPU wait, and layer N + 1 opens with an FC
  // GEMV on the CPU (conv in_proj, conv_block or qkv). Each MoE layer gets
  // the first S MiB of those weights to read during its call. Unset or 0
  // is off; the last layer has no next FC and gets nothing. Off under FSU:
  // the swap can unmap a weight between the load and the touch.
  const char *env = std::getenv("NNTR_MOE_PREFETCH_MIB");
  const size_t cap = env != nullptr ? std::strtoul(env, nullptr, 10) << 20 : 0;
  if (cap == 0 || MEMORY_SWAP)
    return;
  std::map<int, Lfm2MoELayer *> moe;
  std::map<int, std::vector<std::pair<const char *, size_t>>> fc;
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&moe, &fc](ml::train::Layer &l, nntrainer::RunLayerContext &context,
                     void *) {
      const std::string name = l.getName(), type = l.getType();
      int id = -1;
      if (std::sscanf(name.c_str(), "layer%d_", &id) != 1)
        return;
      if (type == Lfm2MoELayer::type) {
        auto *m = dynamic_cast<Lfm2MoELayer *>(
          static_cast<nntrainer::LayerNode &>(l).getLayer());
        if (m != nullptr)
          moe[id] = m;
        return;
      }
      if (fc.count(id) || (type != "fully_connected" && type != "conv_block" &&
                           type != "qkv_layer"))
        return;
      // the layer's Q4_0 weights in file order: the bytes the decode GEMV
      // streams (a QS4CX weight is read from its packed copy instead;
      // FP32 norms and the conv kernel are small)
      auto &spans = fc[id];
      for (auto *w : context.getWeights()) {
        auto &t = w->getVariableRef();
        if (t.getDataType() == ml::train::TensorDim::DataType::Q4_0)
          spans.emplace_back(t.getData<char>(), t.getMemoryBytes());
      }
    };
  model->forEachLayer(fn, nullptr);
  unsigned wired = 0;
  for (auto &[id, m] : moe) {
    auto it = fc.find(id + 1);
    if (it == fc.end() || it->second.empty())
      continue;
    std::vector<std::pair<const char *, size_t>> spans;
    size_t left = cap;
    for (const auto &sp : it->second) {
      if (left == 0)
        break;
      spans.emplace_back(sp.first, std::min(sp.second, left));
      left -= spans.back().second;
    }
    m->setWindowPrefetch(std::move(spans));
    ++wired;
  }
  std::fprintf(stderr, "[CausalLM] moe prefetch: on mib=%zu layers=%u\n",
               cap >> 20, wired);
}

void Lfm2MoeCausalLM::registerCustomLayers() {

  Lfm2CausalLM::registerCustomLayers();
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::Lfm2MoELayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register Lfm2MoELayer factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm
