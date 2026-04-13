// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   custom_rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cstring>
#include <iostream>

#include "rms_norm.h"

#if defined(ENABLE_HTP) && ENABLE_HTP == 1
#include <htp_interface.h>
#endif

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void RMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);
}

void RMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {}

void RMSNormLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  unsigned int _from = from;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      bool htp_done = false;
      const int ne0 = static_cast<int>(in_step.getDim().width());
      const int ne1 = static_cast<int>(in_step.getDim().height());

#if defined(ENABLE_HTP) && ENABLE_HTP == 1
      {
        auto &htp = nntrainer::htp::HtpInterface::instance();
        if (htp.htp_ops_rms_norm_f32 && htp.alloc_shared_mem_buf &&
            htp.free_shared_mem_buf && htp.get_global_handle) {
          auto handle = htp.get_global_handle();
          if (handle != 0) {
            float eps_f = epsilon;
            int32_t eps_bits;
            std::memcpy(&eps_bits, &eps_f, sizeof(float));

            // HVX writes/reads up to VLEN=32 floats past the end of the last
            // row; reserve padding only for the final row.
            const int ne0_padded = ((ne0 + 31) / 32) * 32;
            const size_t buf_elems = static_cast<size_t>(
              (ne1 > 0 ? (ne1 - 1) * ne0 : 0) + ne0_padded);
            const size_t buf_size = buf_elems * sizeof(float);
            const size_t total_size = buf_size * 2; // src + dst back-to-back

            void *io_buf = nullptr;
            int io_fd = -1;
            int err = htp.alloc_shared_mem_buf(&io_buf, &io_fd, total_size);
            if (err == 0 && io_buf) {
              char *base = static_cast<char *>(io_buf);
              std::memset(base, 0, total_size);
              std::memcpy(base, in_step.getData<float>(),
                          static_cast<size_t>(ne0) * ne1 * sizeof(float));

              err = htp.htp_ops_rms_norm_f32(
                handle, io_fd, static_cast<int>(buf_size), io_fd, 0, ne0, ne1,
                eps_bits);

              if (err == 0) {
                std::memcpy(out_step.getData<float>(), base + buf_size,
                            static_cast<size_t>(ne0) * ne1 * sizeof(float));
                htp_done = true;
              }
              htp.free_shared_mem_buf(io_buf, io_fd, total_size);
            }
          }
        }
      }
#endif

      if (!htp_done) {
        auto t = in_step.multiply(in_step).average(3).add(epsilon);
        t.inv_sqrt_i();
        in_step.multiply(t, out_step);
      }
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }
    out_step.multiply_i(gamma);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }
}

void RMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void RMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new RMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
