// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   fake_quantize.cpp
 * @date   31 August 2026
 * @brief  W8_CX quantize-dequantize (QDQ) pass over a HuggingFace safetensors
 *         file. Every 2D F32/BF16 tensor (linear projections and the
 *         embedding table, laid out [out][in] = N x K) is round-tripped through
 *         quant_w8cx_f32 / dequant_w8cx_f32 with the exact same primitive
 *         nntr_quantize uses, and every tensor is written back as F32. The
 *         result is the fp32 reference for the W8_CX accuracy gate.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jiho Lee <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * @usage  nntr_fakequant <in.safetensors> <out.safetensors>
 */

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"
#include <cpu_backend.h>

using json = nlohmann::ordered_json;

namespace {

/**
 * @brief One tensor of the input file and its output disposition.
 */
struct Item {
  std::string name;
  std::string dtype;
  std::vector<size_t> shape;
  uint64_t in_start;
  uint64_t in_end;
  bool fq;
};

bool endsWith(const std::string &s, const std::string &suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " <in.safetensors> <out.safetensors>\n";
    return EXIT_FAILURE;
  }
  const std::string in_path = argv[1];
  const std::string out_path = argv[2];
  if (endsWith(in_path, ".index.json")) {
    std::cerr << "error: sharded checkpoints are not supported; pass a single "
                 "safetensors file\n";
    return EXIT_FAILURE;
  }

  std::ifstream in(in_path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "error: cannot open " << in_path << "\n";
    return EXIT_FAILURE;
  }
  uint64_t header_len = 0;
  in.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
  std::string header_json(header_len, '\0');
  in.read(header_json.data(), static_cast<std::streamsize>(header_len));
  if (!in) {
    std::cerr << "error: cannot read safetensors header\n";
    return EXIT_FAILURE;
  }
  const uint64_t data_base = sizeof(header_len) + header_len;

  json header;
  try {
    header = json::parse(header_json);
  } catch (const std::exception &e) {
    std::cerr << "error: bad header json: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  // Pass 1: plan the output layout. Everything becomes F32, offsets are
  // recomputed in input order; __metadata__ is carried over (format "pt" is
  // what transformers requires to load the result back).
  json out_header = json::object();
  out_header["__metadata__"] = header.contains("__metadata__")
                                 ? header["__metadata__"]
                                 : json{{"format", "pt"}};
  std::vector<Item> items;
  uint64_t out_off = 0;
  for (const auto &[name, t] : header.items()) {
    if (name == "__metadata__")
      continue;
    Item it{name,
            t["dtype"].get<std::string>(),
            t["shape"].get<std::vector<size_t>>(),
            t["data_offsets"][0].get<uint64_t>(),
            t["data_offsets"][1].get<uint64_t>(),
            false};
    if (it.dtype != "F32" && it.dtype != "BF16") {
      std::cerr << "error: unsupported dtype " << it.dtype << " for " << name
                << " (only F32/BF16 inputs are handled)\n";
      return EXIT_FAILURE;
    }
    it.fq = it.shape.size() == 2;
    size_t numel = 1;
    for (size_t d : it.shape)
      numel *= d;
    out_header[name] = {{"dtype", "F32"},
                        {"shape", it.shape},
                        {"data_offsets", {out_off, out_off + numel * 4}}};
    out_off += numel * 4;
    items.push_back(std::move(it));
  }

  std::string out_json = out_header.dump();
  out_json.append((8 - out_json.size() % 8) % 8, ' ');

  std::ofstream out(out_path, std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "error: cannot create " << out_path << "\n";
    return EXIT_FAILURE;
  }
  const uint64_t out_header_len = out_json.size();
  out.write(reinterpret_cast<const char *>(&out_header_len),
            sizeof(out_header_len));
  out.write(out_json.data(), static_cast<std::streamsize>(out_header_len));

  // Pass 2: stream tensors through, QDQ on the 2D ones.
  size_t fq_tensors = 0;
  std::vector<char> raw;
  std::vector<float> f32;
  std::vector<int8_t> q;
  std::vector<float> scales;
  for (const auto &it : items) {
    const size_t nbytes = it.in_end - it.in_start;
    raw.resize(nbytes);
    in.seekg(static_cast<std::streamoff>(data_base + it.in_start));
    in.read(raw.data(), static_cast<std::streamsize>(nbytes));
    if (!in) {
      std::cerr << "error: short read on " << it.name << "\n";
      return EXIT_FAILURE;
    }

    if (it.dtype == "BF16") {
      const size_t numel = nbytes / 2;
      f32.resize(numel);
      const uint16_t *h = reinterpret_cast<const uint16_t *>(raw.data());
      for (size_t i = 0; i < numel; ++i) {
        uint32_t bits = static_cast<uint32_t>(h[i]) << 16;
        std::memcpy(&f32[i], &bits, sizeof(float));
      }
    } else {
      f32.resize(nbytes / 4);
      std::memcpy(f32.data(), raw.data(), nbytes);
    }

    if (it.fq) {
      const size_t N = it.shape[0];
      const size_t K = it.shape[1];
      q.resize(N * K);
      scales.resize(N);
      nntrainer::quant_w8cx_f32(N, K, f32.data(), q.data(), scales.data());
      nntrainer::dequant_w8cx_f32(N, K, q.data(), scales.data(), f32.data());
      ++fq_tensors;
    }

    out.write(reinterpret_cast<const char *>(f32.data()),
              static_cast<std::streamsize>(f32.size() * sizeof(float)));
    std::cout << it.name << " [";
    for (size_t i = 0; i < it.shape.size(); ++i)
      std::cout << (i ? "," : "") << it.shape[i];
    std::cout << "] " << it.dtype << " " << (it.fq ? "fq" : "copy") << "\n";
  }
  if (!out) {
    std::cerr << "error: write failed on " << out_path << "\n";
    return EXIT_FAILURE;
  }
  std::cout << "fq_tensors=" << fq_tensors << "\n";
  return EXIT_SUCCESS;
}
