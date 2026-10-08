// SPDX-License-Identifier: Apache-2.0
/*
 * Ties the offline 2-bit packer to the kernel that expands it.
 *
 * whPack2 (htp_wh_palette.h, C++, runs in the quantizer) and
 * hvx_expand_i2i4 (C, runs on the DSP) are inverse by construction and by
 * nothing else -- they are compiled by different toolchains into different
 * binaries and share no code. If they drift, a QS2CX_WH model silently
 * holds different weights than the QS4CX_WH model whose perplexity was
 * measured to justify it, and nothing downstream would say so.
 *
 * So the check is whPack2 -> expand == whPack, byte for byte, which is the
 * one property the whole 2-bit case rests on. C++ rather than C99 like its
 * neighbours because whPack2 is a C++ header; the kernel comes in through
 * its extern "C" declarations.
 *
 * [plan 229 S2] With two FC WH sidecars as arguments (one written at two
 * bits where it could, its --bits4 twin), the same property on every image
 * of a real file: expand(codes, palette) == the twin's whPack bytes, the
 * scales and column sums byte-equal, the index the same but for bits and
 * offsets.
 *
 * What this does NOT check: the HVX path. On the host, hvx_expand_i2i4
 * compiles to its scalar twin. The vlut32 form is gated by the device test
 * HvxExpandI2I4.MatchesScalarBitExact instead -- see hvx_expand_i2i4.h.
 */

#include "htp_wh_layout.h"
#include "htp_wh_palette.h"

extern "C" {
#include "hvx_expand_i2i4.h"
/* hvx_worker_pool.h itself is not includable from C++ (_Atomic members);
   these two are all this check needs from it. */
hvx_worker_pool *hvx_worker_pool_create(uint32_t n_workers);
void hvx_worker_pool_destroy(hvx_worker_pool *pool);
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  std::printf("%-46s %s\n", what, ok ? "OK" : "FAILED");
  if (!ok)
    ++failures;
}

/* One shape: quantize codes, restrict them, pack both widths, expand. */
void roundTrip(uint32_t K, uint32_t N, unsigned seed, const char *label,
               hvx_worker_pool *pool) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> bell(0.0f, 3.0f);
  std::vector<int8_t> rm(static_cast<size_t>(K) * N);
  for (auto &v : rm) {
    int q = static_cast<int>(std::lround(bell(rng)));
    v = static_cast<int8_t>(q < -8 ? -8 : (q > 7 ? 7 : q));
  }

  int8_t pal[nntrainer::WH_PALETTE_LEVELS];
  nntrainer::whPaletteQuantize(rm.data(), K, N, /*group_k=*/0,
                               /*per_column=*/false, pal);

  std::vector<uint8_t> four(nntrainer::whBytes(K, N));
  nntrainer::whPack(rm.data(), K, N, four.data());

  std::vector<uint8_t> two(nntrainer::whBytes2(K, N));
  nntrainer::whPack2(rm.data(), K, N, pal, two.data());

  std::vector<uint8_t> table(HVX_EXPAND_TABLE_BYTES);
  hvx_expand_i2i4_table(pal, table.data());

  char what[96];
  std::snprintf(what, sizeof what, "%s half the bytes", label);
  check(two.size() * 2 == four.size(), what);

  /* The host readers (NNTR_MOE_DIFF's reference) give back the codes the
     packers were handed: whUnpack at four bits, whUnpack2 at two. */
  std::vector<int8_t> rm_back(rm.size(), 0x7F);
  nntrainer::whUnpack(four.data(), K, N, rm_back.data());
  std::snprintf(what, sizeof what, "%s whUnpack == source", label);
  check(rm_back == rm, what);
  std::fill(rm_back.begin(), rm_back.end(), int8_t{0x7F});
  nntrainer::whUnpack2(two.data(), K, N, pal, rm_back.data());
  std::snprintf(what, sizeof what, "%s whUnpack2 == source", label);
  check(rm_back == rm, what);

  std::vector<uint8_t> back(nntrainer::whBytes(K, N), 0xAA);
  hvx_expand_i2i4(two.data(), static_cast<uint32_t>(two.size()), table.data(),
                  back.data());
  std::snprintf(what, sizeof what, "%s expand == whPack", label);
  check(back == four, what);

  /* The pool split must not change a byte: workers take whole vectors, so
     a boundary that cut a shuffle in half would show up right here. */
  std::vector<uint8_t> pooled(nntrainer::whBytes(K, N), 0x55);
  hvx_expand_i2i4_pool(two.data(), static_cast<uint32_t>(two.size()),
                       table.data(), pooled.data(), pool);
  std::snprintf(what, sizeof what, "%s pooled == inline", label);
  check(pooled == back, what);
}

/* [plan 229 S2] The sidecar pair; returns the process's exit code. */
int sidecars(const char *a_path, const char *b_path) {
  auto read = [](const char *p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),
                                std::istreambuf_iterator<char>());
  };
  const std::vector<uint8_t> a = read(a_path), b = read(b_path);
  auto index = [](const std::vector<uint8_t> &f,
                  std::vector<nntrainer::FcWhEntry> &e) {
    uint32_t vc[2] = {0, 0};
    if (f.size() < 16u || std::memcmp(f.data(), nntrainer::FCWH_MAGIC, 8) != 0)
      return false;
    std::memcpy(vc, f.data() + 8, sizeof vc);
    if (vc[0] != nntrainer::FCWH_VERSION ||
        16u + vc[1] * sizeof(nntrainer::FcWhEntry) > f.size())
      return false;
    e.resize(vc[1]);
    std::memcpy(e.data(), f.data() + 16, e.size() * sizeof(e[0]));
    for (const auto &x : e)
      if (x.off + x.bytes > f.size())
        return false;
    return true;
  };
  std::vector<nntrainer::FcWhEntry> ea, eb;
  if (!index(a, ea) || !index(b, eb) || ea.size() != eb.size()) {
    std::printf("FC WH SIDECAR EXPAND: unreadable index in %s or %s\n", a_path,
                b_path);
    return 1;
  }
  uint32_t bits2 = 0, bad = 0;
  for (size_t i = 0; i < ea.size(); ++i) {
    const nntrainer::FcWhEntry *two = &ea[i], *four = &eb[i];
    const std::vector<uint8_t> *f2 = &a, *f4 = &b;
    if (two->bits != 2u) {
      std::swap(two, four);
      std::swap(f2, f4);
    }
    const uint32_t K = four->K, N = four->N;
    bool ok = std::strncmp(two->name, four->name, 64) == 0 && two->K == K &&
              two->N == N && two->key == four->key && four->bits == 4u &&
              (two->bits == 2u || two->bits == 4u);
    const uint8_t *c4 = f4->data() + four->off, *c2 = f2->data() + two->off;
    if (ok && two->bits == 2u) {
      ++bits2;
      std::vector<uint8_t> table(HVX_EXPAND_TABLE_BYTES),
        back(nntrainer::whBytes(K, N), 0xAA);
      hvx_expand_i2i4_table(
        reinterpret_cast<const int8_t *>(c2 + nntrainer::whBytes2(K, N)),
        table.data());
      hvx_expand_i2i4(c2, static_cast<uint32_t>(nntrainer::whBytes2(K, N)),
                      table.data(), back.data());
      ok = std::memcmp(back.data(), c4, back.size()) == 0;
    } else if (ok) {
      ok = std::memcmp(c2, c4, nntrainer::whBytes(K, N)) == 0;
    }
    ok = ok && std::memcmp(c2 + nntrainer::fcWhCodeBytes(K, N, two->bits),
                           c4 + nntrainer::whBytes(K, N), 8u * N) == 0;
    if (!ok)
      std::printf("FC WH SIDECAR EXPAND: %.64s differs\n", four->name);
    bad += !ok;
  }
  std::printf("FC WH SIDECAR EXPAND == WHPACK images=%zu bits2=%u bad=%u%s\n",
              ea.size(), bits2, bad, bad ? "  FAIL" : " ok");
  return bad ? 1 : 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 3)
    return sidecars(argv[1], argv[2]);
  /* Three workers plus the caller, the same shape worker_pool_host_check
     uses, so the split is exercised rather than assumed. */
  hvx_worker_pool *pool = hvx_worker_pool_create(4);

  /* The real expert shapes, and one that is not square and not a multiple
     of the vector in tiles. */
  roundTrip(2048, 3584, 1u, "gate_up 2048x3584", pool);
  roundTrip(1792, 2048, 2u, "down    1792x2048", pool);
  roundTrip(32, 32, 3u, "one tile   32x32  ", pool);
  roundTrip(96, 64, 4u, "small      96x64  ", pool);

  /* IN PLACE. The 2-bit chunk is DMA'd into the TOP half of the int4 slot
     it expands into, so the expansion needs no VTCM of its own -- which is
     what makes it fit at all (doc 54 5.6). Safe because output byte
     2j+1 only reaches source byte j, so the write pointer never passes the
     read pointer: at vector v the stores cover [256v, 256v+256) and the
     unread source starts at H + 128(v+1), and 128(v+1) <= H = src_bytes
     holds with equality at the last vector. Checked here rather than
     argued, because "with equality at the last vector" is where an
     off-by-one would live. */
  {
    const uint32_t K = 2048, N = 3584;
    std::mt19937 rng(9u);
    std::normal_distribution<float> bell(0.0f, 3.0f);
    std::vector<int8_t> rm(static_cast<size_t>(K) * N);
    for (auto &v : rm) {
      int q = static_cast<int>(std::lround(bell(rng)));
      v = static_cast<int8_t>(q < -8 ? -8 : (q > 7 ? 7 : q));
    }
    int8_t pal[nntrainer::WH_PALETTE_LEVELS];
    nntrainer::whPaletteQuantize(rm.data(), K, N, 0, false, pal);

    std::vector<uint8_t> four(nntrainer::whBytes(K, N));
    nntrainer::whPack(rm.data(), K, N, four.data());
    std::vector<uint8_t> two(nntrainer::whBytes2(K, N));
    nntrainer::whPack2(rm.data(), K, N, pal, two.data());
    std::vector<uint8_t> table(HVX_EXPAND_TABLE_BYTES);
    hvx_expand_i2i4_table(pal, table.data());

    /* One 4-bit slot with the codes sitting in its top half. */
    std::vector<uint8_t> slot(four.size(), 0xCC);
    std::copy(two.begin(), two.end(), slot.begin() + two.size());
    hvx_expand_i2i4(slot.data() + two.size(), static_cast<uint32_t>(two.size()),
                    table.data(), slot.data());
    check(slot == four, "in place (codes in the top half)");
  }

  /* A palette with duplicate entries: whPaletteFitHist pads that way when
     a group has fewer than four distinct codes, and the table has to stay
     well-formed. */
  {
    const int8_t pal[4] = {-3, -3, -3, -3};
    std::vector<uint8_t> table(HVX_EXPAND_TABLE_BYTES);
    hvx_expand_i2i4_table(pal, table.data());
    /* Asserted on the output, not on the table bytes: the entries sit at
       2*index now (hvx_expand_i2i4.c), so which byte holds what is the
       lookup's business and this check should not know. */
    std::vector<uint8_t> codes(128), out(256, 0);
    for (int i = 0; i < 128; ++i)
      codes[i] = static_cast<uint8_t>(i & 0xFF);
    hvx_expand_i2i4(codes.data(), 128, table.data(), out.data());
    bool ok = true;
    for (uint8_t b : out)
      ok = ok && b == 0xDD; /* (-3 & 0xF) in both nibbles */
    check(ok, "degenerate palette, every index");
  }

  hvx_worker_pool_destroy(pool);
  if (failures) {
    std::printf("\n%d CHECK(S) FAILED\n", failures);
    return 1;
  }
  std::printf("\nALL CHECKS PASS\n");
  return 0;
}
