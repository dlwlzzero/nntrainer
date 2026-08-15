// SPDX-License-Identifier: Apache-2.0
/**
 * @file	nntr_htp_common.h
 * @date	15 August 2026
 * @brief	Op-list wire format shared by host (arm64) and DSP
 *		(hexagon-clang). Plain C - compiled by both toolchains.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef NNTR_HTP_COMMON_H
#define NNTR_HTP_COMMON_H

#include <stdint.h>
#include <string.h>

#define NNTR_HTP_OPLIST_MAGIC 0x5054484Eu /* "NHTP" little-endian */
#define NNTR_HTP_ABI_VERSION 1u

/**
 * @brief Leading header of every op-list buffer sent from host to DSP.
 */
struct nntr_htp_oplist_header {
  uint32_t magic;    /**< NNTR_HTP_OPLIST_MAGIC; rejects foreign buffers */
  uint32_t version;  /**< must equal NNTR_HTP_ABI_VERSION on both sides */
  uint32_t n_ops;    /**< number of op entries following this header */
  uint32_t reserved; /**< keep 0; pads the header to 16 bytes */
};

/* The struct is the wire ABI: all three toolchains must agree on 16 bytes. */
typedef char nntr_htp_oplist_header_size_check
  [(sizeof(struct nntr_htp_oplist_header) == 16) ? 1 : -1];

/**
 * @brief Validate an op-list buffer header.
 * @return 0 ok, 1 bad pointer/size, 2 bad magic, 3 version mismatch
 */
static inline int nntr_htp_oplist_check(const void *buf, uint32_t len) {
  struct nntr_htp_oplist_header h;
  if (buf == 0 || len < (uint32_t)sizeof(h))
    return 1;
  memcpy(&h, buf, sizeof(h));
  if (h.magic != NNTR_HTP_OPLIST_MAGIC)
    return 2;
  if (h.version != NNTR_HTP_ABI_VERSION)
    return 3;
  return 0;
}

#endif /* NNTR_HTP_COMMON_H */
