// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_vtcm_cap.c
 * @date   30 Sep 2026
 * @brief  [#132 Part B, T1] Caps the VTCM HexKL acquires at open, so a
 *         second PD can take the rest through the v79 VTCM window
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Linked only into the capped skel variants: build.sh adds this file and
 * the two -Wl,--wrap flags when NNTR_VTCM_CAP_KB is set, so the default skel
 * has no __wrap_ symbol and no code from here.
 *
 * What hexkl_micro_hw_init does (libhexkl_micro.a beta.2, 6.4.0.1,
 * disassembled): compute_resource_query_VTCM(0, &total, .., &avail, ..),
 * fails unless avail == total, then set_vtcm_param_v2(attr, avail, avail,
 * avail) and acquire with HMX. It reads neither page layout. So the query
 * wrap caps both sizes (capping avail alone would make hw_init fail), and
 * the second wrap puts the minimum page back to the real total: HexKL
 * would ask for a min page of cap bytes, which is no page size, while the
 * SDK's VTCM-window case (HAP_compute_res.md) is a smaller request in one
 * whole-VTCM page -- what today's uncapped call already asks for.
 */

#ifdef NNTR_VTCM_CAP_KB

#include <HAP_compute_res.h>
#include <HAP_farf.h>

/** @brief The cap in bytes. */
#define NNTR_VTCM_CAP_BYTES ((unsigned int)(NNTR_VTCM_CAP_KB)*1024u)

/** The real total the query returned; 0 until the query ran. */
static unsigned int g_real_total;

int __real_compute_resource_query_VTCM(
  unsigned int application_id, unsigned int *total_block_size,
  compute_res_vtcm_page_t *total_block_layout, unsigned int *avail_block_size,
  compute_res_vtcm_page_t *avail_block_layout) __attribute__((weak));

int __real_compute_resource_attr_set_vtcm_param_v2(compute_res_attr_t *attr,
                                                   unsigned int vtcm_size,
                                                   unsigned int min_page_size,
                                                   unsigned int min_vtcm_size)
  __attribute__((weak));

/** @brief HexKL's VTCM query, both sizes capped at NNTR_VTCM_CAP_KB. */
int __wrap_compute_resource_query_VTCM(
  unsigned int application_id, unsigned int *total_block_size,
  compute_res_vtcm_page_t *total_block_layout, unsigned int *avail_block_size,
  compute_res_vtcm_page_t *avail_block_layout) {
  if (!__real_compute_resource_query_VTCM) {
    return HAP_COMPUTE_RES_NOT_SUPPORTED;
  }
  const int rc = __real_compute_resource_query_VTCM(
    application_id, total_block_size, total_block_layout, avail_block_size,
    avail_block_layout);
  if (rc != 0 || !total_block_size || !avail_block_size) {
    return rc;
  }
  const unsigned int total = *total_block_size, avail = *avail_block_size;
  g_real_total = total;
  if (*total_block_size > NNTR_VTCM_CAP_BYTES) {
    *total_block_size = NNTR_VTCM_CAP_BYTES;
  }
  if (*avail_block_size > NNTR_VTCM_CAP_BYTES) {
    *avail_block_size = NNTR_VTCM_CAP_BYTES;
  }
  FARF(ALWAYS, "nntr_vtcm_cap: query total=%u avail=%u -> %u / %u (cap %u KiB)",
       total, avail, *total_block_size, *avail_block_size,
       (unsigned)NNTR_VTCM_CAP_KB);
  return rc;
}

/** @brief HexKL's VTCM request; a capped request keeps the real total as
 *         its minimum page (the one-page VTCM-window request). */
int __wrap_compute_resource_attr_set_vtcm_param_v2(compute_res_attr_t *attr,
                                                   unsigned int vtcm_size,
                                                   unsigned int min_page_size,
                                                   unsigned int min_vtcm_size) {
  if (!__real_compute_resource_attr_set_vtcm_param_v2) {
    return HAP_COMPUTE_RES_NOT_SUPPORTED;
  }
  const unsigned int page = min_page_size;
  if (g_real_total > vtcm_size && min_page_size == vtcm_size) {
    min_page_size = g_real_total;
  }
  const int rc = __real_compute_resource_attr_set_vtcm_param_v2(
    attr, vtcm_size, min_page_size, min_vtcm_size);
  FARF(ALWAYS, "nntr_vtcm_cap: vtcm_param size=%u min_page=%u->%u min=%u rc=%d",
       vtcm_size, page, min_page_size, min_vtcm_size, rc);
  return rc;
}

#endif /* NNTR_VTCM_CAP_KB */
