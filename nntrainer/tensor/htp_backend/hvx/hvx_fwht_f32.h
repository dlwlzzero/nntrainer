// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_fwht_f32.h
 * @date   22 Sep 2026
 * @brief  In-place block-256 Walsh-Hadamard rotation of f32 rows (HVX)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_FWHT_F32_H__
#define __NNTRAINER_HVX_FWHT_F32_H__

#include <stdint.h>

#include "fwht_det.h"
#include "hvx_worker_pool.h"

/** @brief The rotation block, in floats: fwht_det.h's constant, which is
 *         also the converter's fold block and its K % 256 refusal. */
#define HVX_FWHT_BLOCK FWHT_DET_BLOCK

/**
 * @brief x[r][0..k) -> x[r]*H/16 per 256-float block, every row in place;
 *        bit-identical to fwht_rows_f32_ref (fwht_det.h).
 *
 * IEEE sf add/sub only, one per butterfly output, stages in ascending
 * order, then one exact multiply by 1/16 -- the specification in
 * fwht_det.h, which is what the MoE kernel's u8 requantization then sees.
 * No qf32 anywhere in the implementation (v75 and v79 disagree on
 * qf32 -> sf).
 *
 * Rows are independent, so @a pool splits by row range; NULL runs
 * single-threaded. @a k must be a multiple of HVX_FWHT_BLOCK -- the MoE
 * layer entry refuses the flag otherwise; any trailing partial block here
 * is left untouched, exactly as the reference leaves it.
 *
 * @param[in,out] x     rows x k floats, row-major, unaligned is fine
 * @param[in]     rows  rows to rotate
 * @param[in]     k     floats per row
 */
void hvx_fwht_rows_f32(float *x, uint32_t rows, uint32_t k,
                       hvx_worker_pool *pool);

#endif /* __NNTRAINER_HVX_FWHT_F32_H__ */
