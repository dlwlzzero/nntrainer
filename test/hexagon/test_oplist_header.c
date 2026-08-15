// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_oplist_header.c
 * @date	15 August 2026
 * @brief	x86 self-check for the op-list header validation shared by host and DSP.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <assert.h>
#include <stdio.h>

#include "../../nntrainer/tensor/hexagon/htp/nntr_htp_common.h"

int main(void) {
  struct nntr_htp_oplist_header h = {NNTR_HTP_OPLIST_MAGIC,
                                     NNTR_HTP_ABI_VERSION, 0, 0};
  assert(nntr_htp_oplist_check(&h, sizeof(h)) == 0);

  h.version = 999u;
  assert(nntr_htp_oplist_check(&h, sizeof(h)) == 3);
  h.version = NNTR_HTP_ABI_VERSION;

  h.magic = 0;
  assert(nntr_htp_oplist_check(&h, sizeof(h)) == 2);
  h.magic = NNTR_HTP_OPLIST_MAGIC;

  assert(nntr_htp_oplist_check(&h, 3) == 1);
  assert(nntr_htp_oplist_check(0, sizeof(h)) == 1);

  puts("oplist header check: PASS");
  return 0;
}
