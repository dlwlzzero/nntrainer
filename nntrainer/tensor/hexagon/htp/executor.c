// SPDX-License-Identifier: Apache-2.0
/**
 * @file	executor.c
 * @date	15 August 2026
 * @brief	DSP-side op-list executor. M1: validates the op-list header,
 *		persistently maps the init-time dma-buf fds, and answers
 *		forward() with a deterministic dummy pattern.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <stdlib.h>

#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_mem.h"

#include "nntr_htp.h"
#include "nntr_htp_common.h"

struct session {
  void *weights;
  void *kv;
  void *act;
  uint32 weights_size;
  uint32 kv_size;
  uint32 act_size;
};

static void unmap_all(struct session *s) {
  if (s->weights)
    HAP_munmap(s->weights, (int)s->weights_size);
  if (s->kv)
    HAP_munmap(s->kv, (int)s->kv_size);
  if (s->act)
    HAP_munmap(s->act, (int)s->act_size);
  s->weights = s->kv = s->act = 0;
}

static void *map_fd(int32 fd, uint32 size) {
  void *p = HAP_mmap(0, (int)size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
  return (p == (void *)-1) ? 0 : p;
}

AEEResult nntr_htp_open(const char *uri, remote_handle64 *h) {
  struct session *s = calloc(1, sizeof(*s));
  (void)uri;
  if (!s)
    return AEE_ENOMEMORY;
  *h = (remote_handle64)(uintptr_t)s;
  FARF(ALWAYS, "nntr_htp: open, abi v%u", NNTR_HTP_ABI_VERSION);
  return AEE_SUCCESS;
}

AEEResult nntr_htp_close(remote_handle64 h) {
  struct session *s = (struct session *)(uintptr_t)h;
  unmap_all(s);
  free(s);
  FARF(ALWAYS, "nntr_htp: close");
  return AEE_SUCCESS;
}

AEEResult nntr_htp_init(remote_handle64 h, const uint8 *oplist, int oplistLen,
                        const uint8 *weights, int weightsLen,
                        int32 weights_fd, int32 kv_fd, uint32 kv_size,
                        int32 act_fd, uint32 act_size,
                        uint32 *dsp_abi_version) {
  struct session *s = (struct session *)(uintptr_t)h;
  int rc;

  (void)weights; /* in-sequence only forces the driver cache flush */
  *dsp_abi_version = NNTR_HTP_ABI_VERSION;
  if (oplistLen < 0 || weightsLen < (int)sizeof(int32_t))
    return AEE_EBADPARM;
  rc = nntr_htp_oplist_check(oplist, (uint32)oplistLen);
  if (rc != 0) {
    FARF(ERROR, "nntr_htp: op-list rejected (rc=%d)", rc);
    return rc == 3 ? AEE_EUNSUPPORTED : AEE_EBADPARM;
  }

  unmap_all(s); /* re-init replaces any previous mapping */
  s->weights_size = (uint32)weightsLen;
  s->kv_size = kv_size;
  s->act_size = act_size;
  s->weights = map_fd(weights_fd, s->weights_size);
  s->kv = map_fd(kv_fd, kv_size);
  s->act = map_fd(act_fd, act_size);
  if (!s->weights || !s->kv || !s->act) {
    FARF(ERROR, "nntr_htp: HAP_mmap failed (w=%p kv=%p act=%p)", s->weights,
         s->kv, s->act);
    unmap_all(s);
    return AEE_ENOMEMORY;
  }
  FARF(ALWAYS, "nntr_htp: init ok, weights=%d kv=%u act=%u", weightsLen,
       kv_size, act_size);
  return AEE_SUCCESS;
}

AEEResult nntr_htp_forward(remote_handle64 h, const int32 *token_ids,
                           int token_idsLen, uint32 pos, float *logits,
                           int logitsLen) {
  struct session *s = (struct session *)(uintptr_t)h;
  int32_t w0;
  int i;

  if (!s->weights)
    return AEE_EBADSTATE;
  if (token_idsLen <= 0 || logitsLen <= 0)
    return AEE_EBADPARM;
  w0 = ((const int32_t *)s->weights)[0];

  // Dummy pattern - must match test/hexagon/hexagon_rpc_test.cpp:
  // logits[i] = token_ids[i % n] + pos + i + weights[0]
  for (i = 0; i < logitsLen; ++i)
    logits[i] = (float)(token_ids[i % token_idsLen] + (int32_t)pos + i + w0);
  return AEE_SUCCESS;
}
