// SPDX-License-Identifier: Apache-2.0
/**
 * @file   htp_interface.h
 * @date   01 April 2026
 * @brief  Runtime interface to libhtp_ops.so via dlopen/dlsym.
 *         Decouples nntrainer from compile-time Hexagon SDK dependency.
 * @see    https://github.com/nntrainer/nntrainer
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstddef>

#include <dynamic_library_loader.h>

namespace nntrainer {
namespace htp {

using remote_handle64 = uint64_t;

/** Function pointer types matching host/session.h signatures */
using open_dsp_session_fn_t = int(int, int);
using close_dsp_session_fn_t = void();
using get_global_handle_fn_t = remote_handle64();
using init_htp_backend_fn_t = void();
using create_htp_message_channel_fn_t = int(int, unsigned int);
using alloc_shared_mem_buf_fn_t = int(void **, int *, size_t);
using free_shared_mem_buf_fn_t = void(void *, int, size_t);

/** Function pointer types matching host/op_export.h signatures */
using htp_ops_rpc_rms_norm_f32_fn_t = int(int, int, int, int, int, int);
using htp_ops_rpc_mat_mul_permuted_w16a32_fn_t =
  int(int, int, int, int, int, int, int, int, int);

/** Function pointer types matching host/htp_ops.h signatures (with handle) */
using htp_ops_mat_mul_permuted_w16a32_fn_t =
  int(remote_handle64, int, int, int, int, int, int, int, int, int);

/**
 * @brief Singleton interface that loads libhtp_ops.so at runtime and resolves
 *        all exported symbols via dlsym. No compile-time link against
 *        libhtp_ops or Hexagon SDK is required.
 */
struct HtpInterface {
  /* session.h functions */
  open_dsp_session_fn_t *open_dsp_session = nullptr;
  close_dsp_session_fn_t *close_dsp_session = nullptr;
  get_global_handle_fn_t *get_global_handle = nullptr;
  init_htp_backend_fn_t *init_htp_backend = nullptr;
  create_htp_message_channel_fn_t *create_htp_message_channel = nullptr;
  alloc_shared_mem_buf_fn_t *alloc_shared_mem_buf = nullptr;
  free_shared_mem_buf_fn_t *free_shared_mem_buf = nullptr;

  /* op_export.h functions */
  htp_ops_rpc_rms_norm_f32_fn_t *htp_ops_rpc_rms_norm_f32 = nullptr;
  htp_ops_rpc_mat_mul_permuted_w16a32_fn_t
    *htp_ops_rpc_mat_mul_permuted_w16a32 = nullptr;

  /* htp_ops.h functions (with handle) */
  htp_ops_mat_mul_permuted_w16a32_fn_t *htp_ops_mat_mul_permuted_w16a32 =
    nullptr;

  static HtpInterface &instance() {
    static HtpInterface inst = load();
    return inst;
  }

private:
  template <typename T>
  static T *sym(void *lib, const char *name) {
    return reinterpret_cast<T *>(
      DynamicLibraryLoader::loadSymbol(lib, name));
  }

  static HtpInterface load() {
    HtpInterface iface;
    void *lib =
      DynamicLibraryLoader::loadLibrary("libhtp_ops.so", RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
      fprintf(stderr, "HtpInterface: failed to load libhtp_ops.so: %s\n",
              DynamicLibraryLoader::getLastError());
      return iface;
    }

    /* session.h */
    iface.open_dsp_session = sym<open_dsp_session_fn_t>(lib, "open_dsp_session");
    iface.close_dsp_session =
      sym<close_dsp_session_fn_t>(lib, "close_dsp_session");
    iface.get_global_handle =
      sym<get_global_handle_fn_t>(lib, "get_global_handle");
    iface.init_htp_backend =
      sym<init_htp_backend_fn_t>(lib, "init_htp_backend");
    iface.create_htp_message_channel =
      sym<create_htp_message_channel_fn_t>(lib, "create_htp_message_channel");
    iface.alloc_shared_mem_buf =
      sym<alloc_shared_mem_buf_fn_t>(lib, "alloc_shared_mem_buf");
    iface.free_shared_mem_buf =
      sym<free_shared_mem_buf_fn_t>(lib, "free_shared_mem_buf");

    /* op_export.h */
    iface.htp_ops_rpc_rms_norm_f32 =
      sym<htp_ops_rpc_rms_norm_f32_fn_t>(lib, "htp_ops_rpc_rms_norm_f32");
    iface.htp_ops_rpc_mat_mul_permuted_w16a32 =
      sym<htp_ops_rpc_mat_mul_permuted_w16a32_fn_t>(
        lib, "htp_ops_rpc_mat_mul_permuted_w16a32");

    /* htp_ops.h (with handle) */
    iface.htp_ops_mat_mul_permuted_w16a32 =
      sym<htp_ops_mat_mul_permuted_w16a32_fn_t>(
        lib, "htp_ops_mat_mul_permuted_w16a32");

    return iface;
  }
};

} // namespace htp
} // namespace nntrainer
