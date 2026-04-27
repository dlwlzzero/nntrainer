// SPDX-License-Identifier: Apache-2.0
/**
 * @file   sdkl_interface.h
 * @date   27 April 2026
 * @brief  Runtime interface to Qualcomm HexKL SDKL (libsdkl.so) via
 *         dlopen/dlsym.  Replaces htp_interface.h by delegating matmul to the
 *         official HexKL CPU Macro API instead of the custom HTP backend.
 * @see    https://github.com/nntrainer/nntrainer
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <dynamic_library_loader.h>

namespace nntrainer {
namespace sdkl {

/* ---- Function-pointer types matching sdkl.h signatures ---- */

/* General lifecycle */
using sdkl_npu_initialize_fn_t = int(int, const void *, void *);
using sdkl_npu_finalize_fn_t = int(int);

/* Memory management */
using sdkl_npu_alloc_fn_t = int(size_t, void **);
using sdkl_npu_free_fn_t = int(void *);

/* HMX lock / unlock */
using sdkl_npu_lock_hmx_fn_t = int(int);
using sdkl_npu_unlock_hmx_fn_t = int(int);

/* Matrix multiplication — FP32 activation × FP16 weight → FP32 output */
using sdkl_npu_mm_f32f16_f32_fn_t = int(int, int, int, int, float *,
                                         const float *, const _FP16 *);

/* Data-layout transforms (CPU-side, in-place) */
using sdkl_cpu_rm_to_wh_f16_inplace_fn_t = int(size_t, size_t, _FP16 *);

/**
 * @brief Singleton interface that loads libsdkl.so at runtime and resolves
 *        the SDKL CPU Macro API symbols via dlsym.  No compile-time link
 *        against libsdkl or the Hexagon SDK is required.
 */
struct SdklInterface {

  /* General */
  sdkl_npu_initialize_fn_t *npu_initialize = nullptr;
  sdkl_npu_finalize_fn_t *npu_finalize = nullptr;

  /* Memory */
  sdkl_npu_alloc_fn_t *npu_alloc = nullptr;
  sdkl_npu_free_fn_t *npu_free = nullptr;

  /* HMX exclusive lock */
  sdkl_npu_lock_hmx_fn_t *npu_lock_hmx = nullptr;
  sdkl_npu_unlock_hmx_fn_t *npu_unlock_hmx = nullptr;

  /* MatMul */
  sdkl_npu_mm_f32f16_f32_fn_t *npu_mm_f32f16_f32 = nullptr;

  /* Layout */
  sdkl_cpu_rm_to_wh_f16_inplace_fn_t *cpu_rm_to_wh_f16_inplace = nullptr;

  /** Whether the library was loaded and the critical symbols resolved. */
  bool is_available() const {
    return npu_initialize && npu_finalize && npu_alloc && npu_free &&
           npu_mm_f32f16_f32 && cpu_rm_to_wh_f16_inplace;
  }

  /** Domain ID cached from initialization (-1 if not yet initialized). */
  int domain = -1;

  /** Whether sdkl_npu_initialize has been called successfully. */
  bool initialized = false;

  /**
   * @brief Ensure the SDKL library is initialized for the given domain.
   *        Safe to call multiple times; only the first call performs init.
   * @return 0 on success, non-zero on failure.
   */
  int ensure_initialized(int cdsp_domain) {
    if (initialized)
      return 0;
    if (!npu_initialize)
      return -1;
    int err = npu_initialize(cdsp_domain, nullptr, nullptr);
    if (err == 0) {
      domain = cdsp_domain;
      initialized = true;
    } else {
      fprintf(stderr, "SdklInterface: sdkl_npu_initialize failed: %d\n", err);
    }
    return err;
  }

  static SdklInterface &instance() {
    static SdklInterface inst = load();
    return inst;
  }

private:
  template <typename T> static T *sym(void *lib, const char *name) {
    return reinterpret_cast<T *>(DynamicLibraryLoader::loadSymbol(lib, name));
  }

  static SdklInterface load() {
    SdklInterface iface;
    void *lib = DynamicLibraryLoader::loadLibrary("libsdkl.so",
                                                   RTLD_LAZY | RTLD_LOCAL);
    if (!lib) {
      fprintf(stderr, "SdklInterface: failed to load libsdkl.so: %s\n",
              DynamicLibraryLoader::getLastError());
      return iface;
    }

    /* General lifecycle */
    iface.npu_initialize =
      sym<sdkl_npu_initialize_fn_t>(lib, "sdkl_npu_initialize");
    iface.npu_finalize =
      sym<sdkl_npu_finalize_fn_t>(lib, "sdkl_npu_finalize");

    /* Memory */
    iface.npu_alloc = sym<sdkl_npu_alloc_fn_t>(lib, "sdkl_npu_alloc");
    iface.npu_free = sym<sdkl_npu_free_fn_t>(lib, "sdkl_npu_free");

    /* HMX lock */
    iface.npu_lock_hmx =
      sym<sdkl_npu_lock_hmx_fn_t>(lib, "sdkl_npu_lock_hmx");
    iface.npu_unlock_hmx =
      sym<sdkl_npu_unlock_hmx_fn_t>(lib, "sdkl_npu_unlock_hmx");

    /* MatMul */
    iface.npu_mm_f32f16_f32 =
      sym<sdkl_npu_mm_f32f16_f32_fn_t>(lib, "sdkl_npu_mm_f32f16_f32");

    /* Layout transforms */
    iface.cpu_rm_to_wh_f16_inplace =
      sym<sdkl_cpu_rm_to_wh_f16_inplace_fn_t>(lib,
                                                "sdkl_cpu_rm_to_wh_f16_inplace");

    return iface;
  }
};

} // namespace sdkl
} // namespace nntrainer
