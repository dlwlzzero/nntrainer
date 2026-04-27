# HTP Backend (HexKL SDKL)

## Overview

The HTP (Hexagon Tensor Processor) backend offloads matrix multiplication
operations to Qualcomm's Hexagon DSP via the official HexKL SDKL CPU Macro
API. It uses pre-built libraries from the Hexagon SDK's `hexkl_addon`:

| Library | Runs on | Description |
|---------|---------|-------------|
| `libsdkl.so` | Host (Android CPU) | SDKL CPU Macro API (FastRPC + session internally) |
| `libhexkl_skel.so` | Hexagon DSP | Pre-built HMX/HVX compute kernels |

For build and test instructions, see [How to Use HTP Backend](../../../docs/how-to-use-htp-backend.md).

## Architecture

```
    Android CPU (Host)                          Hexagon DSP
 +----------------------------+              +---------------------+
 |                            |              |                     |
 |  nntrainer                 |   FastRPC    | libhexkl_skel.so    |
 |    |                       |  (internal)  | (pre-built)         |
 |    v                       |              |                     |
 |  sdkl_interface.h          |              |  HMX matmul kernels |
 |  [dlopen singleton]        |              |  HVX layout/convert |
 |    |                       |              |                     |
 |    v                       |              +---------------------+
 |  libsdkl.so (Qualcomm)     |
 |    sdkl_npu_mm_f32f16_f32  ===============>
 |    sdkl_npu_alloc/free     |
 |    sdkl_cpu_rm_to_wh_f16   |
 +----------------------------+
```

## Key Files

- `sdkl_interface.h` — Runtime dlopen interface to `libsdkl.so`
- `meson.build` — Build configuration (installs pre-built HexKL libraries)

## Runtime Data Flow

1. **Weight caching**: On first matmul call for a given FP16 weight, the
   weight is allocated via `sdkl_npu_alloc`, copied, and converted to WH
   layout via `sdkl_cpu_rm_to_wh_f16_inplace`. The WH buffer is cached
   and reused on subsequent calls.

2. **Matmul dispatch**: `sdkl_npu_mm_f32f16_f32(domain, M, N, K, out, act, wt)`
   accepts FP32 activation/output as ordinary pointers and FP16 WH-layout
   weight from the cache.

## Supported Operations

| Operation | SDKL Function | Input Types |
|-----------|---------------|-------------|
| Mixed-precision matmul | `sdkl_npu_mm_f32f16_f32` | FP32 act × FP16 weight → FP32 out |
