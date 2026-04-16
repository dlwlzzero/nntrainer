# htp backend

## Overview

The HTP (Hexagon Tensor Processor) backend offloads tensor operations to
Qualcomm's Hexagon DSP using HMX (matrix accelerator) and HVX (vector SIMD)
hardware units. The build produces two shared libraries: `libhtp_ops.so`
(host-side FastRPC stub) and `libhtp_ops_skel.so` (DSP-side compute kernels).

For build and test instructions, see [How to Use HTP Backend](../../../docs/how-to-use-htp-backend.md).

## Architecture

### Host-DSP communication

The host application loads `libhtp_ops.so` at runtime via `htp_interface.h`
(dlopen singleton). The host library communicates with `libhtp_ops_skel.so`
on the DSP through two paths:

```
    Android CPU (Host)                             Hexagon DSP
 +------------------------------+               +------------------------------+
 |                              |    FastRPC     |                              |
 |  nntrainer                   |   (direct)     |  libhtp_ops_skel.so          |
 |    |                         |                |                              |
 |    v                         |  htp_ops.idl   |                              |
 |  htp_interface.h             |  (QAIC gen)    |                              |
 |  [dlopen singleton]          |====[stub]=========>[skeleton]                 |
 |    |                         |                |       |                      |
 |    v                         |                |       v                      |
 |  libhtp_ops.so               |                |    commu.c                   |
 |  +------------------------+  |                |       |                      |
 |  | session.c              |  |                |       v                      |
 |  |   (open/close DSP)     |  |                |    op_executor.cc            |
 |  | op_export.c            |  |                |    [dispatch]                |
 |  |   (inject handle)      |  |                |      /       \              |
 |  +------------------------+  |                |     v         v             |
 |                              |  msg channel   |  hvx_ops/  hmx_ops/         |
 |  [OpComputeRequest] -------->==[rpcmem]======>|  (vector)  (matrix)         |
 |                              |                |                              |
 +------------------------------+               +------------------------------+
                |                                               |
                +------------ shared rpcmem (DDR) -------------+
```

- **FastRPC direct calls** -- auto-generated from `htp_ops.idl` by the QAIC
  code generator. Used for session management and individual op invocations.
- **Message channel** -- the host writes structured `OpComputeRequest` messages
  to shared rpcmem, and the DSP-side receiver loop in `commu.c` polls and
  dispatches them. Used for batched or asynchronous operation submission.

### DSP-side execution

```
  OpComputeRequest
        |
        v
  op_executor.cc -------> op_reg.h (HTP_OPS_* enum, param structs)
        |
        +-----------------+-----------------+
        |                 |                 |
        v                 v                 v
    hvx_ops/          hmx_ops/          hmx_ops/
    rms_norm.c        mat_mul.c         flash_attn.c
    (HVX SIMD)        (HMX tiles)       (HMX + HVX)
        |                 |                 |
        v                 v                 v
  +-----------------------------------------------------+
  |  worker_pool.c  (up to 6 threads)                   |
  +-----------------------------------------------------+
        |                 |                 |
        v                 v                 v
  +-----------------------------------------------------+
  |  VTCM (on-chip scratchpad)         vtcm_mgr.cc      |
  +-----------------------------------------------------+
        |                 |                 |
        v                 v                 v
  +-----------------------------------------------------+
  |  DDR (shared rpcmem)               mmap_mgr.cc      |
  +-----------------------------------------------------+
```

`op_executor.cc` routes each request to the appropriate HVX kernel
(`hvx_ops/`) or HMX kernel (`hmx_ops/`). The worker pool provides up to
6 parallel threads, and VTCM is managed by `vtcm_mgr.cc` for fast data
staging between DDR and the compute units.

## Basic guidelines for developers

1. **Adding a new HVX operation**: implement under `htp/hvx_ops/`, add an entry
   to `HtpOpsIndex` in `include/op_reg.h` with its parameter struct, and add a
   dispatch case in `htp/op_executor.cc`.
2. **Adding a new HMX operation**: same registration steps, but implement under
   `htp/hmx_ops/`. HMX kernels operate on 32x32 fp16 tiles and require VTCM
   staging -- refer to `hmx_ops/mat_mul.c` for the pattern.
3. **Exposing an op to the host**: add an RPC method in `include/htp_ops.idl`,
   add a wrapper in `host/op_export.c` that injects the global session handle,
   and register the corresponding function pointer type in `htp_interface.h`.
4. For build and test workflows, see
   [How to Use HTP Backend](../../../docs/how-to-use-htp-backend.md).

## Directory structure

```
htp_backend
├── htp_interface.h        : runtime dlopen loader (singleton) -- resolves libhtp_ops.so symbols at runtime
├── CMakeLists.txt         : cross-compile build for Android host (HLOS) and Hexagon DSP targets
├── meson.build            : integration with nntrainer meson build system
├── build_htp.sh           : standalone build script (invokes CMake for android + hexagon)
├── run.sh                 : push build artifacts to device and run tests via adb
│
├── host/                  [runs on Android CPU -- compiled into libhtp_ops.so]
│   ├── session.c          : FastRPC session lifecycle, rpcmem alloc/free, DSP connection
│   ├── op_export.c        : op wrappers that inject global session handle into RPC calls
│   └── test.c             : host-side standalone test (RMS norm, MatMul correctness checks)
│
├── htp/                   [runs on Hexagon DSP -- compiled into libhtp_ops_skel.so]
│   ├── commu.c            : FastRPC skeleton entry points, message channel receiver loop
│   ├── op_executor.cc     : central dispatch -- routes OpComputeRequest to HVX/HMX kernels
│   ├── worker_pool.c      : thread pool (up to 6 workers) for parallel kernel execution
│   ├── vtcm_mgr.cc        : VTCM (on-chip scratchpad) allocator
│   ├── hmx_mgr.c          : HMX resource acquisition and spin-lock synchronization
│   ├── mmap_mgr.cc        : caches HAP_mmap fd-to-va mappings
│   ├── power.c            : DCVS power and clock configuration
│   ├── op_tests.cc        : on-DSP micro-benchmarks and validation utilities
│   ├── hvx_ops/           (HVX vector kernels)
│   │   ├── rms_norm.c     : RMS normalization (f32)
│   │   ├── precompute_table.c : exp2 lookup table for softmax in flash attention
│   │   └── mm_benchmark.c : HVX MatMul benchmarks (fp16, fp32, int16, int32)
│   └── hmx_ops/           (HMX matrix kernels)
│       ├── mat_mul.c      : MatMul with f16 and quantized weights (Q4_0, Q8_0, IQ4_NL)
│       ├── flash_attn.c   : flash attention with grouped-query attention (GQA) support
│       └── flash_attn_sp_hdim.c : flash attention variant (single-parallel head dim)
│
└── include/
    ├── message.h          : message protocol structs (MessageHeader, RequestHeader, OpComputeRequest)
    ├── op_reg.h           : operation enum (HTP_OPS_*) and parameter structs
    ├── htp_ops.idl        : FastRPC IDL interface definition (input to QAIC code generator)
    ├── host/
    │   ├── session.h      : host session and memory management declarations
    │   └── op_export.h    : host op wrapper declarations
    └── htp/
        ├── ops.h          : all public HTP operation declarations
        ├── op_executor.h  : execute_op_simple() declaration
        ├── worker_pool.h  : worker pool API (init, submit, sync tokens, atomics)
        ├── vtcm_mgr.h     : VTCM allocator API
        ├── hmx_mgr.h      : HMX manager API (setup, lock/unlock)
        ├── power.h        : power setup/reset
        ├── mmap_mgr.h     : mmap cache API
        ├── utils.h        : utility macros (ceil_div, align_up, etc.)
        ├── quants.h       : quantization block structs and ggml_type enum
        ├── dma_utils.h    : DMA descriptor structures and helpers
        ├── hvx_internal.h : HVX constants (VLEN=128), l2fetch, variable-length stores
        ├── hvx_math.h     : HVX math intrinsics (exp2, log2 polynomial approximations)
        ├── hvx_convert.h  : HVX type conversions (fp16/fp32/qf16/wsf)
        └── hmx_utils.h    : HMX tile constants (32x32) and helpers
```
