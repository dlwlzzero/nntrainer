# HTP Backend Build Guide

## Overview

The HTP (Hexagon Tensor Processor) backend offloads tensor operations to
Qualcomm's Hexagon DSP. The build produces two shared libraries:

| Library | Runs on | Description |
|---------|---------|-------------|
| `libhtp_ops.so` | Host (Android CPU) | FastRPC stub + session management |
| `libhtp_ops_skel.so` | Hexagon DSP | HMX/HVX compute kernels |

## Directory Structure

```
htp_backend/
├── host/                        # Host-side source (compiled into libhtp_ops.so)
│   ├── session.c                #   DSP session lifecycle (open/close/init)
│   ├── op_export.c              #   RPC wrapper functions for nntrainer
│   └── test.c                   #   Standalone test binary
├── htp/                         # DSP-side source (compiled into libhtp_ops_skel.so)
│   ├── commu.c                  #   FastRPC skel entry points
│   ├── hmx_mgr.c               #   HMX hardware manager
│   ├── power.c                  #   DSP power/clock management
│   ├── worker_pool.c            #   Multi-threaded worker pool
│   ├── mmap_mgr.cc             #   Shared memory mapping manager
│   ├── op_executor.cc          #   Op dispatch and execution
│   ├── op_tests.cc             #   On-device op validation
│   ├── vtcm_mgr.cc            #   VTCM (tightly coupled memory) manager
│   ├── hmx_ops/                #   HMX accelerated ops
│   │   ├── flash_attn.c        #     Flash attention
│   │   ├── flash_attn_sp_hdim.c#     Flash attention (special head dim)
│   │   └── mat_mul.c           #     Matrix multiplication
│   └── hvx_ops/                #   HVX vector ops
│       ├── rms_norm.c          #     RMS normalization
│       ├── precompute_table.c  #     Lookup table precomputation
│       └── mm_benchmark.c      #     Matrix multiply benchmark
├── include/
│   ├── host/                    # Host-side headers
│   │   ├── session.h           #   Session API (included by nntrainer)
│   │   ├── op_export.h         #   RPC op wrappers
│   │   ├── htp_ops.h           #   QAIC-generated FastRPC header
│   │   ├── htp_ops_stub.c      #   QAIC-generated FastRPC stub
│   │   └── htp_ops_skel.c      #   QAIC-generated FastRPC skel
│   ├── htp/                     # DSP-side headers
│   │   ├── hmx_mgr.h, hmx_utils.h, dma_utils.h
│   │   ├── hvx_convert.h, hvx_internal.h, hvx_math.h
│   │   ├── op_executor.h, ops.h, quants.h, utils.h
│   │   ├── mmap_mgr.h, vtcm_mgr.h, worker_pool.h, power.h
│   │   └── ...
│   ├── htp_ops.idl              # FastRPC IDL definition
│   ├── message.h                # Host-DSP message protocol
│   └── op_reg.h                 # Op registration
├── CMakeLists.txt               # Hexagon SDK cmake build
├── build_htp.sh                 # Build script invoked by meson
└── meson.build                  # Meson integration
```

## Prerequisites

- **Hexagon SDK >= v6.0.0.2** with setup complete
- **CMake >= 3.14.3**
- `HEXAGON_SDK_HOME` environment variable pointing to SDK root

## Build System Architecture

The HTP backend uses a **two-layer build system**:

### Layer 1: Meson (nntrainer integration)

When `enable-htp=true`:

1. **Configure phase** (`meson setup`):
   - Reads `HEXAGON_SDK_HOME` environment variable
   - Adds Hexagon SDK include directories to `nntrainer_inc`
   - Adds `-DENABLE_HTP=1` compiler define (in root `meson.build`)
   - Registers `custom_target` to invoke `build_htp.sh`

2. **Compile phase** (`ninja`):
   - `float_tensor.cpp` compiles with `ENABLE_HTP=1`, includes `host/session.h`
   - `session.h` contains **declarations only** (no SDK dependencies)
   - `libnntrainer.so` links without HTP library dependencies

### Layer 2: CMake via build_htp.sh (target device libraries)

The `custom_target` invokes `build_htp.sh` which:

1. Sources `${HEXAGON_SDK_HOME}/setup_sdk_env.source`
2. Runs `build_cmake android` -> builds host-side `libhtp_ops.so`
3. Runs `build_cmake hexagon DSP_ARCH=v75` -> builds DSP-side `libhtp_ops_skel.so`
4. Copies built libraries to `${BUILD_DIR}/htp_lib/`
5. Copies QAIC-generated files to `include/host/`

### Why two build systems?

- **DSP code** requires the Hexagon cross-compiler toolchain (only available
  through the SDK's cmake integration via `hexagon_fun.cmake`)
- **Host stub** (`htp_ops_stub.c`) links against `libcdsprpc.so` which only
  exists on the target device
- **nntrainer** must compile on the build host without target-only libraries

This is solved by keeping `session.h` free of SDK dependencies (declarations
only). Function definitions live in `session.c` and are compiled by cmake
into `libhtp_ops.so` for the target device.

## Build Instructions

### Full build (with HTP)

```bash
export HEXAGON_SDK_HOME=/path/to/hexagon/sdk

cd nntrainer
meson setup build -Denable-htp=true
ninja -C build
```

### Build output

After a successful build:

```
build/nntrainer/tensor/htp_backend/htp_lib/
├── libhtp_ops.so        # Host stub library (deploy to /vendor/lib64/)
├── libhtp_ops_skel.so   # DSP skel library (deploy to /vendor/lib/rfsa/dsp/sdsp/)
└── htp_ops_test          # Test binary
```

### Standalone cmake build (without meson)

```bash
export HEXAGON_SDK_HOME=/path/to/hexagon/sdk
source ${HEXAGON_SDK_HOME}/setup_sdk_env.source

cd nntrainer/nntrainer/tensor/htp_backend
build_cmake android
build_cmake hexagon DSP_ARCH=v75
```

## Key Design Decisions

### session.h: declarations only

`session.h` is included by `float_tensor.cpp` (C++) but must not pull in
Hexagon SDK headers (`remote.h`, `rpcmem.h`). Therefore:

- All function bodies are in `session.c` (compiled by cmake, not meson)
- `remote_handle64` is typedef'd with an include guard to avoid conflicts
- `extern "C"` wrapping ensures correct C++ linkage

### Meson options

| Option | Default | Effect |
|--------|---------|--------|
| `enable-htp` | `false` | Adds `-DENABLE_HTP=1`, includes HTP paths, builds HTP libs |

### Runtime dependencies

On the target device, `libhtp_ops.so` requires:
- `libcdsprpc.so` (Qualcomm FastRPC runtime)
- `librpcmem.so` (shared memory allocation)
