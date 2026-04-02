---
title: How to Build and Test HTP Backend
...

# How to Build and Test HTP Backend

## Overview

The HTP (Hexagon Tensor Processor) backend offloads tensor operations
(matrix multiply, RMS norm, flash attention) to Qualcomm's Hexagon DSP
using HMX and HVX hardware accelerators.

The build produces two shared libraries:

| Library | Runs on | Description |
|---------|---------|-------------|
| `libhtp_ops.so` | Host (Android CPU) | FastRPC stub + session management |
| `libhtp_ops_skel.so` | Hexagon DSP | HMX/HVX compute kernels |

## Prerequisites

- **Hexagon SDK >= v6.0.0.2** with setup complete
- **CMake >= 3.14.3**
- `HEXAGON_SDK_HOME` environment variable pointing to SDK root

## Build Instructions

### Full build (with meson)

```bash
export HEXAGON_SDK_HOME=/path/to/hexagon/sdk

cd nntrainer
meson setup build -Dwerror=false -Denable-htp=true
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

## Running Unit Tests

### On-device test binary

The build produces `htp_ops_test` in the build output directory. To run it on
an Android device:

```bash
# Push test binary and libraries to device
adb push build/nntrainer/tensor/htp_backend/htp_lib/htp_ops_test /data/local/tmp/
adb push build/nntrainer/tensor/htp_backend/htp_lib/libhtp_ops.so /data/local/tmp/

# Run on device
adb shell
cd /data/local/tmp
export LD_LIBRARY_PATH=.
./htp_ops_test
```

### On-device op validation

The DSP-side library includes built-in op validation tests (`htp/op_tests.cc`).
These tests run directly on the Hexagon DSP and validate correctness of HMX/HVX
operations against reference implementations.
