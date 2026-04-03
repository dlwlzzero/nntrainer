# How to Build and Test HTP Backend

## Overview

The HTP (Hexagon Tensor Processor) backend offloads tensor operations
to Qualcomm's Hexagon DSP
using HMX and HVX hardware accelerators.

The build produces two shared libraries:

| Library | Runs on | Description |
|---------|---------|-------------|
| `libhtp_ops.so` | Host (Android CPU) | FastRPC stub + session management |
| `libhtp_ops_skel.so` | Hexagon DSP | HMX/HVX compute kernels |

## Prerequisites

- nntrainer basic setup must be completed first. See [Getting Started](getting-started.md).
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
├── libhtp_ops.so        # Host stub library
├── libhtp_ops_skel.so   # DSP skel library
└── htp_ops_test         # Test binary
```

### Standalone cmake build (without meson)

```bash
export HEXAGON_SDK_HOME=/path/to/hexagon/sdk

cd nntrainer/nntrainer/tensor/htp_backend
./build_htp.sh
```

After a successful build, the output files are located at:

```
nntrainer/nntrainer/tensor/htp_backend/build_htp/
├── libhtp_ops.so        # Host stub library
├── libhtp_ops_skel.so   # DSP skel library
└── htp_ops_test         # Test binary
```

### DSP architecture setting

The target device's DSP architecture must be specified via the `DSP_ARCH`
option inside `build_htp.sh`. Set it to `v73` or `v75` depending on the
Android device before running the build.

## Running Unit Tests

Unit tests for the HTP backend are located under `test/unittest/`.
Before running the tests, `libhtp_ops.so` and `libhtp_ops_skel.so` must
already be built via the [Full build](#full-build-with-meson) or
[Standalone cmake build](#standalone-cmake-build-without-meson) steps above.

```bash
# 1. Build and push nntrainer test binaries to device
$ ./tools/android_test.sh

# 2. Push HTP libraries to the test directory on device
$ adb push /path/to/libhtp_ops.so /data/local/tmp/nntr_android_test
$ adb push /path/to/libhtp_ops_skel.so /data/local/tmp/nntr_android_test

# 3. Run unittest on device
$ adb shell
(adb) $ cd /data/local/tmp/nntr_android_test
(adb) $ export LD_LIBRARY_PATH=.
(adb) $ export DSP_LIBRARY_PATH=.
(adb) $ ./<unittest_name>
```
