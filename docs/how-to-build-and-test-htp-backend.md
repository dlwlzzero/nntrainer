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
- **CMake >= 3.14.3**
- **Hexagon SDK >= v6.0.0.2** with setup complete
- `HEXAGON_SDK_HOME` environment variable pointing to SDK root
  ```bash
  export HEXAGON_SDK_HOME=/path/to/hexagon/sdk
  ```
- The target device's DSP architecture must be specified via the `DSP_ARCH`
  option inside `build_htp.sh`. Set it to `v73` or `v75` depending on the
  Android device before running the build.
  ```
  # use can choose v73, v75, v79
  build_cmake hexagon DSP_ARCH=<dsp_ver>
  ```

## Build Instructions

Hexagon DSP is only available on Android devices, so the build must target Android.
There are three build methods depending on the use case:

| Method | Script | Use case |
|--------|--------|----------|
| [Android build (with meson)](#android-build-with-meson) | `package_android.sh` | Application + HTP backend |
| [Standalone cmake build](#standalone-cmake-build-without-meson) | `build_htp.sh` | HTP kernel development and standalone testing |
| [Unit test build](#unit-test-build) | `android_test.sh` | Running HTP unit tests |

### Android build (with meson)

This method produces both `libhtp_ops.so` and `libhtp_ops_skel.so`.
Use this when building applications that depend on the HTP backend.
For a run example, see the [CausalLM README](../Applications/CausalLM/README.md).

```bash
cd nntrainer
./tools/package_android.sh -Dwerror=false -Dmmap-read=false -Denable-htp=true
```

Build output:

```
builddir/nntrainer/tensor/htp_backend/
├── libhtp_ops.so        # Host stub library
└── libhtp_ops_skel.so   # DSP skel library
```

### Standalone cmake build (without meson)

This method builds only the HTP backend and produces the `htp_ops_test` binary
for on-device testing via `run.sh`.

```bash
cd nntrainer/nntrainer/tensor/htp_backend
./build_htp.sh
```

Build output:

```
nntrainer/nntrainer/tensor/htp_backend/build_htp/
├── libhtp_ops.so        # Host stub library
├── libhtp_ops_skel.so   # DSP skel library
└── htp_ops_test         # Test binary
```

After building, `run.sh` automates pushing the build artifacts to a connected
Android device and executing the test binary (`htp_ops_test`) compiled from
`host/test.c`.

```bash
cd nntrainer/nntrainer/tensor/htp_backend

# Use default target directory (/data/local/tmp/htp_backend)
./run.sh

# Or specify a custom target directory on the device
./run.sh /data/local/tmp/my_test_dir
```

**Requirements:**
- A Qualcomm Android device must be connected and accessible via `adb`.
- The `adb_dir` variable at the top of `run.sh` must be set to the actual `adb` path on your system before running the script:
  ```bash
  # run.sh
  adb_dir="/usr/lib/android-sdk/platform-tools/adb"  # modify this to match your adb path
  ```

### Unit test build

To run the HTP backend unit tests located under `test/unittest/`,
use `android_test.sh` with the `-Denable-htp=true` option.
This builds the nntrainer test binaries, and automatically pushes
`libhtp_ops.so` and `libhtp_ops_skel.so` to the device.

```bash
# 1. Build and push test binaries + HTP libraries to device
$ ./tools/android_test.sh -Denable-htp=true

# 2. Run unittest on device
$ adb shell
(adb) $ cd /data/local/tmp/nntr_android_test
(adb) $ export LD_LIBRARY_PATH=.
(adb) $ ./<unittest_name>
```


