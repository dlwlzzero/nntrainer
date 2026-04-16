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

## Build Instructions

Hexagon DSP is only available on Android devices, so the build must target Android.
(TODO: Add comments about there are two methods.)

### Android build (with meson)
(TODO: Add comment about this build process will be needed when doing Application + HTP backend)

```bash
cd nntrainer
./tools/package_android.sh -Dwerror=false -Dmmap-read=false -Denable-htp=true
```

Build output:

```
builddir/nntrainer/tensor/htp_backend/
├── libhtp_ops.so        # Host stub library
├── libhtp_ops_skel.so   # DSP skel library
└── htp_ops_test         # Test binary (TODO: We need to change this binary created only when doing Standalone cmake build)
```

### Standalone cmake build (without meson)
(TODO: Add comment about this build process will be needed )

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

TODO: Maybe we need to split the docs, like build and test examples....

### CausalLM application build (TODO: We need to move this section somewhere...)

To build the CausalLM application with HTP support:

```bash
cd Applications/CausalLM
./build_android.sh
```

This will:
1. Build nntrainer (if not already built via `package_android.sh`)
2. Auto-detect `libhtp_ops.so` from the nntrainer build output
3. Copy HTP libraries to `jni/libs/arm64-v8a/`
4. Build CausalLM executables and libraries with `-DENABLE_HTP=1`

After build, install to device and run:

```bash
./install_android.sh
adb push res/qwen3/qwen3-4b /data/local/tmp/nntrainer/causallm/models/qwen3-4b/
adb shell /data/local/tmp/nntrainer/causallm/run_causallm.sh \
  /data/local/tmp/nntrainer/causallm/models/qwen3-4b
```



### DSP architecture setting

The target device's DSP architecture must be specified via the `DSP_ARCH`
option inside `build_htp.sh`. Set it to `v73` or `v75` depending on the
Android device before running the build.
```
# use can choose v73, v75, v79
build_cmake hexagon DSP_ARCH=<dsp_ver> 
```

## Running Standalone Tests on Device

After the [Standalone cmake build](#standalone-cmake-build-without-meson),
`run.sh` automates pushing the build artifacts to a connected Android device
and executing the test binary (`htp_ops_test`) compiled from `host/test.c`.

```bash
cd nntrainer/nntrainer/tensor/htp_backend

# Use default target directory (/data/local/tmp/htp_backend)
./run.sh

# Or specify a custom target directory on the device
./run.sh /data/local/tmp/my_test_dir
```

**Requirements:**
- A Qualcomm Android device must be connected and accessible via `adb`.
- The standalone cmake build must be completed first (`./build_htp.sh`).
- The `adb_dir` variable at the top of `run.sh` must be set to the actual `adb` path on your system before running the script:
  ```bash
  # run.sh
  adb_dir="/usr/lib/android-sdk/platform-tools/adb"  # modify this to match your adb path
  ```

## Running Unit Tests

Unit tests for the HTP backend are located under `test/unittest/`.
Before running the tests, `libhtp_ops.so` and `libhtp_ops_skel.so` must
already be built via the [Android build](#android-build-with-meson) steps above.

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
(adb) $ ./<unittest_name>
```
