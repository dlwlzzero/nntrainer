# How to Use HTP Backend

## Overview

The HTP (Hexagon Tensor Processor) backend offloads matrix multiplication
operations to Qualcomm's Hexagon DSP via the official HexKL SDKL CPU Macro
API. It uses pre-built libraries from the Hexagon SDK's `hexkl_addon`:

| Library | Runs on | Description |
|---------|---------|-------------|
| `libsdkl.so` | Host (Android CPU) | SDKL CPU Macro API (FastRPC + session internally) |
| `libhexkl_skel.so` | Hexagon DSP | Pre-built HMX/HVX compute kernels |

No custom DSP code is required — both libraries are provided by the
Qualcomm HexKL addon.

## Prerequisites

- nntrainer basic setup must be completed first. See [Getting Started](getting-started.md).
- **Hexagon SDK >= v6.3.0** with setup complete
- The HexKL addon must be present at `${HEXAGON_SDK_HOME}/addons/hexkl_addon/`
- `HEXAGON_SDK_HOME` environment variable pointing to SDK root
  ```bash
  export HEXAGON_SDK_HOME=/path/to/hexagon/sdk
  ```
- `libhexkl_skel.so` must be deployed to the target device's DSP library
  search path (e.g., `/vendor/lib/rfsa/adsp/`).

## HexKL SDKL API Overview

The HTP backend uses the SDKL CPU Macro API — the highest-level API in
the HexKL addon. SDKL handles FastRPC communication, shared memory
management, and DSP dispatch internally.

Key functions used by nntrainer:

| Function | Description |
|----------|-------------|
| `sdkl_npu_initialize` | Open a DSP session (called once at startup) |
| `sdkl_npu_finalize` | Close the DSP session |
| `sdkl_npu_alloc` / `sdkl_npu_free` | Allocate/free DSP-shared memory (for weights) |
| `sdkl_cpu_rm_to_wh_f16_inplace` | Convert FP16 row-major weight to WH layout |
| `sdkl_npu_mm_f32f16_f32` | FP32 activation × FP16 weight → FP32 output |

The SDKL tier was chosen over the lower-level Macro and Micro APIs
because it is the only tier that directly supports FP32×FP16→FP32
mixed-precision matmul, and it eliminates the need for custom DSP
skel code. For details on the API tiers, see the HexKL addon examples
under `${HEXAGON_SDK_HOME}/addons/hexkl_addon/examples/`.

## Build Instructions

Hexagon DSP is only available on Android devices, so the build must target Android.

| Method | Script | Use case |
|--------|--------|----------|
| [Android build (with meson)](#android-build-with-meson) | `package_android.sh` | Application + HTP backend |
| [Unit test build](#unit-test-build) | `android_test.sh` | Running HTP unit tests |

### Android build (with meson)

This method installs the pre-built `libsdkl.so` from the HexKL addon.
Use this when building applications that depend on the HTP backend.
For a run example, see the [CausalLM README](../Applications/CausalLM/README.md).

```bash
cd nntrainer
./tools/package_android.sh -Dwerror=false -Dmmap-read=false -Denable-htp=true
```

Build output:

```
builddir/
└── <install_dir>/lib/
    └── libsdkl.so        # Installed from hexkl_addon (pre-built)
```

Note: `libhexkl_skel.so` (the DSP-side library) must be pushed to
the device separately. It is located at:
```
${HEXAGON_SDK_HOME}/addons/hexkl_addon/lib/hexagon_<ver>/libhexkl_skel.so
```

### Unit test build

To run the HTP backend unit tests located under `test/unittest/`,
use `android_test.sh` with the `-Denable-htp=true` option.
This builds the nntrainer test binaries and automatically pushes
`libsdkl.so` to the device.

```bash
# 1. Build and push test binaries + SDKL library to device
$ ./tools/android_test.sh -Denable-htp=true

# 2. Push libhexkl_skel.so to device DSP library path
$ adb push ${HEXAGON_SDK_HOME}/addons/hexkl_addon/lib/hexagon_<ver>/libhexkl_skel.so \
    /vendor/lib/rfsa/adsp/

# 3. Run unittest on device
$ adb shell
(adb) $ cd /data/local/tmp/nntr_android_test
(adb) $ export LD_LIBRARY_PATH=.
(adb) $ ./<unittest_name>
```

## Directory Structure

```
nntrainer/tensor/htp_backend/
├── include/
│   └── sdkl_interface.h    # Runtime dlopen interface to libsdkl.so
├── meson.build              # Build configuration
└── README.md                # Backend overview
```
