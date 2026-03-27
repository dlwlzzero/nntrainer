# DSP Backend Build Instructions

This directory contains the HTP (Hexagon Tnesor Processor) backend implementation for nntrainer.

## Directory Structure

```
dsp_backend/
├── htp/              # DSP-side code (runs on Hexagon DSP)
├── host/             # Host-side code (runs on CPU)
├── include/          # Shared header files
├── build_dsp.sh      # TODO: Add explantion
└── meson.build       # TODO: Add explantion
```

## Prerequisites

1. **Hexagon SDK v6.0.0.2** - Set `HEXAGON_SDK_HOME` environment variable to SDK path
2. **CMake** (version 3.10 or higher)

**Note**: The SDK environment setup script will be sourced automatically during build.

## Building the Complete HTP Backend

The HTP backend (both DSP and host code) is now built automatically during the meson build process.

### Step 1: Set Up Environment

Set the `HEXAGON_SDK_HOME` environment variable to your Hexagon SDK installation path:

```bash
export HEXAGON_SDK_HOME=/path/to/hexagon/sdk
```

The build script will automatically:
- Source `${HEXAGON_SDK_HOME}/setup_sdk_env.source` to set up the SDK environment
- Add cross-compiler tools to PATH
- Configure and build the DSP library

### Step 2: Build with Meson

The DSP library will be built automatically as part of the meson build:

```bash
cd nntrainer
meson setup build --reconfigure -Dwerror=false -Denable-hmx=true # TODO: Add comment of werror option will be deleted in future.
ninja -C build
```

### Step 3: Verify the Build

After successful build, you should find:
```
build/htp_dsp/
├── htp_ops_test        # TODO: Add explanation
├── libhtp_ops_skel.so  # TODO: Add explanation
└── libhtp_ops.so       # TODO: Add explanation
```