# DSP Backend Build Instructions

This directory contains the HTP (Hexagon Tnesor Processor) backend implementation for nntrainer.

## Directory Structure

```
hmx_backend/
├── htp/              # DSP-side code (runs on Hexagon DSP)
├── host/             # Host-side code (runs on CPU)
├── include/          # Shared header files
├── build_dsp.sh      # Build script for DSP code
└── meson.build       # Meson integration for host code
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
meson setup build --reconfigure -Denable-hmx=true
ninja -C build
```

Meson will:
1. Check for `HEXAGON_SDK_HOME` environment variable
2. Automatically build the DSP library using `build_dsp.sh`
3. Compile host-side code (session.c)

### Step 3: Verify the Build

After successful build, you should find:
```
build/hmx_dsp/libhmx_dsp.a  # DSP static library (compiled for Hexagon DSP)
```

**Important Note**: The DSP library is compiled for Hexagon DSP architecture, not for the host CPU. It is NOT linked to the host code during build. Instead, it is loaded dynamically by the FastRPC runtime when needed.

The build process:
1. **Host code** (`host/session.c`) is compiled and linked into the nntrainer library
2. **DSP library** (`libhmx_dsp.a`) is compiled separately for the Hexagon DSP
3. At runtime, the host code loads the DSP library via FastRPC

## Manual DSP Build (Optional)

If you need to build the DSP code separately (for debugging or development):

```bash
cd nntrainer/nntrainer/tensor/hmx_backend
./build_dsp.sh
```

The manual build script will:
1. Check for Hexagon SDK installation
2. Create a CMakeLists.txt for the DSP build
3. Configure CMake with Hexagon cross-compiler
4. Build the DSP library using `make`

## Building the Host Code with Meson

The host code is built as part of the main nntrainer meson build:

```bash
cd nntrainer
meson setup build --reconfigure -Denable-hmx=true
ninja -C build
```

This will compile the host-side files:
- `host/session.c` - DSP session management

## Integration with Main Build

The HMX backend is integrated into nntrainer through:

1. **Meson Configuration** (`nntrainer/meson.build`):
   ```python
   if get_option('enable-hmx')
     extra_defines += '-DENABLE_HMX=1'
   endif
   ```

2. **Tensor Directory** (`nntrainer/tensor/meson.build`):
   ```python
   if get_option('enable-hmx')
     subdir('hmx_backend')
   endif
   ```

3. **Usage in FloatTensor** (`float_tensor.cpp`):
   ```cpp
   #if defined(ENABLE_HMX) && ENABLE_HMX == 1
     htp_ops_mat_mul_permuted_qk_0_d16a32(handle, output_fd, 0, 
                                           activation_fd, 0, weight_fd, 0, 
                                           M, K, N, 2);
   #endif
   ```

## Original Build Command

In the original setup, after sourcing the SDK environment, you could use:
```bash
build_cmake
```

This command is provided by the Hexagon SDK and handles the CMake configuration and build for DSP code automatically. The `build_dsp.sh` script provided here replicates this functionality.

## Troubleshooting

### "HEXAGON_ROOT environment variable not set"
Set the environment variable to your Hexagon SDK installation path:
```bash
export HEXAGON_ROOT=/path/to/hexagon/sdk
```

The build script will automatically source `${HEXAGON_ROOT}/setup_sdk_env.source` to set up the SDK environment, including adding cross-compiler tools to PATH.

### "Warning: SDK environment script not found"
This warning appears if the setup script is not found at `${HEXAGON_ROOT}/setup_sdk_env.source`. The build will still proceed if cross-compiler tools are available in PATH.

### "hexagon-unknown-linux-clang: command not found"
This usually means the SDK environment script wasn't sourced properly. The build script attempts to source it automatically. If the issue persists, manually source the SDK environment:
```bash
source ${HEXAGON_ROOT}/setup_sdk_env.source
```

### "build_cmake: command not found"
The `build_cmake` command is only available after sourcing the SDK environment script. The new build process uses `build_dsp.sh` which is integrated into meson and sources the SDK environment automatically.

### DSP library not found during runtime
Make sure the DSP library is built and the runtime can locate it. The DSP library needs to be accessible by the FastRPC runtime.

## Architecture Notes

### DSP Code
- Runs on Hexagon DSP (v68 architecture)
- Contains optimized matrix operations for Q4_0 quantized tensors
- Uses FastRPC for communication with host

### Host Code
- Runs on CPU
- Manages DSP sessions
- Handles data transfer between CPU and DSP
- Uses shared memory (rpcmem) for efficient communication

## Supported Operations

- `htp_ops_mat_mul_permuted_qk_0_d16a32` - Q4_0 matrix multiplication
- `htp_ops_rms_norm_f32` - RMS normalization (planned)
- `htp_ops_attention_qkt` - Attention operations (planned)

## Testing

Run the HMX unit tests:
```bash
cd nntrainer/build
./test/unittest/unittest_hmx_kernels --gtest_filter=nntrainer_hmx_kernels*
```

## Additional Resources

- Hexagon SDK Documentation: `/mnt/harddisk/6.0.0.2/docs/`
- FastRPC API: `/mnt/harddisk/6.0.0.2/incs/remote.md`
- IDL File Format: `/mnt/harddisk/6.0.0.2/incs/stddef/`