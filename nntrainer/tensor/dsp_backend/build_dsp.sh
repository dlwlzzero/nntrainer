#!/bin/bash
# Build script for HMX DSP code using Hexagon SDK
# This script is called by meson during the build process

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

set -e

# Set Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESON_BUILD_DIR="$1"
# BUILD_DIR="${SCRIPT_DIR}/build_dsp"

# if [ -n "${MESON_BUILD_DIR}" ]; then
#     BUILD_DIR="${MESON_BUILD_DIR}/htp_dsp"
# fi

echo -e "${GREEN}=== Building HTP DSP Backend ===${NC}"
# echo "Build directory: ${BUILD_DIR}"

if [ -z "${HEXAGON_SDK_HOME}" ]; then
    echo -e "${RED}ERROR: HEXAGON_SDK_HOME not set!"
    exit 1
fi

# Source SDK environment setup script
if [ -f "${HEXAGON_SDK_HOME}/setup_sdk_env.source" ]; then
    echo -e "${YELLOW}Sourcing SDK environment...${NC}"
    set +e  # Temporarily disable exit on error for sourcing
    source "${HEXAGON_SDK_HOME}/setup_sdk_env.source"
    set -e
    echo -e "${GREEN}SDK environment sourced successfully${NC}"
else
    echo -e "${YELLOW}Warning: SDK environment script not found at ${HEXAGON_SDK_HOME}/setup_sdk_env.source${NC}"
fi

# Create build directory
# mkdir -p "${BUILD_DIR}"
# cd "${BUILD_DIR}"

# Configure and build
echo "Configuring CMake..."
# build_cmake android
build_cmake hexagon DSP_ARCH=v75

echo -e "${GREEN}=== HMX DSP Backend build complete ===${NC}"
echo "Library location: ${BUILD_DIR}/libhmx_dsp.a"