#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Generates FastRPC stub/skel from nntr_htp.idl and cross-builds the DSP skel.
# Output: build_hexagon/generated/ (qaic output), build_hexagon/skel/libnntr_htp_skel.so
#
# Prerequisite: source $HEXAGON_SDK_ROOT/setup_sdk_env.source
# Override target arch: HEX_ARCH=v75 ./tools/hexagon/build_skel.sh

set -eu

: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"

HEX_ARCH="${HEX_ARCH:-v79}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HTP_DIR="$REPO/nntrainer/tensor/hexagon/htp"
OUT="$REPO/build_hexagon"

mkdir -p "$OUT/generated" "$OUT/skel"

"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/Ubuntu/qaic" \
    -I "$HEXAGON_SDK_ROOT/incs" \
    -I "$HEXAGON_SDK_ROOT/incs/stddef" \
    -mdll -o "$OUT/generated" "$HTP_DIR/nntr_htp.idl"

"$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang" \
    -m"$HEX_ARCH" -mhvx -mhvx-length=128B -G0 -O3 -fPIC -shared \
    -Wall -Werror \
    -I "$OUT/generated" \
    -I "$HTP_DIR" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/qurt" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/posix" \
    -isystem "$HEXAGON_SDK_ROOT/incs" \
    -isystem "$HEXAGON_SDK_ROOT/incs/stddef" \
    -isystem "$HEXAGON_SDK_ROOT/ipc/fastrpc/incs" \
    "$HTP_DIR/executor.c" "$OUT/generated/nntr_htp_skel.c" \
    -o "$OUT/skel/libnntr_htp_skel.so"

echo "built: $OUT/skel/libnntr_htp_skel.so ($HEX_ARCH)"
