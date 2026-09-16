#!/bin/bash
# tools/hexagon/run_sim_test.sh <test-name> [args...]
# SIM_TIMING=1 adds --timing (cycle-accurate core model; slower)
set -eu
: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"
HEX_ARCH="${HEX_ARCH:-v75}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO/build_hexagon/sim"
# run_main_on_hexagon ships one image per (toolchain, arch). Use the one built
# by the toolchain that compiled libnntr_sim_test.so ($DEFAULT_TOOLS_VARIANT from
# setup_sdk_env.source, e.g. toolv19 for HEXAGON_Tools 19.0.04); a lexical
# "ls | tail -1" would pick toolv88 over toolv19. Fall back to the newest
# variant only when that image is absent, and fail loudly when none exists.
SHIP="$HEXAGON_SDK_ROOT/libs/run_main_on_hexagon/ship"
RUNMAIN_DIR="$SHIP/hexagon_${DEFAULT_TOOLS_VARIANT:-toolv?}_${HEX_ARCH}"
if [ ! -d "$RUNMAIN_DIR" ]; then
  RUNMAIN_DIR="$(ls -d "$SHIP"/hexagon_toolv*_"${HEX_ARCH}" 2>/dev/null | sort -V | tail -1 || true)"
  if [ -z "$RUNMAIN_DIR" ]; then
    echo "run_sim_test.sh: no run_main_on_hexagon image for $HEX_ARCH under $SHIP" >&2
    exit 1
  fi
  echo "run_sim_test.sh: no hexagon_${DEFAULT_TOOLS_VARIANT:-toolv?}_${HEX_ARCH} image, falling back to $(basename "$RUNMAIN_DIR")" >&2
fi
RUNMAIN="$RUNMAIN_DIR/run_main_on_hexagon_sim"
ISS="$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/lib/iss"
cd "$OUT"
printf '%s\n' \
  "$ISS/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1" \
  "$ISS/l2vic.so 32 0xFC910000" > q6ss.cfg
echo "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/debugger/lnx64/qurt_model.so" > osam.cfg
if [ -n "${SIM_TIMING:-}" ]; then echo "SIM_RUN timing=on"; else echo "SIM_RUN timing=off"; fi
echo "SIM_RUN runmain=$(basename "$RUNMAIN_DIR")"
"$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-sim" -m"$HEX_ARCH" \
    ${SIM_TIMING:+--timing} \
    --simulated_returnval --usefs "$OUT" --nullptr=2 \
    --cosim_file "$OUT/q6ss.cfg" --l2tcm_base 0xd800 --rtos "$OUT/osam.cfg" \
    "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/sdksim_bin/runelf.pbn" -- \
    "$RUNMAIN" -- ./libnntr_sim_test.so "$@"
