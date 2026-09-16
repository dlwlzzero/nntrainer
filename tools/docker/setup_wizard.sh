#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One-time, human-driven setup for the Mac client. Walks through the steps
# that only a person can do (install the container runtime, log in to
# Qualcomm, copy model files) and verifies each one. Re-run it any time;
# completed steps are skipped.
#
#   tools/docker/setup_wizard.sh            # all steps
#   tools/docker/setup_wizard.sh sdk        # only the SDK step
#
# Steps: runtime | image | sdk | model | check

set -euo pipefail

# OrbStack installs its docker CLI under ~/.orbstack/bin; make it visible to non-login shells.
[ -d "$HOME/.orbstack/bin" ] && export PATH="$HOME/.orbstack/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HEXAGON_SDK_DIR="${HEXAGON_SDK_DIR:-$HOME/Qualcomm/Hexagon_SDK}"
QPM_DEB_DIR="${QPM_DEB_DIR:-$HOME/Qualcomm/downloads}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/Applications/CausalLM/res/qwen3/qwen3-0.6b}"
IMAGE="${NNTR_DOCKER_IMAGE:-nntrainer-hexagon-dev:ubuntu24.04}"

say()  { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m   ok: %s\033[0m\n' "$*"; }
todo() { printf '\033[1;33m   TODO: %s\033[0m\n' "$*"; }
ask()  { read -r -p "   $1 [Enter to continue] " _; }

step_runtime() {
  say "1/5 container runtime"
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    ok "docker is available: $(docker version --format '{{.Server.Platform.Name}} {{.Server.Version}}' 2>/dev/null || echo unknown)"
  else
    todo "install OrbStack (recommended, fastest amd64 emulation on Apple silicon):"
    echo  "      brew install --cask orbstack        # or download from https://orbstack.dev"
    echo  "      open -a OrbStack                    # first launch creates the docker socket"
    echo  "   (Docker Desktop also works: enable 'Use Rosetta for x86_64/amd64 emulation' in Settings > General.)"
    ask "Install it, then press"
    docker info >/dev/null 2>&1 || { echo "   docker still not reachable; rerun the wizard later."; exit 1; }
    ok "docker is available"
  fi
  local arch
  arch="$(docker run --rm --platform linux/amd64 ubuntu:24.04 uname -m 2>/dev/null || true)"
  if [ "$arch" = "x86_64" ]; then ok "linux/amd64 emulation works"; else
    echo "   linux/amd64 containers do not run (got '$arch'). Enable Rosetta/amd64 emulation in the runtime settings."; exit 1; fi
}

step_image() {
  say "2/5 dev image"
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then ok "$IMAGE exists (rebuild with: tools/docker/run.sh --build-image)"; else
    echo "   building $IMAGE (downloads NDK r26d ~1 GB; 10-20 min)"; "$SCRIPT_DIR/run.sh" --build-image; ok "built"; fi
}

step_sdk() {
  say "3/5 Hexagon SDK (6.4 or newer) under $HEXAGON_SDK_DIR"
  mkdir -p "$HEXAGON_SDK_DIR" "$QPM_DEB_DIR"
  local found
  found="$(ls -d "$HEXAGON_SDK_DIR"/6.*/ 2>/dev/null | sort -V | tail -1 || true)"
  if [ -n "$found" ] && [ -f "$found/setup_sdk_env.source" ]; then ok "SDK present: $found"; return; fi
  local deb
  deb="$(ls "$QPM_DEB_DIR"/QualcommPackageManager3*Linux*x86*.deb 2>/dev/null | head -1 || true)"
  if [ -z "$deb" ]; then
    todo "download the LINUX x86 .deb of Qualcomm Package Manager 3 (qpm-cli) with your Qualcomm account:"
    echo  "      https://qpm.qualcomm.com  ->  Download  ->  Linux (x86_64) .deb"
    echo  "      save it into: $QPM_DEB_DIR"
    echo  "   (The Linux build is used because the SDK is installed inside the amd64 container; a macOS qpm is not needed.)"
    ask "Put the .deb there, then press"
    deb="$(ls "$QPM_DEB_DIR"/QualcommPackageManager3*Linux*x86*.deb 2>/dev/null | head -1 || true)"
    [ -n "$deb" ] || { echo "   no .deb found in $QPM_DEB_DIR"; exit 1; }
  fi
  echo "   installing qpm-cli in a throwaway root container and running the interactive login + install."
  echo "   You will be asked for your Qualcomm credentials. Product name: hexagonsdk6.x (choose the newest 6.4+ version when prompted)."
  ask "Press to start"
  docker run --rm -it --platform linux/amd64 \
    -v "$QPM_DEB_DIR:/deb:ro" -v "$HEXAGON_SDK_DIR:/opt/qcom/Hexagon_SDK" \
    ubuntu:24.04 bash -lc '
      set -e
      apt-get update >/dev/null && apt-get install -y --no-install-recommends ca-certificates libglib2.0-0 >/dev/null
      dpkg -i /deb/QualcommPackageManager3*Linux*x86*.deb || apt-get install -y -f
      qpm-cli --version
      qpm-cli --login
      qpm-cli --license-activate hexagonsdk6.x
      qpm-cli --install hexagonsdk6.x --path /opt/qcom/Hexagon_SDK
      ls /opt/qcom/Hexagon_SDK
    '
  found="$(ls -d "$HEXAGON_SDK_DIR"/6.*/ 2>/dev/null | sort -V | tail -1 || true)"
  [ -n "$found" ] && [ -f "$found/setup_sdk_env.source" ] && ok "SDK installed: $found" || { echo "   SDK not found after install; check the qpm-cli output."; exit 1; }
}

step_model() {
  say "4/5 model files in $MODEL_DIR"
  local missing=0
  ls "$MODEL_DIR"/*.bin >/dev/null 2>&1 || { todo "copy the qwen3-0.6b W8CX .bin (nntr_hexpack input) here"; missing=1; }
  [ -f "$MODEL_DIR/tokenizer.json" ] || { todo "copy the HuggingFace Qwen3-0.6B directory files (tokenizer.json, tokenizer_config.json, config.json) here"; missing=1; }
  if [ $missing = 1 ]; then
    echo  "      scp -r <workstation>:<path>/qwen3-0.6b/{*.bin,tokenizer.json,tokenizer_config.json,config.json} '$MODEL_DIR/'"
    echo  "   (the packed .hexw image is regenerated in the container with nntr_hexpack; do not copy it)"
    ask "Copy them, then press"
    step_model; return
  fi
  ok "model files present"
}

step_check() {
  say "5/5 smoke test inside the container"
  "$SCRIPT_DIR/run.sh" bash -c 'echo "sdk: ${HEXAGON_SDK_ROOT:-<none>}"; echo "ndk: $ANDROID_NDK"; which hexagon-clang qaic clang-format-14 meson ninja g++ 2>/dev/null || true; python3 -c "import transformers, numpy; print(\"python ok\")"'
  "$SCRIPT_DIR/run.sh" ./tools/hexagon/build_host_x86.sh test_lowering && "$SCRIPT_DIR/run.sh" ./build_x86_hexagon/test_lowering
  ok "container builds and runs the x86 lowering test"
  echo
  echo "Done. Next: in Claude Code run /hexagon-cycle (issue #23 rebuilds hvx_impl with the new SDK)."
}

case "${1:-all}" in
  runtime) step_runtime ;;
  image)   step_runtime; step_image ;;
  sdk)     step_runtime; step_sdk ;;
  model)   step_model ;;
  check)   step_check ;;
  all)     step_runtime; step_image; step_sdk; step_model; step_check ;;
  *) echo "usage: $0 [runtime|image|sdk|model|check|all]"; exit 1 ;;
esac
