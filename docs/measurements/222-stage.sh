#!/usr/bin/env bash
##
# @file    222-stage.sh
# @brief   Stages #222's closing LFM2.5 sitting (old vs new nntr_config.json)
#          from a built checkout (handoff docs/measurements/222-config-refresh.md)
#
# Copies the app set, both skels (v79 = S25, v81 = S26; the runner pushes
# the one its second argument names), the evictor, the eight prompts, the
# two configs and the runner into the stage directory and writes md5.txt,
# which the runner checks on both ends. Refuses a skel whose ELF flags do
# not match its name.
# Usage: bash docs/measurements/222-stage.sh [checkout] [stage dir]
set -eo pipefail
R=${1:-/home/j2z0-lee/nntrainer-222}
W=${2:-/local/mnt/workspace/htp_moe/222}
NDK=${ANDROID_NDK:-$HOME/android-ndk-r30}
for a in 79 81; do
  readelf -h "$R"/test/htp/build/libnntr_hvx_skel.v$a.so | grep -q "Flags:.*0x$a\b" ||
    { echo "test/htp/build/libnntr_hvx_skel.v$a.so missing or not v$a" >&2; exit 1; }
done
A=$W/app
rm -rf "$A"; mkdir -p "$A"
cp "$R"/Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so} \
   "$R"/Applications/CausalLM/jni/obj/local/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so} \
   "$R"/test/htp/build/libnntr_hvx_skel.v{79,81}.so "$A"/
cp "$NDK"/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so "$A"/
cp "${HEXKL_ROOT:-$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon}"/lib/6.4.0.1/armv8_android26/libsdkl.so "$A"/
"$NDK"/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang -O2 -Wall \
  -o "$A"/page_cache_evict "$R"/tools/htp/page_cache_evict.c
cp "$R"/docs/measurements/77-prompt512.txt "$A"/p01.txt
for i in 2 3 4 5 6 7 8; do cp "$R"/docs/measurements/prompts/bitset-0$i-*.txt "$A"/p0$i.txt; done
cp "$R"/docs/measurements/config/q40-qs4cx-wh.nntr_config.2026-09-21.json "$A"/cfg_old.json
cp "$R"/docs/measurements/config/q40-qs4cx-wh.nntr_config.json "$A"/cfg_new.json
cp "$R"/docs/measurements/222-run.sh "$W"/run_222.sh
(cd "$W" && { find app -type f | sort | xargs md5sum; md5sum run_222.sh; } > md5.txt)
echo "staged from $R @ $(git -C "$R" rev-parse --short=9 HEAD)"
cat "$W"/md5.txt
