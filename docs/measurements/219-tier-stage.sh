#!/usr/bin/env bash
##
# @file    219-tier-stage.sh
# @brief   Stages #219's close-out sitting (the E2E row with the tier on by
#          default) from a built checkout: the app set, both skels, the
#          evictor, the three prompts, the config of record, the FC WH
#          sidecar, the runner and loop_check.py, with md5.txt
#
# 225-stage.sh with the prompts the runner reads (p64 / p01 / p1024) in app/
# and the runner and loop_check.py beside it, so there is no extra/ step.
# Refuses a skel whose ELF flags do not match its name and a sidecar that is
# not the packer's.
# Usage: bash docs/measurements/219-tier-stage.sh [checkout] [stage dir] [model dir]
set -eo pipefail
R=${1:-/home/j2z0-lee/nntrainer-sync}
W=${2:-/local/mnt/workspace/htp_moe/219t}
MD=${3:-/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh}
NDK=${ANDROID_NDK:-$HOME/android-ndk-r30}
FCWH=nntr_lfm2_8b_a1b_q40_arm_fcwh.bin
FCWH_MD5=71812a91d5acdbe9e026c479db8e275e
for a in 79 81; do
  readelf -h "$R"/test/htp/build/libnntr_hvx_skel.v$a.so | grep -q "Flags:.*0x$a\b" ||
    { echo "test/htp/build/libnntr_hvx_skel.v$a.so missing or not v$a" >&2; exit 1; }
done
grep -q "\"fc_wh_file_name\": \"$FCWH\"" "$R"/docs/measurements/config/q40-qs4cx-wh.nntr_config.json ||
  { echo "the config of record does not name $FCWH" >&2; exit 1; }
A=$W/app
rm -rf "$A" "$W"/model; mkdir -p "$A" "$W"/model
cp "$R"/Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so} \
   "$R"/Applications/CausalLM/jni/obj/local/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so} \
   "$R"/test/htp/build/libnntr_hvx_skel.v{79,81}.so "$A"/
cp "$NDK"/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so "$A"/
cp "${HEXKL_ROOT:-$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon}"/lib/6.4.0.1/armv8_android26/libsdkl.so "$A"/
"$NDK"/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang -O2 -Wall \
  -o "$A"/page_cache_evict "$R"/tools/htp/page_cache_evict.c
cp "$R"/docs/measurements/77-prompt512.txt "$A"/p01.txt
cp "$R"/docs/measurements/prompts/{p64,p1024}.txt "$A"/
cp "$R"/docs/measurements/219-tier-run.sh "$W"/run_219t.sh
cp "$R"/tools/htp/loop_check.py "$W"/
cp "$R"/docs/measurements/config/q40-qs4cx-wh.nntr_config.json "$A"/cfg_new.json
cp "$MD/$FCWH" "$W"/model/
[ "$(md5sum < "$W"/model/$FCWH | cut -d' ' -f1)" = $FCWH_MD5 ] ||
  { echo "$MD/$FCWH is not the packer's sidecar ($FCWH_MD5)" >&2; exit 1; }
(cd "$W" && find app model run_219t.sh loop_check.py -type f | sort | xargs md5sum > md5.txt)
echo "staged from $R @ $(git -C "$R" rev-parse --short=9 HEAD)"
cat "$W"/md5.txt
