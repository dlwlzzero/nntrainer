#!/usr/bin/env bash
##
# @file    236-stage.sh
# @brief   Stages #236's device sitting (plan 236 section 4 step 4): the new
#          app set beside #225's skels, sidecar and config of record, the
#          P512 / P1024 prompts, md5.txt
#
# 225-stage.sh's shape cut to what 236-run.sh reads. No DSP source changed
# (plan 236 section 2), and build.sh's skel is not bit-reproducible (LEDGER
# rule 14), so the skels are copied from #225's staged set (refused unless
# that set passes its own md5.txt): A' and B then differ in the app
# libraries only. This workstation's #225 set has v79 58e3a85f... / v81
# ca25ec2d...; the farm's rebuilt one (md5.txt 23000b96...) has its own.
# Usage: bash docs/measurements/236-stage.sh [checkout] [stage dir] [model dir] [#225 set]
set -eo pipefail
R=${1:-/home/j2z0-lee/nntrainer}
W=${2:-/local/mnt/workspace/htp_moe/236}
MD=${3:-/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh}
S225=${4:-/local/mnt/workspace/htp_moe/225}
NDK=${ANDROID_NDK:-$HOME/android-ndk-r30}
FCWH=nntr_lfm2_8b_a1b_q40_arm_fcwh.bin
FCWH_MD5=71812a91d5acdbe9e026c479db8e275e
CFG_MD5=3f6808e30b6e16e1c8592fa39977814c
[ "$(cd "$S225" && grep 'app/libnntr_hvx_skel.v[78][19].so$' md5.txt | LC_ALL=C md5sum -c | grep -c ': OK$')" = 2 ] ||
  { echo "$S225/app skels missing or not its md5.txt's" >&2; exit 1; }
[ "$(md5sum < "$R"/docs/measurements/config/q40-qs4cx-wh.nntr_config.json | cut -d' ' -f1)" = $CFG_MD5 ] ||
  { echo "the config of record is not $CFG_MD5" >&2; exit 1; }
A=$W/app
rm -rf "$A" "$W"/model; mkdir -p "$A" "$W"/model
cp "$R"/Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so} \
   "$R"/Applications/CausalLM/jni/obj/local/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so} \
   "$S225"/app/libnntr_hvx_skel.v{79,81}.so "$A"/
cp "$NDK"/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so "$A"/
cp "${HEXKL_ROOT:-$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon}"/lib/6.4.0.1/armv8_android26/libsdkl.so "$A"/
"$NDK"/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang -O2 -Wall \
  -o "$A"/page_cache_evict "$R"/tools/htp/page_cache_evict.c
cp "$R"/docs/measurements/77-prompt512.txt "$A"/p01.txt
cp "$R"/docs/measurements/prompts/p1024.txt "$A"/
cp "$R"/docs/measurements/config/q40-qs4cx-wh.nntr_config.json "$A"/cfg_new.json
cp "$MD/$FCWH" "$W"/model/
[ "$(md5sum < "$W"/model/$FCWH | cut -d' ' -f1)" = $FCWH_MD5 ] ||
  { echo "$MD/$FCWH is not the packer's sidecar ($FCWH_MD5)" >&2; exit 1; }
(cd "$W" && find app model -type f | sort | xargs md5sum > md5.txt)
echo "staged from $R @ $(git -C "$R" rev-parse --short=9 HEAD)"
cat "$W"/md5.txt
