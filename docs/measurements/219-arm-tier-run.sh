#!/usr/bin/env bash
##
# @file    219-arm-tier-run.sh
# @brief   #219 device A/B: NNTR_MOE_TIER unset / 1 / 2 on Q28 one PD, and
#          the hybrid A with and without it (plan 219 §4 step 3)
#
# 216-fadvise-run.sh with the tier's variants and columns. Pushes the staged
# set and the config of record, reboots the phone, starts
# 216-fadvise-sampler.sh from sys.boot_completed (every 0.5 s, the model
# file's resident MiB every 4th sample), waits to uptime 60 s ("fresh") or
# 600 s ("old"), then, one binary set, env only:
#   A1 B1 A2 B2 A3 B3 A4 B4   G = 64, Q28 one PD; A = knob unset, B = 1;
#                             the 4th of each profiled (NNTR_HTP_PROFILE=2)
#   B2one                     G = 64, NNTR_MOE_TIER=2 (one-thread copy), prof
#   A512 B512                 G = 512, profiled
#   A1024 B1024               G = 1024
#   H0 H                      G = 64, hybrid A (nothing set) / with =1
# Every Q run pre-reads the model file (cat + page_cache_evict -1, #201 S3's
# "warm") for A and B alike; the hybrid runs do not. Text against this
# boot's first A run of the same G (A1 for H0 / H; strip + cmp). After each
# run: the file's resident MiB, uptime before and after, S1's ceiling
# (STOP < 3840). Stops on an md5 mismatch, a run that does not generate, a
# B run whose load did not print `tier: ... direct=1`, or a leak.
# Usage: bash 219-arm-tier-run.sh <serial> <label> [fresh|old]
set -u -o pipefail
S=${1:?serial}; LB=${2:?label}; MODE=${3:-fresh}
H=$(cd "$(dirname "$0")" && pwd)
W=/local/mnt/workspace/htp_moe/219; L=$W/logs/$LB; mkdir -p $L
APP=$W/app; CFG=$W/nntr_config.json
C=/data/local/tmp/nntrainer/causallm; D=$C/s219; M=../models/q40-qs4cx-wh; AD="adb -s $S"
exec > >(tee -a $L/sitting.out) 2>&1
stop() { echo "STOP: $*"; $AD shell touch $D/core.samples.stop; exit 1; }
[ -f $CFG ] || stop "no $CFG: stage the config of record first (handoff step 1)"
MF=$M/$(sed -n 's/.*"model_file_name": *"\([^"]*\)".*/\1/p' $CFG)
up() { $AD shell cat /proc/uptime | tr -d '\r' | cut -d' ' -f1; }
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP[^\]\n]*\] [^\n]*\n//g; s/\[PPL\] [^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|num_to_generate\|^resident [0-9]* MiB\|HTP-PROFILE'; }
cfgsum() { grep -v num_to_generate | md5sum | cut -d' ' -f1; }
$AD shell mkdir -p $D
for f in $(cd $APP && ls); do
  [ "$($AD shell md5sum $D/$f 2>/dev/null | cut -d' ' -f1)" = "$(md5sum < $APP/$f | cut -d' ' -f1)" ] || $AD push $APP/$f $D/ >/dev/null
done
$AD push $H/216-fadvise-sampler.sh $D/sampler.sh >/dev/null
$AD push $CFG $C/models/q40-qs4cx-wh/nntr_config.json >/dev/null
$AD shell chmod 755 $D/page_cache_evict $D/nntrainer_causallm $D/unittest_hvx_two_sessions
$AD shell "cd $D && md5sum $(cd $APP && ls | tr '\n' ' ')" | tr -d '\r' | sort -k2 > $L/md5_device.log
diff <(sort -k2 $W/md5.txt) $L/md5_device.log && echo "MD5 OK" || stop md5
c0=$(cfgsum < $CFG); c1=$($AD shell cat $C/models/q40-qs4cx-wh/nntr_config.json | tr -d '\r' | cfgsum)
[ "$c0" = "$c1" ] && echo "CONFIG OK $c0 (without num_to_generate)" || stop "config $c1 != $c0"
$AD shell ls -L $D/$MF > /dev/null || stop "no $MF on the device (the config's model_file_name)"
echo "model file $MF; generation_config.json md5 $($AD shell md5sum $C/models/q40-qs4cx-wh/generation_config.json | cut -d' ' -f1)"
echo "=== $LB $MODE reboot $(date '+%F %T')"
$AD reboot; $AD wait-for-device
until [ "$($AD shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = 1 ]; do sleep 1; done
echo "boot_completed uptime $(up)"
$AD shell "rm -f $D/core.samples; nohup sh $D/sampler.sh $D/core.samples 0.5 $D/$MF $D/page_cache_evict >/dev/null 2>&1 &"
wait_up() { while [ "$(up | cut -d. -f1)" -lt $1 ]; do sleep 2; done; }
run() { # run <name> <A|B|B2|H0|H> <G> <profile 0|2> <text reference run>
  local log=$1 v=$2 g=$3 pr=$4 ref=$5 e="" pre="cat $MF > /dev/null && ./page_cache_evict $MF -1 &&"
  local q="NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28"
  case $v in
    A) e="$q" ;;
    B) e="$q NNTR_MOE_TIER=1" ;;
    B2) e="$q NNTR_MOE_TIER=2" ;;
    H0) pre="" ;;
    H) pre=""; e="NNTR_MOE_TIER=1" ;;
  esac
  [ $pr = 2 ] && e="$e NNTR_HTP_PROFILE=2"
  local u0; u0=$(up)
  $AD shell "cd $D && sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $M/nntr_config.json && \
    $pre $e NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat prompt512.txt)\"" > $L/$log.log 2>&1
  local u1; u1=$(up)
  local res; res=$($AD shell "cd $D && ./page_cache_evict $MF -1" | tr -d '\r' | cut -d' ' -f2)
  grep -q '^generation:' $L/$log.log || stop "$log cannot generate ($(grep -m1 -i 'error\|fatal' $L/$log.log | cut -c1-160))"
  case $v in B|B2) grep -q 'tier: experts=.* direct=1' $L/$log.log || stop "$log: no tier line with direct=1" ;; esac
  local t; t=$(cmp -s <(strip $L/$ref.log) <(strip $L/$log.log) && echo same || echo DIFF)
  echo "$log: up $u0-$u1 | $v G=$g prof=$pr | $(grep -h '^prefill:' $L/$log.log | grep -o '[0-9.]* TPS') prefill | $(grep -h '^generation:' $L/$log.log | grep -o '[0-9.]* TPS') | $(grep -h -o 'tier=[0-9]*' $L/$log.log) $(grep -h -o 'misses=[0-9]* misses/token=.* refill_ms=[0-9.]*' $L/$log.log) | $(grep -h -o 'tier: experts=.*' $L/$log.log | tail -1) | $(grep -h -o 'file read [0-9.]* ms ([0-9.]* ms/miss)' $L/$log.log) | $(grep -h -o 'calls/token=[0-9.]*' $L/$log.log) $(grep -h -o 'close tokens=.* timeouts=0 stale=0 ' $L/$log.log | grep -q . && echo close-clean) | resident_after $res | text $t"
  local c; c=$($AD shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_two_sessions --gtest_filter=TwoSessions.S1Ceiling" 2>&1 | grep -o 'CEILING s1_mmap_mib=[0-9]*' | cut -d= -f2)
  echo "  ceiling ${c:-?}"; [ "${c:-0}" -ge 3840 ] || stop "LEAK: ceiling ${c:-?} after $log"
}
if [ "$MODE" = old ]; then wait_up 600; else wait_up 60; fi
for i in 1 2 3 4; do p=0; [ $i = 4 ] && p=2; run A$i A 64 $p A1; run B$i B 64 $p A1; done
run B2one B2 64 2 A1
run A512 A 512 2 A512; run B512 B 512 2 A512
run A1024 A 1024 0 A1024; run B1024 B 1024 0 A1024
run H0 H0 64 0 A1; run H H 64 0 A1
$AD shell touch $D/core.samples.stop; sleep 2
$AD pull $D/core.samples $L/core.samples >/dev/null
$AD shell "sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 64/' $C/models/q40-qs4cx-wh/nntr_config.json"
echo "=== $LB done $(date '+%F %T') uptime $(up)"
