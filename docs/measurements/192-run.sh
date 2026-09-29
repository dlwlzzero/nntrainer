#!/usr/bin/env bash
# Issue #192 (docs/measurements/192-map-window.md on htp/192-map-window):
# the single-session alternative to #178's second session -- what mapping the
# next layer's FC weights into a rotating window costs per token.
# P = unittest_hvx_two_sessions --gtest_filter='MapWindow.*' (W1-W4, W_FIELD
# lines): one session (as the app opens it) holds the app's 3696 MiB arena
# ladder and measures fastrpc_mmap/munmap per map flag (W1 a), DSP-side
# HAP_mmap/munmap (W1 b), the 24 MiB window re-attach (W2 c), the exact FC
# from a just-mapped vs a long-mapped buffer (W3 d) and the projection with
# the stop rule (W4). The probe changes no product code: A (the app set
# built from the same tree, nothing set) runs only as a G = 8 sanity before
# and after the probe (the app must still map its whole arena and generate;
# a failed sanity after the probe names a leak).
# REBOOT THE PHONE FIRST (a previous sitting may have left mappings on the
# cDSP; MapWindow's SetUp needs the full 3696 MiB ladder).
# Usage: bash /local/mnt/workspace/htp_moe/192/run_192.sh [serial]   (~20 min)
# Logs to $L, summary to $L/sitting.out. Stops on a stale skel (0x8000040e in
# the canary or A), a missing unit or an md5 mismatch; every other missing
# expected line is counted (expectation mismatches: N); W_STOP lines are
# reported, not enforced. No adb from the agent: the user runs it.
set -u -o pipefail
S=${1:-R3CY10WM83Y}
W=/local/mnt/workspace/htp_moe/192; L=$W/logs; SET=$W/set; mkdir -p $L
C=/data/local/tmp/nntrainer/causallm; D=$C/s192; M=../models/q40-qs4cx-wh
{ set +u; source /home/j2z0-lee/nntrainer-192/tools/htp/env.sh > /dev/null 2>&1; set -u; }
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0
stop() { echo "STOP: $*"; exit 1; }
want() { # want <label> <got> <expected>
  if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
therm() { echo "$1 $(date +%H:%M:%S) $(adb -s $S shell 'dumpsys battery | grep -E "^  (level|temperature)"; cat /sys/class/thermal/thermal_zone0/temp' | tr -d '\r' | tr '\n' ' ')" | tee -a $L/therm.log; }
zone0() { adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
cool() { for i in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
  echo "zone0=$z > 35000, waiting 30 s ($i/40)"; sleep 30; done; echo "zone0=$(zone0)"; }
sanity() { # sanity <log name>: A at G = 8, prompt 512, nothing set
  local log=$1
  adb -s $S shell "cd $D && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": 8/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum libnntr_hvx_skel.so && \
    LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 \
    ./nntrainer_causallm $M \"\$(cat prompt512.txt)\"" > $L/$log.log 2>&1
  grep -qi '0x8000040e' $L/$log.log && stop "$log: 0x8000040e (stale skel, rule 3)"
  echo "$log: $(grep -h -E '^(prefill|generation):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')"
  want "$log applied=0x703e1 dma_bypass=1" "$(grep -c 'moe m1 gemv: on (applied=0x703e1).* dma_bypass=1' $L/$log.log)" 1
  want "$log dspq close bad=0" "$(grep -c 'dspq: close calls=\([0-9]*\) served=\1 bad=0' $L/$log.log)" 1
  want "$log generation line" "$(grep -c '^generation:' $L/$log.log)" 1
  grep -h 'HTP arena' $L/$log.log | head -3
  grep -q '^generation:' $L/$log.log || echo "LEAK?: the app does not generate at $log (see its log; reboot)"
}

echo "=== 192 sitting  $(date '+%F %T %Z')  unit=$S  (reboot first: uptime below)"
# ---- 0. device state, cool start (1-5 min)
adb devices | tee $L/devices.log
adb devices | grep -q "^$S[[:space:]]*device" || stop "unit $S not attached"
adb -s $S shell uptime | tee $L/uptime.log
adb -s $S shell input keyevent 223 || true
therm t0; cool

# ---- 1. install and config (3 min)
[ "$(cd $SET && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation set differs from set/md5.txt"
adb -s $S shell ls -l $C/models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin
adb -s $S shell "rm -rf $D && mkdir -p $D" && adb -s $S push $SET/. $D/ > /dev/null
adb -s $S shell "rm -f $D/md5.txt; chmod 755 $D/nntrainer_causallm $D/unittest_*"
adb -s $S shell "cd $D && md5sum \$(ls -p | grep -v / | sort)" > $L/md5_device.log
diff <(sort -k2 $SET/md5.txt | tr -d '\r') <(sort -k2 $L/md5_device.log | tr -d '\r') && echo "MD5 OK" || stop "device md5 differs from md5.txt"
adb -s $S shell "cd $D/$M && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers' nntr_config.json" \
  | tee $L/config.log
for k in '"do_sample": false' '"bad_word_ids": \[124900\]' '"init_seq_len": 512' '"moe_engine": "htp"' '"moe_htp_layers": ""'; do
  want "config $k" "$(grep -c "$k" $L/config.log)" 1; done

# ---- 2. stale-skel canary: the exact FC == spec on this skel (2 min)
adb -s $S shell "cd $D && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./unittest_hvx_softmax --gtest_filter='HvxFcQ4.MatchesSpecBitExact'" > $L/canary.log 2>&1
grep -qi '0x8000040e' $L/canary.log && stop "canary: 0x8000040e (stale skel, rule 3)"
want "canary hvx_intrin cells bad=0 (5 shapes x 2 feeds)" "$(grep -c '^FC_Q4_FIELD K=.* variant=hvx_intrin feed=.* rows=8 bad=0$' $L/canary.log)" 10
therm t1; cool

# ---- 3. A sanity before the probe (2 min)
sanity sanity_0; therm t2; cool

# ---- 4. the probe (3-6 min), FARF / driver lines from logcat
adb -s $S logcat -c || true
echo "P zone0 before: $(zone0)"
adb -s $S shell "cd $D && md5sum libnntr_hvx_skel.so unittest_hvx_two_sessions && \
  LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. timeout 900 ./unittest_hvx_two_sessions \
  --gtest_filter='MapWindow.*'" > $L/P.log 2>&1
echo "P zone0 after: $(zone0)"
adb -s $S logcat -d 2>/dev/null | grep -E 'map_window|q4m1|arena|AEE|fastrpc|apps_mem|munmap|mmap' | tail -300 > $L/P.logcat || true
n40e=$(grep -ci '0x8000040e' $L/P.log)
[ "$n40e" = 0 ] || { echo "BAD P: $n40e lines with 0x8000040e (stale skel?)"; MIS=$((MIS + 1)); }
grep -E '^W_(FIELD|STOP|VERDICT)|OK \]|FAILED|SKIPPED|PASSED' $L/P.log
want "P four tests ran" "$(grep -cE '^\[ *(OK|FAILED|SKIPPED) *\] MapWindow\.W[1-4]_[A-Za-z]* \(' $L/P.log)" 4
want "P ladder_mib=3696" "$(grep -c '^W_FIELD ladder_mib=3696$' $L/P.log)" 1
want "P no leak" "$(grep -c '^W_STOP rule=leak' $L/P.log)" 0
therm t3

# ---- 5. A sanity after the probe (2 min): the app must still map its arena
sanity sanity_1; therm t4

# ---- 6. summary
echo "--- sanity (prefill / decode TPS)"
for s in sanity_0 sanity_1; do echo "$s: $(grep -h -E '^(prefill|generation):' $L/$s.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')"; done
echo "--- P"; grep -h -E '^W_(FIELD|STOP|VERDICT)' $L/P.log
echo "--- driver (apps_mem / munmap errors in the probe logcat)"; grep -h -iE 'apps_mem|munmap' $L/P.logcat | tail -20
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z')  expectation mismatches: $MIS"
