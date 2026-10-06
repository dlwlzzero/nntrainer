#!/usr/bin/env bash
##
# @file    219-tier-run.sh
# @brief   #219 close-out device sitting: the E2E one-PD row of the table of
#          record (P64 / P512 / P1024 x G64 / G512 / G1024) with the tier on
#          by default (handoff docs/measurements/219-tier-table.md)
#
# 225-run.sh cut to the E2E row plus two anchors. Staged with 219-tier-stage.sh
# as run_219t.sh; run it from there, phone on USB (farm session):
#   bash /local/mnt/workspace/htp_moe/219t/run_219t.sh <serial> [v79|v81]
# Variants (one binary set, the config of record cfg_new.json; env only):
#   Q   NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28, NNTR_MOE_TIER unset (the
#       new default: 2, the one-thread copy); model file pre-read +
#       page_cache_evict -1 (as #201 / #216 / #219 / #225)
#   Q0  Q + NNTR_MOE_TIER=0 (no tier), P512 G64 only: the in-sitting anchor
#       against #225's 42.78
#   B   the hybrid of record (nothing set), P512 G64 only: unchanged
#       against #225's B. Its experts are resident, so it never reaches the
#       tier hook; the hybrid-with-pool case is the host line's
#       (run_inproc_e2e.sh "E2E tier unset ... hybrid no tier ok")
# Order per prompt length (cool start per block, zone0 <= 35 C):
#   G64 Q, cool, Q (r2); G512 Q; G1024 Q; after P512's G64 block: Q0, B
# A run that does not load or dies is VOID (logs/void_<run>: its error and
# arena lines) and the sitting goes on. RESUMABLE per run (logs/done/<run>);
# after a STOP reboot the phone and run the same command again.
set -u -o pipefail
S=${1:?usage: run_219t.sh <serial> [v79|v81]}
ARCH=${2:-v79}
case $ARCH in v79 | v81) ;; *) echo "arch must be v79 or v81" >&2; exit 1 ;; esac
W=$(cd "$(dirname "$0")" && pwd); L=$W/logs; mkdir -p $L/done
C=/data/local/tmp/nntrainer/causallm; D=$C/s219t; M=../models/q40-qs4cx-wh
MF=$M/nntr_lfm2_8b_a1b_q40_arm.bin; FCWH=nntr_lfm2_8b_a1b_q40_arm_fcwh.bin
AD="adb -s $S"
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0
stop() { echo "STOP: $*"; echo "  -> reboot the phone and run the same command again: it resumes at the first unfinished run"; exit 1; }
want() { if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
therm() { echo "$1 $(date +%H:%M:%S) $($AD shell 'dumpsys battery | grep -E "^  (level|temperature)"; cat /sys/class/thermal/thermal_zone0/temp' | tr -d '\r' | tr '\n' ' ')" | tee -a $L/therm.log; }
zone0() { $AD shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
cool() { local i z; for i in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
    echo "zone0=$z > 35000, waiting 30 s ($i/40)"; sleep 30; done; echo "block start zone0=$(zone0)"; }
pf() { case $1 in 64) echo p64.txt ;; 512) echo p01.txt ;; 1024) echo p1024.txt ;; esac; }
# the generated text: what follows the prompt's echo, before the ===== box;
# [HTP] / [PPL] banners print inside it (stdout not line-terminated): drop
# them wherever they start (#225 open item 35)
gen() { python3 - "$1" "$W/app/$2" <<'PY'
import re, sys
log = open(sys.argv[1], errors="replace").read().split("\n=====")[0]
log = re.sub(r"\[(HTP|PPL)[^\n]*\n?", "", log)
p = open(sys.argv[2], errors="replace").read().rstrip("\n")
i = log.rfind(p)
print(log[i + len(p):].strip("\n") if i >= 0 else "<prompt echo not found>")
PY
}
run() { # run <Q|Q0|B> <P> <G> <log>
  local v=$1 p=$2 g=$3 log=$4 pr; pr=$(pf $2)
  local e="NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28" pre="cat $MF > /dev/null && ./page_cache_evict $MF -1 &&"
  [ -f $L/done/$log ] && { echo "$log: done earlier, skipped"; return 0; }
  case $v in Q0) e="$e NNTR_MOE_TIER=0" ;; B) e=""; pre="" ;; esac
  $AD logcat -c
  $AD shell "cd $D && cp cfg_new.json $M/nntr_config.json && sed -i -e 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' \
    -e 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' $M/nntr_config.json && \
    md5sum libnntr_hvx_skel.so && \
    $pre $e NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat $pr)\"" > $L/$log.log 2>&1
  $AD logcat -d | grep -iE 'adsprpc|fastrpc|nntr_hvx|token' > $L/$log.logcat || true
  grep -qi '0x8000040e' $L/$log.log && stop "$log: 0x8000040e (stale skel or stub)"
  touch $L/done/$log
  if ! grep -q '^generation:' $L/$log.log; then
    { grep -h -m1 -E 'FATAL|failed|refused|no room|ENOMEM|0x80000402' $L/$log.log | cut -c1-240 || echo "no generation line, no error line (see $log.log / .logcat)"
      grep -h -E 'arena|fc wh:|tier:|heap|mapped' $L/$log.log | sed 's/^/    /'; } > $L/void_$log
    echo "VOID $log:"; sed 's/^/  /' $L/void_$log; return 0
  fi
  echo "$log: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'calls/token=[0-9.]*' $L/$log.log) rss=$(grep -h -o 'Max Resident Set Size: [0-9]*' $L/$log.log | grep -o '[0-9]*$')KB"
  want "$log prefill tokens" "$(sed -n 's/^prefill: \([0-9]*\) tokens.*/\1/p' $L/$log.log)" $p
  want "$log one moe m1 gemv banner" "$(grep -c 'moe m1 gemv: on (applied=' $L/$log.log)" 1
  if [ $v = B ]; then
    want "$log hybrid (dspq on, no token driver)" "$(grep -c '\[HTP\] dspq: on' $L/$log.log)/$(grep -c 'token driver: on' $L/$log.log)" 1/0
    want "$log no tier on the hybrid" "$(grep -c 'tier: experts=' $L/$log.log)" 0
    return 0
  fi
  local k=2 ex=88; [ $v = Q0 ] && { k=0; ex=none; }
  want "$log driver tier=$k" "$(grep -o 'token driver: on .* tier=[0-9]*' $L/$log.log | grep -o 'tier=[0-9]*$')" tier=$k
  want "$log tier: experts" "$(grep -o 'tier: experts=[0-9]*' $L/$log.log | tail -1 | cut -d= -f2 | grep . || echo none)" $ex
  [ $v = Q ] && want "$log tier: direct=1" "$(grep -c 'tier: experts=.* direct=1' $L/$log.log)" 3
  [ $v = Q ] && want "$log tier_reads=0" "$(grep -o 'tier_reads=[0-9]*' $L/$log.log)" tier_reads=0
  want "$log fc wh banner: the sidecar, heap_kib=0 requant=0" "$(grep -c "fc wh: file=[^ ]*$FCWH .* heap_kib=0 requant=0$" $L/$log.log)" 1
  want "$log close clean" "$(grep -c 'token driver: close tokens=[0-9]* hops/token=0.00 .* timeouts=0 stale=0 .* id_mismatch=0 ' $L/$log.log)" 1
  want "$log arena unmapped" "$(grep -c 'e2e: close .* unmap_fail=0 detach_fail=0 ' $L/$log.log)" 1
  want "$log calls/token=1.00" "$(grep -cF 'calls/token=1.00' $L/$log.log)" 1
  local mm s1; mm=$(sed -n 's/.*e2e: fc arena .* mapped_mib=\([0-9]*\) .*/\1/p' $L/$log.log)
  s1=$(sed -n 's/.*e2e: fc arena .* s1_arena_mib=\([0-9]*\) .*/\1/p' $L/$log.log)
  want "$log mapped_mib + s1_arena_mib <= the 3840 MiB ceiling" "$([ $((${mm:-9999} + ${s1:-9999})) -le 3840 ] && echo yes || echo "no (${mm:-none} + ${s1:-none})")" yes
  grep -h -E 'tier: experts=|e2e: fc arena|e2e: close|token driver: pool' $L/$log.log | sed 's/^/    /'; }

echo "=== 219t $(date '+%F %T %Z') unit=$S arch=$ARCH (done: $(ls $L/done | wc -l) runs)"
$AD get-state > /dev/null 2>&1 || stop "unit $S not attached"
echo "model: $($AD shell getprop ro.product.model | tr -d '\r') soc: $($AD shell getprop ro.soc.model | tr -d '\r')"
echo "uptime (reboot first, then wait 5 min: rule 61): $($AD shell cat /proc/uptime | tr -d '\r')"
$AD shell input keyevent 223 || true
therm t0
[ "$(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation set differs from md5.txt"
$AD shell "mkdir -p $D" && $AD push $W/app/. $D/ > /dev/null
$AD shell "cd $D && cp libnntr_hvx_skel.$ARCH.so libnntr_hvx_skel.so && chmod 755 nntrainer_causallm page_cache_evict"
$AD shell "cd $D && md5sum $(grep -o '  app/.*' $W/md5.txt | sed 's|  app/||' | tr '\n' ' ')" | tr -d '\r' | awk '{print $1"  app/"$2}' > $L/md5_device.log
diff <(grep -E '^[0-9a-f]{32}  app/' $W/md5.txt | sort -k2) <(sort -k2 $L/md5_device.log) || stop "device app md5 differs"
want_fc=$(grep "  model/$FCWH" $W/md5.txt | cut -d' ' -f1)
[ "$($AD shell md5sum $D/$M/$FCWH 2>/dev/null | cut -d' ' -f1)" = "$want_fc" ] || { echo "pushing the sidecar (228 MB)"; $AD push $W/model/$FCWH $D/$M/ > /dev/null; }
$AD shell "cd $D/$M && ln -sf nntr_lfm2_8b_a1b_q40_arm.bin nntr_lfm2.5_8b_a1b_q40_arm.bin && \
  md5sum nntr_lfm2_8b_a1b_q40_arm.bin $FCWH" | tr -d '\r' | tee $L/md5_models.log
grep -q "^$want_fc  $FCWH" $L/md5_models.log && echo "MD5 OK (app, sidecar)" || stop "device sidecar md5 differs from md5.txt"
$AD shell "cd $D/$M && ([ -f generation_config.json.219t ] || cp generation_config.json generation_config.json.219t) && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && grep -H do_sample generation_config.json" | tee -a $L/config.log

for p in 64 512 1024; do
  echo "--- P$p G=64 $(date +%H:%M:%S)"; cool
  run Q $p 64 Q_P${p}_G64_r1
  cool
  run Q $p 64 Q_P${p}_G64_r2
  [ $p = 512 ] && { run Q0 512 64 Q0_P512_G64_r1; run B 512 64 B_P512_G64_r1; }
  for g in 512 1024; do
    echo "--- P$p G=$g $(date +%H:%M:%S)"; cool
    run Q $p $g Q_P${p}_G${g}_r1
  done
  therm t_P$p
done
$AD shell "cd $D && cp cfg_new.json $M/nntr_config.json && cp $M/generation_config.json.219t $M/generation_config.json"
echo "device configs: $M = the config of record (pristine), generation config restored"
therm t_end

echo "--- speed (prefill / decode / last 64 TPS, peak RSS) and the tier (Q: tier_hits / tier_waits / tier_reads, miss_wait_us/token; mapped + s1_arena MiB)"
for f in $L/Q*_P*_G*.log $L/B_P*_G*.log; do [ -f $f ] || continue; b=$(basename $f .log)
  grep -q '^generation:' $f || { echo "$b: VOID"; continue; }
  echo "$b: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $f | grep -o '[0-9.]* TPS' | tr '\n' ' ')rss=$(grep -h -o 'Max Resident Set Size: [0-9]*' $f | grep -o '[0-9]*$')KB $(grep -h -o 'tier: experts=[0-9]* mib=[0-9.]*' $f | tail -1) $(grep -h -o 'misses=[0-9]* misses/token=[0-9.]* miss_wait_us/token=[0-9.]*' $f) $(grep -h -o 'tier_hits=[0-9]* tier_waits=[0-9]* tier_wait_us=[0-9]* tier_reads=[0-9]*' $f) ceiling=$(sed -n 's/.*e2e: fc arena .* mapped_mib=\([0-9]*\) .* s1_arena_mib=\([0-9]*\) .*/\1+\2/p' $f)"
done | tee $L/speed.txt
echo "--- G64 run 2 == run 1 (the same binary and config are deterministic)"
n=0; for f in $L/Q_P*_G64_r2.log; do r1=${f%_r2.log}_r1.log; p=$(basename $f); p=${p#*_P}; p=${p%%_*}
  grep -q '^generation:' $f && grep -q '^generation:' $r1 || continue
  [ "$(gen $f $(pf $p))" = "$(gen $r1 $(pf $p))" ] || { echo "DIFF $(basename $f .log)"; n=$((n + 1)); }; done
want "every r2 text == its r1" $n 0
echo "--- Q P512 G64 text == Q0 (the tier copies the file's bytes)"
[ -f $L/Q0_P512_G64_r1.log ] && want "Q == Q0 text" "$([ "$(gen $L/Q_P512_G64_r1.log p01.txt)" = "$(gen $L/Q0_P512_G64_r1.log p01.txt)" ] && echo same || echo DIFF)" same
echo "--- loops (loop_check.py; recorded, T2)"
for p in 64 512 1024; do ls $L/[QB]*_P${p}_G*.log > /dev/null 2>&1 || continue
  python3 $W/loop_check.py --prompt $W/app/$(pf $p) $(for f in $L/[QB]*_P${p}_G*.log; do grep -q '^generation:' $f && echo $f; done)
done | tee $L/loops.txt
echo "--- texts (G=64 r1, for the approval table)"
for p in 64 512 1024; do for v in Q Q0 B; do f=$L/${v}_P${p}_G64_r1.log; [ -f $f ] && grep -q '^generation:' $f || continue
  echo "## $v P$p"; gen $f $(pf $p); echo; done; done | tee $L/texts.txt
echo "--- VOID"; for f in $L/void_*; do [ -f $f ] && { echo "$(basename $f):"; sed 's/^/  /' $f; }; done
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z') expectation mismatches (this invocation): $MIS"
