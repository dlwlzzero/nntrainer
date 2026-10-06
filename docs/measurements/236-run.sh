#!/usr/bin/env bash
##
# @file    236-run.sh
# @brief   #236's device sitting (plan 236 section 4 step 5): the hybrid (B)
#          at P1024 with the qkv / o_proj FCs in 512-row chunks, on the
#          config of record (handoff docs/measurements/236-qkv-p1024.md)
#
# 225-run.sh cut to the hybrid's cells. Staged beside 236-stage.sh's set as
# run_236.sh (handoff step 2); run it from there, phone on USB (farm session):
#   bash /local/mnt/workspace/htp_moe/236/run_236.sh <serial> [v79|v81]
# Variants (config of record cfg_new.json for both; env only otherwise):
#   A'  the #225 set as staged on this machine (A225=<dir>, default ../225;
#       skipped when absent): P1024 G64 (control: expected to die with
#       err=0x80000600 at M=1024, #225's G3) and P512 G64 (the in-sitting
#       reference for B's P512 anchor). Recorded, not gated.
#   B   the #236 set (this directory): P512 G64 (drift anchor vs #225's
#       727.3 / 55.17), P1024 G64 r1, cool, r2, G512, G1024, one
#       NNTR_HTP_PROFILE=2 run at P1024 G64 (not a speed cell)
#   Q   optional (Q=1): B + NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28 at
#       P1024 G64, model file pre-read + page_cache_evict -1
# A case that does not load or dies is VOID (logs/void_<v>_P<p>: its error,
# arena / heap lines); its other cells at that length are skipped and the
# sitting goes on. RESUMABLE per run (logs/done/<run>).
set -u -o pipefail
S=${1:?usage: run_236.sh <serial> [v79|v81]}
ARCH=${2:-v79}
case $ARCH in v79 | v81) ;; *) echo "arch must be v79 or v81" >&2; exit 1 ;; esac
W=$(cd "$(dirname "$0")" && pwd); L=$W/logs; X=$W/extra; mkdir -p $L/done
A225=${A225:-$W/../225}
C=/data/local/tmp/nntrainer/causallm; M=../models/q40-qs4cx-wh
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
pf() { case $1 in 512) echo p01.txt ;; 1024) echo p1024.txt ;; esac; }
# the generated text: what follows the prompt's echo, before the ===== box
gen() { python3 - "$1" "$W/app/$2" <<'PY'
import sys
log = open(sys.argv[1], errors="replace").read().split("\n=====")[0]
import re
# [HTP] banners print inside the generated text (stdout not line-terminated):
# drop them wherever they start, not only at a line start (#225, item 35)
log = re.sub(r"\[(HTP|PPL)[^\n]*\n?", "", log)
p = open(sys.argv[2], errors="replace").read().rstrip("\n")
i = log.rfind(p)
print(log[i + len(p):].strip("\n") if i >= 0 else "<prompt echo not found>")
PY
}
run() { # run <Ap|B|Q> <P> <G> <log> [env ...]
  local v=$1 p=$2 g=$3 log=$4 e="" pre="" d=$C/s236 pr; shift 4; pr=$(pf $p)
  [ -f $L/done/$log ] && { echo "$log: done earlier, skipped"; return 0; }
  [ -f $L/void_${v}_P$p ] && { echo "$log: VOID, $v did not load at P$p earlier ($(head -1 $L/void_${v}_P$p))"; return 0; }
  [ $v = Ap ] && d=$C/s236a
  [ $v = Q ] && { e="NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28"; pre="cat $MF > /dev/null && ./page_cache_evict $MF -1 &&"; }
  $AD logcat -c
  $AD shell "cd $d && cp cfg_new.json $M/nntr_config.json && sed -i -e 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' \
    -e 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' $M/nntr_config.json && \
    md5sum libnntr_hvx_skel.so libnntrainer.so && grep -h -E 'init_seq_len|num_to_generate|_engine' $M/nntr_config.json | tr -d ' \n' && echo && \
    $pre $e $* NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $M \"\$(cat $pr)\"" > $L/$log.log 2>&1
  $AD logcat -d | grep -iE 'adsprpc|fastrpc|nntr_hvx|token' > $L/$log.logcat || true
  grep -qi '0x8000040e' $L/$log.log && stop "$log: 0x8000040e (stale skel or stub)"
  echo "    0x80000600 (AEE_ERPC) lines: $(grep -c '0x80000600' $L/$log.log), 0x80000402 (AEE_ENOMEMORY): $(grep -c '0x80000402' $L/$log.log)"
  local fw; fw=$(grep -h '\[HTP\] fc wh:' $L/$log.log); [ -n "$fw" ] && echo "    $fw"
  if ! grep -q '^generation:' $L/$log.log; then
    { grep -h -m1 -E 'FATAL|failed|refused|no room|ENOMEM|0x80000402|0x80000600' $L/$log.log | cut -c1-240 || echo "no generation line, no error line (see $log.log / .logcat)"
      grep -h -E 'arena|fc wh:|e2e: fc arena|heap|mapped|rpcmem' $L/$log.log | sed 's/^/    /'; } > $L/void_${v}_P$p
    echo "VOID $log:"; sed 's/^/  /' $L/void_${v}_P$p
    touch $L/done/$log; return 0
  fi
  echo "$log: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'calls/token=[0-9.]*' $L/$log.log) rss=$(grep -h -o 'Max Resident Set Size: [0-9]*' $L/$log.log | grep -o '[0-9]*$')KB"
  want "$log prefill tokens" "$(sed -n 's/^prefill: \([0-9]*\) tokens.*/\1/p' $L/$log.log)" $p
  want "$log one moe m1 gemv banner" "$(grep -c 'moe m1 gemv: on (applied=' $L/$log.log)" 1
  case $v in
    Ap | B)
      want "$log hybrid (dspq on, no token driver)" "$(grep -c '\[HTP\] dspq: on' $L/$log.log)/$(grep -c 'token driver: on' $L/$log.log)" 1/0
      want "$log fc wh banner: the sidecar, heap_kib=<n> requant=0" "$(grep -c "fc wh: file=[^ ]*$FCWH .* heap_kib=[0-9]* requant=0$" $L/$log.log)" 1 ;;
    Q)
      want "$log fc wh banner: the sidecar, heap_kib=0 requant=0" "$(grep -c "fc wh: file=[^ ]*$FCWH .* heap_kib=0 requant=0$" $L/$log.log)" 1
      want "$log calls/token=1.00" "$(grep -cF 'calls/token=1.00' $L/$log.log)" 1
      grep -h -E 'graph: q4m1 weights=|e2e: fc arena|e2e: close|token driver: pool' $L/$log.log | sed 's/^/    /' ;;
  esac
  touch $L/done/$log; }

echo "=== 236 $(date '+%F %T %Z') unit=$S arch=$ARCH (done: $(ls $L/done | wc -l) runs)"
$AD get-state > /dev/null 2>&1 || stop "unit $S not attached"
echo "model: $($AD shell getprop ro.product.model | tr -d '\r') soc: $($AD shell getprop ro.soc.model | tr -d '\r')"
echo "uptime (reboot first, then wait 5 min: rule 61): $($AD shell cat /proc/uptime | tr -d '\r')"
$AD shell input keyevent 223 || true
therm t0
echo "B set md5.txt: $(md5sum < $W/md5.txt | cut -d' ' -f1)"
[ "$(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation set differs from md5.txt"
[ -f $X/loop_check.py ] || stop "extra/loop_check.py missing (handoff step 2)"
HAVE_AP=0
if [ -f $A225/md5.txt ]; then
  echo "A' set $A225 md5.txt: $(md5sum < $A225/md5.txt | cut -d' ' -f1)"
  [ "$(cd $A225 && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "A' set differs from its md5.txt"
  cmp -s $A225/app/libnntr_hvx_skel.$ARCH.so $W/app/libnntr_hvx_skel.$ARCH.so || echo "NOTE: A' and B skels differ (236-stage.sh copies #225's)"
  cmp -s $A225/app/cfg_new.json $W/app/cfg_new.json || stop "A' and B configs differ"
  HAVE_AP=1
else echo "no #225 set at $A225: A' (run 0) skipped"; fi
sets="s236"; [ $HAVE_AP = 1 ] && sets="s236a s236"
for s in $sets; do src=$W; [ $s = s236a ] && src=$A225
  $AD shell "mkdir -p $C/$s" && $AD push $src/app/. $C/$s/ > /dev/null
  [ $s = s236a ] && $AD push $W/app/p1024.txt $C/$s/ > /dev/null
  $AD shell "cd $C/$s && cp libnntr_hvx_skel.$ARCH.so libnntr_hvx_skel.so && chmod 755 nntrainer_causallm page_cache_evict"
  $AD shell "cd $C/$s && md5sum $(grep -o '  app/.*' $src/md5.txt | sed 's|  app/||' | tr '\n' ' ')" | tr -d '\r' | awk '{print $1"  app/"$2}' > $L/md5_device_$s.log
  diff <(grep -E '^[0-9a-f]{32}  app/' $src/md5.txt | sort -k2) <(sort -k2 $L/md5_device_$s.log) || stop "device app md5 differs ($s)"
done
want_fc=$(grep "  model/$FCWH" $W/md5.txt | cut -d' ' -f1)
[ "$($AD shell md5sum $C/models/q40-qs4cx-wh/$FCWH 2>/dev/null | cut -d' ' -f1)" = "$want_fc" ] || { echo "pushing the sidecar (228 MB)"; $AD push $W/model/$FCWH $C/models/q40-qs4cx-wh/ > /dev/null; }
$AD shell "cd $C/models/q40-qs4cx-wh && ln -sf nntr_lfm2_8b_a1b_q40_arm.bin nntr_lfm2.5_8b_a1b_q40_arm.bin && \
  md5sum nntr_lfm2_8b_a1b_q40_arm.bin $FCWH" | tr -d '\r' | tee $L/md5_models.log
grep -q "^$want_fc  $FCWH" $L/md5_models.log && echo "MD5 OK (app, sidecar)" || stop "device sidecar md5 differs from md5.txt"
$AD shell "cd $C/models/q40-qs4cx-wh && ([ -f generation_config.json.236 ] || cp generation_config.json generation_config.json.236) && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && grep -H do_sample generation_config.json" | tee -a $L/config.log

if [ $HAVE_AP = 1 ]; then
  echo "--- A' (#225 set): P1024 G64 control (expected VOID: err=0x80000600 M=1024), P512 G64 reference $(date +%H:%M:%S)"; cool
  run Ap 1024 64 Ap_P1024_G64_r1
  run Ap 512 64 Ap_P512_G64_r1
fi
echo "--- B P512 G64 (anchor) $(date +%H:%M:%S)"; cool
run B 512 64 B_P512_G64_r1
echo "--- B P1024 G64 r1 $(date +%H:%M:%S)"
run B 1024 64 B_P1024_G64_r1
cool
run B 1024 64 B_P1024_G64_r2
for g in 512 1024; do echo "--- B P1024 G$g $(date +%H:%M:%S)"; cool; run B 1024 $g B_P1024_G${g}_r1; done
[ "${Q:-0}" = 1 ] && { echo "--- Q P1024 G64 (optional)"; cool; run Q 1024 64 Q_P1024_G64_r1; }
echo "--- profile (not a speed cell): B P1024 G64, NNTR_HTP_PROFILE=2"; cool
run B 1024 64 prof_B_P1024 NNTR_HTP_PROFILE=2
f=$L/prof_B_P1024.log
if grep -q '^generation:' $f; then
  grep -h -E 'M>1' $f | sed 's/^/    /'
  # 6 attention layers x 1024 rows = 6144 rows in 12 calls per FC; any other
  # FC on the row is read the same way: every call 512 rows, none above
  for n in 3072 2048; do cr=$(sed -n "s/.*K=2048 *N=$n *M>1 FC *calls=\([0-9]*\) *rows=\([0-9]*\).*/\1 \2/p" $f)
    echo "    M>1 FC K=2048 N=$n calls/rows: ${cr:-none} (expect 12 6144)"
    want "M>1 FC K=2048 N=$n chunked (calls x 512 = rows)" "$(set -- $cr; [ -n "${1:-}" ] && [ $(($1 * 512)) = "$2" ] && echo yes || echo no)" yes
  done
fi
$AD shell "cd $C/models/q40-qs4cx-wh && cp $C/s236/cfg_new.json nntr_config.json && cp generation_config.json.236 generation_config.json"
echo "device: models/q40-qs4cx-wh = the config of record (pristine), generation config restored"
therm t_end

echo "--- speed (prefill / decode / last 64 TPS, peak RSS, heap_kib)"
for f in $L/[AB]*_P*_G*.log $L/Q_*.log; do [ -f $f ] || continue; b=$(basename $f .log)
  p=${b#*_P}; p=${p%%_*}
  grep -q '^generation:' $f || { echo "$b: VOID ($(head -1 $L/void_${b%%_P*}_P$p 2>/dev/null))"; continue; }
  echo "$b: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $f | grep -o '[0-9.]* TPS' | tr '\n' ' ')rss=$(grep -h -o 'Max Resident Set Size: [0-9]*' $f | grep -o '[0-9]*$')KB heap_kib=$(sed -n 's/.*fc wh: .* heap_kib=\([0-9]*\) .*/\1/p' $f)"
done | tee $L/speed.txt
echo "--- B P1024 G64 r2 == r1 (determinism)"
[ -f $L/B_P1024_G64_r2.log ] && grep -q '^generation:' $L/B_P1024_G64_r2.log &&
  want "B P1024 G64 r2 text == r1" "$([ "$(gen $L/B_P1024_G64_r1.log p1024.txt)" = "$(gen $L/B_P1024_G64_r2.log p1024.txt)" ] && echo same || echo DIFF)" same
echo "--- loops (loop_check.py; B fails a prompt only where it loops and #225's A does not)"
ref=""; for r in A_P1024_G64_r1 A_P1024_G512_r1 A_P1024_G1024_r1 Bfb_P1024_G64_r1 Bfb_P1024_G512_r1 Bfb_P1024_G1024_r1; do
  [ -f $A225/logs/$r.log ] && ref="$ref $A225/logs/$r.log"; done
[ -z "$ref" ] && echo "NOTE: #225's A / Bfb P1024 logs not found under $A225/logs: B's loop line has no reference beside it"
python3 $X/loop_check.py --prompt $W/app/p1024.txt $ref $(for f in $L/[BQ]*_P1024_G*.log; do grep -q '^generation:' $f && echo $f; done) | tee $L/loops.txt
echo "--- texts (G64 r1, for the approval table)"
for f in $A225/logs/A_P1024_G64_r1.log $L/B_P512_G64_r1.log $L/Ap_P512_G64_r1.log $L/B_P1024_G64_r1.log $L/Q_P1024_G64_r1.log; do
  [ -f $f ] && grep -q '^generation:' $f || continue; b=$(basename $f .log); p=${b#*_P}; p=${p%%_*}
  echo "## $b$([ $f = $A225/logs/A_P1024_G64_r1.log ] && echo ' (#225 sitting, CPU)')"; gen $f $(pf $p); echo; done | tee $L/texts.txt
echo "--- VOID"; for f in $L/void_*; do [ -f $f ] && { echo "$(basename $f):"; sed 's/^/  /' $f; }; done
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z') expectation mismatches (this invocation): $MIS"
