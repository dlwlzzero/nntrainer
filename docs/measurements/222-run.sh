#!/usr/bin/env bash
##
# @file    222-run.sh
# @brief   #222's closing LFM2.5 sitting: the 2026-09-21 nntr_config.json
#          against the config of record, hybrid and one PD, plus the CPU
#          control (handoff docs/measurements/222-config-refresh.md)
#
# Staged by 222-stage.sh as run_222.sh; run from the stage directory's copy:
#   bash /local/mnt/workspace/htp_moe/222/run_222.sh <serial> <v79|v81> [pool C]
# The serial is required (no default). v79 = S25 Ultra, v81 = S26 Ultra:
# the skel of that arch is pushed. pool C (default 28): use 24 when Q28
# cannot load and say so.
# Variants (one binary set; config and env only):
#   Aold   hybrid, cfg_old.json (the sittings' config until now)
#   Anew   hybrid, cfg_new.json (the config of record: conv_block / dense_ffn /
#          attn_proj on the HTP, init_seq_len 1024, relative tokenizer,
#          model_file_name nntr_lfm2.5_8b_a1b_q40_arm.bin = a symlink)
#   Qold   Aold + NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=<C> (one PD)
#   Qnew   Anew + the same
#   CPU    the q40 model, its own config (text and speed control)
# Both NPU configs get the sittings' overlay: num_to_generate G,
# bad_word_ids [124900]; generation_config.json do_sample false.
# Order (cool start per block, zone0 <= 35 C):
#   G 64:   Aold Anew Qold Qnew, cool, Qnew Qold Anew Aold, CPU
#   G 512:  Aold Anew Qold Qnew CPU     G 1024: the same
#   profile: Anew and Qnew at G 64, NNTR_HTP_PROFILE=2 (the M>1 rows)
#   PPL: 8 prompts at G 256 with NNTR_PPL=1 (prefill PPL) and
#        NNTR_PPL_DECODE (Aold writes its continuation; Aold p01 once more
#        forced as the null check; Anew and Qnew forced on it)
# RESUMABLE per run (logs/done/<run>); after any STOP reboot the phone and
# run the same command again.
set -u -o pipefail
S=${1:?usage: run_222.sh <serial> <v79|v81> [pool C]}
ARCH=${2:?usage: run_222.sh <serial> <v79|v81> [pool C]}
PC=${3:-28}
case $ARCH in v79 | v81) ;; *) echo "arch must be v79 or v81" >&2; exit 1 ;; esac
W=$(cd "$(dirname "$0")" && pwd); L=$W/logs; mkdir -p $L/done
C=/data/local/tmp/nntrainer/causallm; D=$C/s222; M=../models/q40-qs4cx-wh; MC=../models/q40
MF=$M/nntr_lfm2_8b_a1b_q40_arm.bin
AD="adb -s $S"
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0
stop() { echo "STOP: $*"; echo "  -> reboot the phone and run the same command again: it resumes at the first unfinished run"; exit 1; }
want() { if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
therm() { echo "$1 $(date +%H:%M:%S) $($AD shell 'dumpsys battery | grep -E "^  (level|temperature)"; cat /sys/class/thermal/thermal_zone0/temp' | tr -d '\r' | tr '\n' ' ')" | tee -a $L/therm.log; }
zone0() { $AD shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
cool() { local i z; for i in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
    echo "zone0=$z > 35000, waiting 30 s ($i/40)"; sleep 30; done; echo "block start zone0=$(zone0)"; }
# the generated text: what follows the prompt's echo, before the ===== box
gen() { python3 - "$1" "$W/app/$2" <<'PY'
import sys
log = open(sys.argv[1], errors="replace").read().split("\n=====")[0]
log = "\n".join(l for l in log.split("\n") if not l.startswith(("[HTP", "[PPL]")))
p = open(sys.argv[2], errors="replace").read().rstrip("\n")
i = log.rfind(p)
print(log[i + len(p):].strip("\n") if i >= 0 else "<prompt echo not found>")
PY
}
run() { # run <Aold|Anew|Qold|Qnew|CPU> <G> <log> <prompt file> [env ...]
  local v=$1 g=$2 log=$3 p=$4 e="" pre="" m=$M cfg; shift 4
  [ -f $L/done/$log ] && { echo "$log: done earlier, skipped"; return 0; }
  [ -f $L/void_$v ] && { echo "$log: VOID, $v did not load earlier ($(cat $L/void_$v))"; return 0; }
  case $v in
    CPU) m=$MC ;;
    Q*) e="NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=$PC"
        pre="cat $MF > /dev/null && ./page_cache_evict $MF -1 &&" ;;
  esac
  case $v in *old) cfg=cfg_old.json ;; *new) cfg=cfg_new.json ;; *) cfg= ;; esac
  [ -n "$cfg" ] && cfg="cp $cfg $m/nntr_config.json && sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' $m/nntr_config.json &&"
  $AD logcat -c
  $AD shell "cd $D && $cfg sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' $m/nntr_config.json && \
    md5sum libnntr_hvx_skel.so $m/nntr_config.json && $pre $e $* NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $m \"\$(cat $p)\"" > $L/$log.log 2>&1
  $AD logcat -d | grep -iE 'adsprpc|fastrpc|nntr_hvx|token' > $L/$log.logcat || true
  grep -qi '0x8000040e' $L/$log.log && stop "$log: 0x8000040e (stale skel or stub)"
  grep -q FATAL $L/$log.log && stop "$log: $(grep -m1 FATAL $L/$log.log | cut -c1-200)"
  grep -q 'token driver: token .* failed\|token driver: .* transport failed' $L/$log.log && stop "$log: a token failed"
  if ! grep -q '^generation:' $L/$log.log; then
    # a new-config variant that does not load is a measured outcome (the
    # keys' FC copies beside the pool and the FC set may not fit the PD's
    # address budget): recorded, its other cells skipped, the sitting goes on
    if [ "${v%new}" != "$v" ]; then
      grep -h -m1 -E 'FATAL|failed|refused|no room' $L/$log.log | cut -c1-240 > $L/void_$v
      echo "VOID $log: $(cat $L/void_$v)"; grep -h -E 'arena|registered|rpcmem' $L/$log.log | sed 's/^/    /'
      touch $L/done/$log; return 0
    fi
    stop "$log cannot generate (Q$PC does not load: rerun with pool C 24; logcat in $log.logcat)"
  fi
  echo "$log: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'calls/token=[0-9.]*' $L/$log.log) $(grep -h -o 'resident [0-9]* MiB' $L/$log.log) rss=$(grep -h -o 'Max Resident Set Size: [0-9]*' $L/$log.log | grep -o '[0-9]*$')KB $(grep -h -o '\[PPL\] prompt .*\|\[PPL\] decode tokens=[^ ]* nll/token=[^ ]* ppl=[^ ]* top1=[^ ]*' $L/$log.log | tr '\n' ' ')"
  if [ $v = CPU ]; then
    want "$log no HTP banner" "$(grep -c 'moe m1 gemv\|dspq: on' $L/$log.log)" 0
  else
    want "$log one moe m1 gemv banner" "$(grep -c 'moe m1 gemv: on (applied=' $L/$log.log)" 1
    # the CPU's skipped FC work per resident row: 18 conv blocks + 6 x
    # (qkv, attention_out) + the dense FFNs (old: 2 x 3 FC layers, new:
    # 2 dense_ffn layers)
    local per=36; [ "${v%new}" != "$v" ] && per=32
    case $v in
      A*) want "$log hybrid (dspq on, no token driver)" "$(grep -c '\[HTP\] dspq: on' $L/$log.log)/$(grep -c 'token driver: on' $L/$log.log)" 1/0 ;;
      Q*)
        want "$log close clean" "$(grep -c 'token driver: close tokens=[0-9]* hops/token=0.00 .* timeouts=0/0 stale=0/0 .* id_mismatch=0 ' $L/$log.log)" 1
        want "$log fc arena unmapped" "$(grep -c 's2: close .* unmap_fail=0 detach_fail=0 ' $L/$log.log)" 1
        want "$log calls/token=1.00" "$(grep -cF 'calls/token=1.00' $L/$log.log)" 1
        local t sk; t=$(sed -n 's/.*graph: forward calls=[0-9]* tokens=\([0-9]*\) .*/\1/p' $L/$log.log)
        sk=$(sed -n 's/.*graph: cpu fc skipped=\([0-9]*\)$/\1/p' $L/$log.log)
        want "$log cpu fc skipped = $per x tokens" "${sk:-none}" "$((per * ${t:-0}))"
        grep -h 'token driver: pool' $L/$log.log | sed 's/^/    /' ;;
    esac
  fi
  touch $L/done/$log; }

echo "=== 222 $(date '+%F %T %Z') unit=$S arch=$ARCH pool C=$PC (done: $(ls $L/done | wc -l) runs)"
$AD get-state > /dev/null 2>&1 || stop "unit $S not attached"
echo "model: $($AD shell getprop ro.product.model | tr -d '\r') soc: $($AD shell getprop ro.soc.model | tr -d '\r')"
echo "uptime (reboot first, then wait 5 min: rule 61): $($AD shell cat /proc/uptime | tr -d '\r')"
$AD shell input keyevent 223 || true
therm t0
[ "$(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation set differs from md5.txt"
$AD shell "mkdir -p $D" && $AD push $W/app/. $D/ > /dev/null
$AD shell "cd $D && cp libnntr_hvx_skel.$ARCH.so libnntr_hvx_skel.so && chmod 755 nntrainer_causallm page_cache_evict"
$AD shell "cd $D && md5sum $(grep -o '  app/.*' $W/md5.txt | sed 's|  app/||' | tr '\n' ' ')" | tr -d '\r' | awk '{print $1"  app/"$2}' > $L/md5_device.log
diff <(grep -E '^[0-9a-f]{32}  app/' $W/md5.txt | sort -k2) <(sort -k2 $L/md5_device.log) && echo "MD5 OK" || stop "device md5 differs"
# the config of record's model_file_name: a symlink to the same bin (user step 2 of the handoff, idempotent)
$AD shell "cd $D/$M && ln -sf nntr_lfm2_8b_a1b_q40_arm.bin nntr_lfm2.5_8b_a1b_q40_arm.bin && ls -l nntr_lfm2*.bin && \
  md5sum nntr_lfm2_8b_a1b_q40_arm.bin nntr_lfm2.5_8b_a1b_q40_arm.bin $D/$MC/nntr_lfm2_8b_a1b_q40_arm.bin" | tr -d '\r' | tee $L/md5_models.log
for m in $M $MC; do
  $AD shell "cd $D/$m && ([ -f generation_config.json.222 ] || cp generation_config.json generation_config.json.222) && \
    ([ -f nntr_config.json.222 ] || cp nntr_config.json nntr_config.json.222) && \
    sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && grep -H do_sample generation_config.json" | tee -a $L/config.log
done
$AD shell "cd $D/$MC && sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && grep -H -E 'bad_word_ids|init_seq_len' nntr_config.json" | tee -a $L/config.log

echo "--- G=64 $(date +%H:%M:%S)"; cool
for v in Aold Anew Qold Qnew; do run $v 64 ${v}_G64_r1 p01.txt; done
cool
for v in Qnew Qold Anew Aold; do run $v 64 ${v}_G64_r2 p01.txt; done
run CPU 64 CPU_G64 p01.txt
therm t_G64
for g in 512 1024; do
  echo "--- G=$g $(date +%H:%M:%S)"; cool
  for v in Aold Anew Qold Qnew CPU; do run $v $g ${v}_G${g}_r1 p01.txt; done
  therm t_G$g
done
echo "--- profiles (not speed cells): the keys' M>1 rows"; cool
for v in Anew Qnew; do run $v 64 prof_$v p01.txt NNTR_HTP_PROFILE=2; done
for v in Anew Qnew; do
  [ -f $L/void_$v ] && continue
  r=""; for k in conv dense FC; do grep -qE "M>1 $k +calls=[1-9]" $L/prof_$v.log && r="$r$k "; done
  want "prof_$v M>1 rows (the keys acted at prefill)" "$r" "conv dense FC "
  grep -h 'M>1' $L/prof_$v.log | sed 's/^/    /'
done
echo "--- PPL: 8 prompts at G=256 (prefill PPL + decode PPL forced on Aold's continuation)"; cool
for i in 1 2 3 4 5 6 7 8; do
  [ -f $L/done/ppl_Aold_p0$i ] || $AD shell "rm -f $D/cont_p0$i.ids"
  run Aold 256 ppl_Aold_p0$i p0$i.txt NNTR_PPL=1 NNTR_PPL_DECODE=cont_p0$i.ids
  [ $i = 1 ] && run Aold 256 ppl_Aoldf_p01 p01.txt NNTR_PPL=1 NNTR_PPL_DECODE=cont_p01.ids
  for v in Anew Qnew; do run $v 256 ppl_${v}_p0$i p0$i.txt NNTR_PPL=1 NNTR_PPL_DECODE=cont_p0$i.ids; done
done
$AD shell "cd $D && cp cfg_new.json $M/nntr_config.json && cp $M/generation_config.json.222 $M/generation_config.json && \
  cp $MC/nntr_config.json.222 $MC/nntr_config.json && cp $MC/generation_config.json.222 $MC/generation_config.json"
echo "device configs: $M = the config of record (pristine), $MC and both generation configs restored"
therm t_end

echo "--- speed (prefill / decode / last 64 TPS, peak RSS; text vs Aold r1 of the same G: same | same+1st (the old config dropped the first token) | DIFF)"
for f in $L/[AQC]*_G*.log; do b=$(basename $f .log); g=${b#*_G}; g=${g%%_*}
  grep -q '^generation:' $f || { echo "$b: VOID"; continue; }
  ref=$(gen $L/Aold_G${g}_r1.log p01.txt); got=$(gen $f p01.txt)
  if [ "$ref" = "$got" ]; then t=same; elif [ "${got%"$ref"}" != "$got" ]; then t=same+1st; else t=DIFF; fi
  echo "$b: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $f | grep -o '[0-9.]* TPS' | tr '\n' ' ')rss=$(grep -h -o 'Max Resident Set Size: [0-9]*' $f | grep -o '[0-9]*$')KB text=$t"
done | tee $L/speed.txt
want "every Qold text == Aold r1 of its G" "$(grep '^Qold' $L/speed.txt | grep -vc 'text=same$')" 0
echo "--- Qnew vs Anew (one PD keeps the hybrid's bits under the new config too)"
n=0; for f in $L/Qnew_G*.log; do b=$(basename $f .log); a=$L/Anew_${b#Qnew_}.log
  grep -q '^generation:' $f && grep -q '^generation:' $a || continue
  [ "$(gen $f p01.txt)" = "$(gen $a p01.txt)" ] || { echo "DIFF $b"; n=$((n + 1)); }; done
want "every Qnew text == Anew of the same G and run" $n 0
echo "--- PPL (prompt = prefill PPL; decode forced on Aold's continuation)"
for f in $L/ppl_*.log; do echo "$(basename $f .log): $(grep -h -o '\[PPL\] prompt .*\|\[PPL\] decode tokens=.*source=[a-z]*' $f | tr '\n' ' ')"; done | tee $L/ppl.txt
echo "--- texts (G=64 r1, for the approval table)"
for v in Aold Anew Qold Qnew CPU; do f=$L/${v}_G64_r1.log; [ -f $f ] || f=$L/${v}_G64.log
  echo "## $v"; gen $f p01.txt; echo; done | tee $L/texts.txt
for v in Anew Qnew; do [ -f $L/void_$v ] && echo "VOID $v: $(cat $L/void_$v)"; done
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z') expectation mismatches (this invocation): $MIS"
