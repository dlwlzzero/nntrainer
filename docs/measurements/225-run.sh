#!/usr/bin/env bash
##
# @file    225-run.sh
# @brief   #225's device sitting (plan 225 section 4 step 7): the LFM2.5 table
#          of record, 9 cells x 3 cases on the config of record with the FC WH
#          sidecar (handoff docs/measurements/225-fcwh-table.md)
#
# Staged beside 225-stage.sh's set as run_225.sh (handoff step 2); run it from
# there, phone on USB (farm session):
#   bash /local/mnt/workspace/htp_moe/225/run_225.sh <serial> [v79|v81]
# The serial is required (no default); arch defaults to v79 (S25 Ultra).
# Variants (one binary set; config and env only):
#   A     CPU only: the q40 model, its own config (init_seq_len 1024 at P1024
#         only, as #222 did); writes the PPL block's continuations
#   Aoff  NPU model, the config of record minus the three engine keys
#         (cfg_off.json): no HTP prefill FC; P512 x G64 / G512 + PPL only
#   B     hybrid, the config of record (cfg_new.json)
#   Bfb   B's fallback (user decision (b), 2026-10-06): cfg_fb.json =
#         attn_proj_engine and dense_ffn_engine cpu, conv_block_engine htp.
#         Run in a B cell's place once B did not load at that prompt length
#         (logs/void_B_P<p>); recorded as the hybrid case
#   Q     B + NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28 (one PD; model file
#         pre-read + page_cache_evict -1, as #201 / #216 / #222)
# Prompts: P64 = p64.txt (p01's first 64 tokens), P512 = p01.txt,
# P1024 = p1024.txt (p01 + p07 + p05 + p03 cut at 1024 tokens).
# Order per prompt length (cool start per block, zone0 <= 35 C):
#   G64 A B Q, cool, Q B (mirrored second runs); G512 A B Q; G1024 A B Q;
#   Aoff after P512's G64 and G512 blocks
#   profiles: B and Q at P512 G64, NNTR_HTP_PROFILE=2 (not speed cells)
#   PPL: 8 prompts at G 256, NNTR_PPL=1 + NNTR_PPL_DECODE (A writes its
#        continuation; A p01 once more forced = the null check; Aoff, B, Q
#        forced on it)
# A non-A case that does not load or dies is VOID (logs/void_<v>_P<p>: its
# error, arena / heap lines); its other cells at that prompt length are
# skipped and the sitting goes on. RESUMABLE per run (logs/done/<run>);
# after a STOP reboot the phone and run the same command again.
set -u -o pipefail
S=${1:?usage: run_225.sh <serial> [v79|v81]}
ARCH=${2:-v79}
case $ARCH in v79 | v81) ;; *) echo "arch must be v79 or v81" >&2; exit 1 ;; esac
W=$(cd "$(dirname "$0")" && pwd); L=$W/logs; X=$W/extra; mkdir -p $L/done
C=/data/local/tmp/nntrainer/causallm; D=$C/s225; M=../models/q40-qs4cx-wh; MC=../models/q40
MF=$M/nntr_lfm2_8b_a1b_q40_arm.bin; FCWH=nntr_lfm2_8b_a1b_q40_arm_fcwh.bin
AD="adb -s $S"
# the files beside staged md5.txt (dd24b4fc...): handoff step 2 copies the
# prompts and loop_check.py, this runner derives the two configs
XMD5="c0d3e9ffb0c9565e51fa044c53ac71d7  p64.txt
2e47c5f45f538babcc7b4a7bb48a4e70  p1024.txt
01b492ad749d2a315ae242a276e886b6  cfg_off.json
49fcc38aa665a3f4c6825d937ff33eb5  cfg_fb.json"
exec > >(tee -a $L/sitting.out) 2>&1
MIS=0
stop() { echo "STOP: $*"; echo "  -> reboot the phone and run the same command again: it resumes at the first unfinished run"; exit 1; }
want() { if [ "$2" = "$3" ]; then echo "OK  $1 = $2"; else echo "BAD $1: got '$2' want '$3'"; MIS=$((MIS + 1)); fi; }
therm() { echo "$1 $(date +%H:%M:%S) $($AD shell 'dumpsys battery | grep -E "^  (level|temperature)"; cat /sys/class/thermal/thermal_zone0/temp' | tr -d '\r' | tr '\n' ' ')" | tee -a $L/therm.log; }
zone0() { $AD shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
cool() { local i z; for i in $(seq 1 40); do z=$(zone0); [ "$z" -le 35000 ] && break
    echo "zone0=$z > 35000, waiting 30 s ($i/40)"; sleep 30; done; echo "block start zone0=$(zone0)"; }
pf() { case $1 in 64) echo p64.txt ;; 512) echo p01.txt ;; 1024) echo p1024.txt ;; esac; }
pp() { [ -f $X/$1 ] && echo $X/$1 || echo $W/app/$1; } # a prompt file on the workstation
# the generated text: what follows the prompt's echo, before the ===== box
gen() { python3 - "$1" "$(pp $2)" <<'PY'
import sys
log = open(sys.argv[1], errors="replace").read().split("\n=====")[0]
log = "\n".join(l for l in log.split("\n") if not l.startswith(("[HTP", "[PPL]")))
p = open(sys.argv[2], errors="replace").read().rstrip("\n")
i = log.rfind(p)
print(log[i + len(p):].strip("\n") if i >= 0 else "<prompt echo not found>")
PY
}
# the hybrid case at prompt length $1: B, or Bfb once B was VOID there
hyb() { [ -f $L/void_B_P$1 ] && echo Bfb || echo B; }
run() { # run <A|Aoff|B|Bfb|Q> <P> <G> <log> <prompt file> [env ...]
  local v=$1 p=$2 g=$3 log=$4 pr=$5 e="" pre="" m=$M cfg=""; shift 5
  [ -f $L/done/$log ] && { echo "$log: done earlier, skipped"; return 0; }
  [ -f $L/void_${v}_P$p ] && { echo "$log: VOID, $v did not load at P$p earlier ($(head -1 $L/void_${v}_P$p))"; return 0; }
  case $v in
    A) m=$MC; cfg="cp nntr_config.json.225 nntr_config.json"
       [ $p = 1024 ] && cfg="$cfg && sed -i 's/\"init_seq_len\": [0-9]*/\"init_seq_len\": 1024/' nntr_config.json"
       cfg="(cd $m && $cfg) &&" ;;
    Aoff) cfg=cfg_off.json ;;
    B | Q) cfg=cfg_new.json ;;
    Bfb) cfg=cfg_fb.json ;;
  esac
  [ $v != A ] && cfg="cp $cfg $m/nntr_config.json &&"
  [ $v = Q ] && { e="NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28"; pre="cat $MF > /dev/null && ./page_cache_evict $MF -1 &&"; }
  $AD logcat -c
  $AD shell "cd $D && $cfg sed -i -e 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $g/' \
    -e 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' $m/nntr_config.json && \
    md5sum libnntr_hvx_skel.so && grep -h -E 'init_seq_len|num_to_generate|_engine' $m/nntr_config.json | tr -d ' \n' && echo && \
    $pre $e $* NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
    ./nntrainer_causallm $m \"\$(cat $pr)\"" > $L/$log.log 2>&1
  $AD logcat -d | grep -iE 'adsprpc|fastrpc|nntr_hvx|token' > $L/$log.logcat || true
  grep -qi '0x8000040e' $L/$log.log && stop "$log: 0x8000040e (stale skel or stub)"
  if ! grep -q '^generation:' $L/$log.log; then
    [ $v = A ] && stop "$log: the CPU reference did not generate ($(grep -h -m1 -E 'FATAL|rror' $L/$log.log | cut -c1-200))"
    # a case that does not load (or dies) at this prompt length is a
    # measured outcome: recorded with its arena / heap lines, its other
    # cells at this length skipped, the sitting goes on
    { grep -h -m1 -E 'FATAL|failed|refused|no room|ENOMEM|0x80000402' $L/$log.log | cut -c1-240 || echo "no generation line, no error line (see $log.log / .logcat)"
      grep -h -E 'arena|fc wh:|e2e: fc arena|heap|mapped|rpcmem' $L/$log.log | sed 's/^/    /'; } > $L/void_${v}_P$p
    echo "VOID $log:"; sed 's/^/  /' $L/void_${v}_P$p
    touch $L/done/$log
    # user decision (b): the hybrid that cannot hold the overflow re-runs
    # with attn_proj / dense_ffn on the CPU, as the hybrid case
    if [ $v = B ]; then echo "  -> B's fallback Bfb (cfg_fb.json) for this cell and B's other cells at P$p"
      run Bfb $p $g ${log/B/Bfb} $pr "$@"; fi
    return 0
  fi
  echo "$log: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $L/$log.log | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -h -o 'calls/token=[0-9.]*' $L/$log.log) rss=$(grep -h -o 'Max Resident Set Size: [0-9]*' $L/$log.log | grep -o '[0-9]*$')KB $(grep -h -o '\[PPL\] prompt .*\|\[PPL\] decode tokens=[^ ]* nll/token=[^ ]* ppl=[^ ]* top1=[^ ]*' $L/$log.log | tr '\n' ' ')"
  [ "${log#ppl_}" = "$log" ] && want "$log prefill tokens" "$(sed -n 's/^prefill: \([0-9]*\) tokens.*/\1/p' $L/$log.log)" $p
  if [ $v = A ]; then
    want "$log no HTP banner" "$(grep -c 'moe m1 gemv\|dspq: on' $L/$log.log)" 0
    touch $L/done/$log; return 0
  fi
  want "$log one moe m1 gemv banner" "$(grep -c 'moe m1 gemv: on (applied=' $L/$log.log)" 1
  local fw; fw=$(grep -h '\[HTP\] fc wh:' $L/$log.log)
  [ -n "$fw" ] && echo "    $fw"
  case $v in
    Aoff)
      want "$log hybrid (dspq on, no token driver)" "$(grep -c '\[HTP\] dspq: on' $L/$log.log)/$(grep -c 'token driver: on' $L/$log.log)" 1/0
      want "$log no fc wh banner (no key opens the sidecar)" "$(grep -c '\[HTP\] fc wh:' $L/$log.log)" 0 ;;
    B | Bfb)
      want "$log hybrid (dspq on, no token driver)" "$(grep -c '\[HTP\] dspq: on' $L/$log.log)/$(grep -c 'token driver: on' $L/$log.log)" 1/0
      want "$log fc wh banner: the sidecar, requant=0" "$(grep -c "fc wh: file=[^ ]*$FCWH .* requant=0$" $L/$log.log)" 1 ;;
    Q)
      want "$log fc wh banner: the sidecar, heap_kib=0 requant=0" "$(grep -c "fc wh: file=[^ ]*$FCWH .* heap_kib=0 requant=0$" $L/$log.log)" 1
      want "$log wh_handles on the bind line" "$(grep -c 'graph: q4m1 weights=.* wh_handles=[1-9]' $L/$log.log)" 1
      want "$log close clean" "$(grep -c 'token driver: close tokens=[0-9]* hops/token=0.00 .* timeouts=0 stale=0 .* id_mismatch=0 ' $L/$log.log)" 1
      want "$log arena unmapped" "$(grep -c 'e2e: close .* unmap_fail=0 detach_fail=0 ' $L/$log.log)" 1
      want "$log calls/token=1.00" "$(grep -cF 'calls/token=1.00' $L/$log.log)" 1
      local t sk hu mm; t=$(sed -n 's/.*graph: forward calls=[0-9]* tokens=\([0-9]*\) .*/\1/p' $L/$log.log)
      sk=$(sed -n 's/.*graph: cpu fc skipped=\([0-9]*\)$/\1/p' $L/$log.log)
      want "$log cpu fc skipped = 32 x tokens" "${sk:-none}" "$((32 * ${t:-0}))"
      hu=$(sed -n 's/.*e2e: close .* heap_used_kib=\([0-9]*\) .*/\1/p' $L/$log.log)
      want "$log heap_used_kib < #222's 223003" "$([ "${hu:-999999}" -lt 223003 ] && echo yes || echo "no (${hu:-none})")" yes
      mm=$(sed -n 's/.*e2e: fc arena .* mapped_mib=\([0-9]*\) .*/\1/p' $L/$log.log)
      want "$log mapped_mib <= the 3840 MiB ceiling" "$([ "${mm:-9999}" -le 3840 ] && echo yes || echo "no (${mm:-none})")" yes
      grep -h -E 'graph: q4m1 weights=|e2e: fc arena|e2e: close|token driver: pool' $L/$log.log | sed 's/^/    /' ;;
  esac
  touch $L/done/$log; }

echo "=== 225 $(date '+%F %T %Z') unit=$S arch=$ARCH (done: $(ls $L/done | wc -l) runs)"
$AD get-state > /dev/null 2>&1 || stop "unit $S not attached"
echo "model: $($AD shell getprop ro.product.model | tr -d '\r') soc: $($AD shell getprop ro.soc.model | tr -d '\r')"
echo "uptime (reboot first, then wait 5 min: rule 61): $($AD shell cat /proc/uptime | tr -d '\r')"
$AD shell input keyevent 223 || true
therm t0
[ "$(cd $W && LC_ALL=C md5sum -c md5.txt | grep -vc ': OK$')" = 0 ] || stop "workstation set differs from md5.txt"
grep -q "\"fc_wh_file_name\": \"$FCWH\"" $W/app/cfg_new.json || stop "app/cfg_new.json does not name $FCWH"
sed '/"\(conv_block\|dense_ffn\|attn_proj\)_engine"/d' $W/app/cfg_new.json > $X/cfg_off.json
sed 's/"\(dense_ffn\|attn_proj\)_engine": "htp"/"\1_engine": "cpu"/' $W/app/cfg_new.json > $X/cfg_fb.json
diff <(echo "$XMD5") <(cd $X && md5sum p64.txt p1024.txt cfg_off.json cfg_fb.json) > /dev/null || stop "extra/ differs from the runner's md5s (handoff step 2)"
[ -f $X/loop_check.py ] || stop "extra/loop_check.py missing (handoff step 2)"
$AD shell "mkdir -p $D" && $AD push $W/app/. $D/ > /dev/null && $AD push $X/p64.txt $X/p1024.txt $X/cfg_off.json $X/cfg_fb.json $D/ > /dev/null
$AD shell "cd $D && cp libnntr_hvx_skel.$ARCH.so libnntr_hvx_skel.so && chmod 755 nntrainer_causallm page_cache_evict"
$AD shell "cd $D && md5sum $(grep -o '  app/.*' $W/md5.txt | sed 's|  app/||' | tr '\n' ' ')" | tr -d '\r' | awk '{print $1"  app/"$2}' > $L/md5_device.log
diff <(grep -E '^[0-9a-f]{32}  app/' $W/md5.txt | sort -k2) <(sort -k2 $L/md5_device.log) || stop "device app md5 differs"
diff <(echo "$XMD5") <($AD shell "cd $D && md5sum p64.txt p1024.txt cfg_off.json cfg_fb.json" | tr -d '\r') || stop "device extra md5 differs"
# the sidecar beside the main bin (fc_wh_file_name is read from the model dir); pushed once
want_fc=$(grep "  model/$FCWH" $W/md5.txt | cut -d' ' -f1)
[ "$($AD shell md5sum $D/$M/$FCWH 2>/dev/null | cut -d' ' -f1)" = "$want_fc" ] || { echo "pushing the sidecar (228 MB)"; $AD push $W/model/$FCWH $D/$M/ > /dev/null; }
# the config of record's model_file_name: a symlink to the same bin (#222, idempotent)
$AD shell "cd $D/$M && ln -sf nntr_lfm2_8b_a1b_q40_arm.bin nntr_lfm2.5_8b_a1b_q40_arm.bin && ls -l nntr_lfm2*.bin && \
  md5sum nntr_lfm2_8b_a1b_q40_arm.bin nntr_lfm2.5_8b_a1b_q40_arm.bin $FCWH $D/$MC/nntr_lfm2_8b_a1b_q40_arm.bin && \
  md5sum $D/cfg_new.json $D/cfg_off.json $D/cfg_fb.json" | tr -d '\r' | tee $L/md5_models.log
grep -q "^$want_fc  $FCWH" $L/md5_models.log && echo "MD5 OK (app, extra, sidecar)" || stop "device sidecar md5 differs from md5.txt"
for m in $M $MC; do
  $AD shell "cd $D/$m && ([ -f generation_config.json.225 ] || cp generation_config.json generation_config.json.225) && \
    ([ -f nntr_config.json.225 ] || cp nntr_config.json nntr_config.json.225) && \
    sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && grep -H do_sample generation_config.json" | tee -a $L/config.log
done
$AD shell "cd $D/$MC && grep -H -E 'bad_word_ids|init_seq_len' nntr_config.json.225" | tee -a $L/config.log

for p in 64 512 1024; do pr=$(pf $p)
  echo "--- P$p G=64 $(date +%H:%M:%S)"; cool
  run A $p 64 A_P${p}_G64_r1 $pr
  v=$(hyb $p); run $v $p 64 ${v}_P${p}_G64_r1 $pr
  run Q $p 64 Q_P${p}_G64_r1 $pr
  cool
  run Q $p 64 Q_P${p}_G64_r2 $pr
  v=$(hyb $p); run $v $p 64 ${v}_P${p}_G64_r2 $pr
  [ $p = 512 ] && run Aoff 512 64 Aoff_P512_G64_r1 $pr
  for g in 512 1024; do
    echo "--- P$p G=$g $(date +%H:%M:%S)"; cool
    run A $p $g A_P${p}_G${g}_r1 $pr
    v=$(hyb $p); run $v $p $g ${v}_P${p}_G${g}_r1 $pr
    run Q $p $g Q_P${p}_G${g}_r1 $pr
    [ $p = 512 ] && [ $g = 512 ] && run Aoff 512 512 Aoff_P512_G512_r1 $pr
  done
  therm t_P$p
done
echo "--- profiles (not speed cells): P512 G64, NNTR_HTP_PROFILE=2"; cool
v=$(hyb 512); run $v 512 64 prof_$v p01.txt NNTR_HTP_PROFILE=2
run Q 512 64 prof_Q p01.txt NNTR_HTP_PROFILE=2
for f in $L/prof_*.log; do [ -f $f ] && grep -q '^generation:' $f || continue
  b=$(basename $f .log)
  r=""; for k in conv dense FC; do grep -qE "M>1 $k +calls=[1-9]" $f && r="$r$k "; done
  echo "    $b M>1 rows present: ${r:-none} (B: conv dense FC; Bfb: conv)"
  grep -h -E 'convert to registry|M>1|M==1|graph: tokens=|pcyc/token|per kind' $f | sed 's/^/    /'
done
echo "--- PPL: 8 prompts at G=256 (prefill PPL + decode PPL forced on A's continuation)"; cool
for i in 1 2 3 4 5 6 7 8; do
  [ -f $L/done/ppl_A_p0$i ] || $AD shell "rm -f $D/cont_p0$i.ids"
  run A 512 256 ppl_A_p0$i p0$i.txt NNTR_PPL=1 NNTR_PPL_DECODE=cont_p0$i.ids
  [ $i = 1 ] && run A 512 256 ppl_Af_p01 p01.txt NNTR_PPL=1 NNTR_PPL_DECODE=cont_p01.ids
  for v in Aoff $(hyb 512) Q; do run $v 512 256 ppl_${v}_p0$i p0$i.txt NNTR_PPL=1 NNTR_PPL_DECODE=cont_p0$i.ids; done
done
$AD shell "cd $D && cp cfg_new.json $M/nntr_config.json && cp $M/generation_config.json.225 $M/generation_config.json && \
  cp $MC/nntr_config.json.225 $MC/nntr_config.json && cp $MC/generation_config.json.225 $MC/generation_config.json"
echo "device configs: $M = the config of record (pristine), $MC and both generation configs restored"
therm t_end

echo "--- speed (prefill / decode / last 64 TPS, peak RSS; text vs A r1 of the same P and G: same | same+1st (one of the two drops its first token: init_seq_len) | DIFF)"
for f in $L/[ABQ]*_P*_G*.log; do b=$(basename $f .log); p=${b#*_P}; p=${p%%_*}; g=${b#*_G}; g=${g%%_*}
  grep -q '^generation:' $f || { echo "$b: VOID"; continue; }
  ref=$(gen $L/A_P${p}_G${g}_r1.log $(pf $p)); got=$(gen $f $(pf $p))
  if [ "$ref" = "$got" ]; then t=same; elif [ "${got%"$ref"}" != "$got" ] || [ "${ref%"$got"}" != "$ref" ]; then t=same+1st; else t=DIFF; fi
  echo "$b: $(grep -h -E '^(prefill|generation|generation\(last 64\)):' $f | grep -o '[0-9.]* TPS' | tr '\n' ' ')rss=$(grep -h -o 'Max Resident Set Size: [0-9]*' $f | grep -o '[0-9]*$')KB text=$t"
done | tee $L/speed.txt
echo "--- G64 run 2 == run 1 (the same binary and config are deterministic)"
n=0; for f in $L/[BQ]*_G64_r2.log; do r1=${f%_r2.log}_r1.log; p=$(basename $f); p=${p#*_P}; p=${p%%_*}
  grep -q '^generation:' $f && grep -q '^generation:' $r1 || continue
  [ "$(gen $f $(pf $p))" = "$(gen $r1 $(pf $p))" ] || { echo "DIFF $(basename $f .log)"; n=$((n + 1)); }; done
want "every r2 text == its r1" $n 0
echo "--- loops (tools/htp/loop_check.py; a variant fails a prompt only where it loops and A does not)"
for p in 64 512 1024; do ls $L/[ABQ]*_P${p}_G*.log > /dev/null 2>&1 || continue
  python3 $X/loop_check.py --prompt $(pp $(pf $p)) $(for f in $L/[ABQ]*_P${p}_G*.log; do grep -q '^generation:' $f && echo $f; done)
done | tee $L/loops.txt
for i in 1 2 3 4 5 6 7 8; do python3 $X/loop_check.py --prompt $W/app/p0$i.txt $(for f in $L/ppl_*_p0$i.log; do grep -q '^generation:' $f && echo $f; done); done | tee -a $L/loops.txt
echo "--- PPL (prompt = prefill PPL; decode forced on A's continuation; recorded, not gated: T2)"
for f in $L/ppl_*.log; do echo "$(basename $f .log): $(grep -h -o '\[PPL\] prompt .*\|\[PPL\] decode tokens=.*source=[a-z]*' $f | tr '\n' ' ')"; done | tee $L/ppl.txt
echo "--- texts (G=64 r1, for the approval table)"
for p in 64 512 1024; do for v in A Aoff B Bfb Q; do f=$L/${v}_P${p}_G64_r1.log; [ -f $f ] && grep -q '^generation:' $f || continue
  echo "## $v P$p"; gen $f $(pf $p); echo; done; done | tee $L/texts.txt
echo "--- VOID"; for f in $L/void_*; do [ -f $f ] && { echo "$(basename $f):"; sed 's/^/  /' $f; }; done
echo "--- therm"; cat $L/therm.log
echo "=== done $(date '+%F %T %Z') expectation mismatches (this invocation): $MIS"
