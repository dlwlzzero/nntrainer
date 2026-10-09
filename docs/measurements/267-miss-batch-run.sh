#!/usr/bin/env bash
##
# @file    267-miss-batch-run.sh
# @brief   #267 L3 sitting (plan 267 S2, minimal): L = htp_decode @ 94ba7a706
#          + miss-path batching (HEXKL_MOE_FLAG_ROWS_OUT); Gemma-4 26B QS4CX,
#          C 16, S25 R3CY205ZMND. A = the cells already on record (no re-run)
#
# Reuses 267-run.sh (and through it 260-e-run.sh's cell / cool start / S1
# ceiling); the variant is L, the binaries live in s267m/.
#
# usage:
#   267-miss-batch-run.sh install <app dir> <v79 skel>   (device go from the coordinator)
#   267-miss-batch-run.sh one <log dir> <prompt> <G> [suffix] [extra env]
#   267-miss-batch-run.sh sum <log dir> <ref log dir>   text vs the ref's
#                         E_p*_g*_D.log (267 sitting 1, ids == 260 r2 E)
set -u -o pipefail
HERE267=$(cd "$(dirname "$0")" && pwd)
# shellcheck disable=SC1090
source <(sed '/^case "${1:-}" in/,$d' "$HERE267/267-run.sh")
BIN[L]=$CR/s267m
XENV[L]=""

install_m() {
  local app=${1:?app dir} sk=${2:?skel} f d=$CR/s267m
  $AD shell "mkdir -p $d && cd $d && cp $CR/s260r2/libc++_shared.so $CR/s260r2/libsdkl.so $CR/s260r2/unittest_hvx_two_sessions ."
  for f in jni/libs/arm64-v8a/nntrainer_causallm jni/libs/arm64-v8a/libcausallm_core.so \
    jni/obj/local/arm64-v8a/libnntrainer.so jni/obj/local/arm64-v8a/libccapi-nntrainer.so; do
    $AD push "$app/$f" "$d/" >/dev/null
  done
  $AD push "$sk" "$d/libnntr_hvx_skel.so" >/dev/null
  $AD shell "chmod 755 $d/nntrainer_causallm"
  $AD shell "cd $d && md5sum nntrainer_causallm libcausallm_core.so libnntrainer.so libccapi-nntrainer.so libnntr_hvx_skel.so" | tr -d '\r'
}

sum_m() {
  python3 - "${1:?log dir}" "${2:?ref log dir}" <<'PY'
import glob, os, re, sys
L, ref = sys.argv[1:]
def g(rx, s):
    m = re.findall(rx, s)
    return m[-1] if m else "-"
def text(s):
    m = "</Text><turn|>\n<|turn>model"
    i = s.find(m)
    j = s.find("=================[ LLM", i)
    return re.sub(r"\[(HTP|PPL)\][^\n]*\n", "", s[i + len(m):j]) if i >= 0 else None
print("| cell | start / end bat zone0 | prefill tok/s (ms) | decode tok/s (last 64) | tokens | text == ref | misses/token | miss wait ms/token | moe calls/token (one-at-a-time) | MOE wall ms (net of wait) | FC | DENSE_FFN | ATTN_M1 | compute_mhz | mapped MiB | heap_used_kib | peak RSS KiB | S1 ceiling | nll |")
print("|" + "---|" * 19)
for f in sorted(glob.glob(os.path.join(L, "E_*.log"))):
    b = os.path.basename(f)[:-4]
    s = open(f, errors="replace").read()
    pf = re.search(r"prefill: (\d+) tokens, (\d+) ms, ([\d.]+) TPS", s)
    ge = re.search(r"generation: (\d+) tokens, (\d+) ms, ([\d.]+) TPS", s)
    l64 = g(r"generation\(last 64\): \d+ tokens, \d+ ms, ([\d.]+) TPS", s)
    rf = os.path.join(ref, re.sub(r"_L(_.*)?$", "", b) + "_D.log")
    t = text(s)
    same = "-"
    if "ppl" not in b and os.path.exists(rf) and t is not None:
        same = "yes" if t == text(open(rf, errors="replace").read()) else "**NO**"
    us = g(r"graph per-kind us/token:([^|]*)\|", s)
    kv = dict(re.findall(r"(\w+)=([\d.]+)", us)) if us != "-" else {}
    net = g(r"MOE=[\d.]+\(net of miss wait ([-\d.]+)\)", s)
    ms = lambda k: "%.2f" % (float(kv[k]) / 1000) if k in kv else "-"
    st = re.findall(r"start soc=\d+ bat=(\d+) zone0=(\d+)", s)
    en = re.findall(r"END soc=\d+ bat=(\d+) zone0=(\d+)", s)
    tt = lambda x: "%.1f / %.1f" % (int(x[0][0]) / 10, int(x[0][1]) / 1000) if x else "-"
    print("| %s | %s → %s | %s (%s) | %s (%s) | %s | %s | %s | %s | %s (%s) | %s (%s) | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        b, tt(st), tt(en), pf.group(3) if pf else "-", pf.group(2) if pf else "-",
        ge.group(3) if ge else "-", l64, ge.group(1) if ge else "-", same,
        g(r"misses/token=([\d.]+)", s),
        "%.1f" % (float(g(r"miss_wait_us/token=([\d.]+)", s)) / 1000) if "miss_wait_us" in s else "-",
        g(r"moe calls/token=([\d.]+)", s), g(r"one-at-a-time ([\d.]+)\)", s),
        ms("MOE"), "%.2f" % (float(net) / 1000) if net != "-" else "-",
        ms("FC"), ms("DENSE_FFN"), ms("ATTN_M1"), g(r"compute_mhz=(\d+)", s),
        g(r"mapped_mib=([\d.]+)", s), g(r"heap_used_kib=(\d+)", s),
        g(r"Max RSS \(KiB\): (\d+)", s), g(r"S1_CEILING (\S+)", s),
        g(r"nll/token=([\d.]+)", s)))
PY
}

case "${1:-}" in
install) shift; install_m "$@" ;;
one) shift; L=${1:?log dir}; mkdir -p "$L/done"; CLADDER=16 v L "$2" "$3" "${4:-}" "${5:-}" ;;
sum) shift; sum_m "$@" ;;
*) sed -n '2,18p' "$0"; exit 1 ;;
esac
