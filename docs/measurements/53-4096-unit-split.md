# Measurement 53: #23 B's exact v79 tree vs `hvx_impl` at 4096 on `R3CY10WM83Y` — unit/session or code?

Branch `hvx/53-4096-unit-split` (DSP tree `87186bb3` = `hvx_impl` @ `05d99341`; DSP sources identical to
`202278a4`: `git diff 202278a4 87186bb3 --stat -- nntrainer/tensor/hexagon/htp` is empty) — estimated
device time: **20 min device-active** (four 4096 speed runs ≈ 12 min + two 4096 `--eval` ≈ 8 min),
**≈ 40 min wall** with the two 10 min idles. No source change on this branch; v79 only; 4096 only.

> **Separate sitting from #25 H1.** This handoff exists to compare two skels from one controlled thermal
> state. Do **not** run it in the same sitting as `docs/measurements/25-decode-prefetch.md` (≈ 30 min of
> sustained load, four skels, 4096 rows). Either run it on another day, or only after **≥ 10 min of phone
> idle following #25's last run**, and write the battery temperature (`adb shell dumpsys battery`, tenths
> of °C) into the result table **before O1 starts**. If that is not possible, stop and run it later — a
> hot start makes outcome 3 (inconclusive) a property of the session, not of the code.

## Why
On `R3CY10WM83Y` the 4096 row reads +5–6 % DSP Mcyc/step and −5–7 % prefill tok/s against #23 B
(`R3CY205ZMND`, 2026-09-17) on two skels and two days, while 512 / 1024 sit within 2 %; the DSP diff since
#23 has no context-proportional change (issue #53, HEXAGON.md §7 rule 9(e)). Running #23 B's exact tree
(S1) next to `hvx_impl` (S2) on this unit decides whether the #23 4096 cells get re-baselined to this
unit (outcome 1) or a regression between `e086248a` and `hvx_impl` is bisected (outcome 2); every later
4096 verdict (#25, #26, the ≥ 27.5 tok/s @ 4096 goal) gates on that answer.

## Artifacts

Both skels must come from the **workstation's** toolchain (SDK 6.4.0.1, hexagon-clang 19.0.04 `toolv19`)
so that toolchain is not a variable. The skel link is not byte-reproducible (HEXAGON.md / gates note:
relocation order differs per link, code and data identical), so a rebuilt skel has a **new md5 at the
documented size**; write the md5 you get and check the **size** — a wrong size means a wrong tree or SDK.

| file | md5 (size) | built with |
|---|---|---|
| `build_hexagon/skel/libnntr_htp_skel.v79-23b.so` (**S1**, the #23 tree) | `aba0126e848f9d993ef72ecfbf6fcab7` (**50,920 B**) if the #23 B binary is still on the workstation; else the rebuild's md5, same size | `HEX_ARCH=v79 ./tools/hexagon/build_skel.sh` @ `e086248a` (#23 B, `docs/measurements/23-sdk64-baseline.md`) |
| `build_hexagon/skel/libnntr_htp_skel.v79-head.so` (**S2**, `hvx_impl`) | rebuild — expect **50,984 B** (container 50,792 B + the 192 B SDK 6.4.0.1 delta seen in #35). **Not** `8d8cdb26…` / 50,920 B: that is #35 B1's binary at `305c51f1`, pre-#50. **Not** any `libnntr_htp_skel.{A,B,C,D}.so` from #25 (≈ 51.3 / 55.4 KB on the workstation, `HTP_MM_*` variants of a different tree) | `HEX_ARCH=v79 ./tools/hexagon/build_skel.sh` @ `87186bb3` (this branch; DSP sources = `202278a4`) |
| `build_hexagon/host/hexagon_e2e_test`, `hexagon_rpc_test` | rebuild on this branch; record md5 | `./tools/hexagon/build_host_test.sh` @ `87186bb3` |
| `/tmp/qwen3_full4k.hexw` / `.hexcfg` | `e5c92fad696fb35405918941d1062be3` / `b0d8aca9a6c7626b0af8dc81e1de59eb` | `nntr_hexpack … --max-seq 4224` (P4; packer unchanged; ABI v4 at `e086248a` and HEAD, so S1 loads it as is) |
| `/tmp/t4096.i32` | `29684fbd372495912951cb3cc0d25e20` | existing, P4 run (#24 table) |
| Mac-side reference, **not used on the device**: container v79 skel @ `87186bb3` | `03baa64ebe1b54dbc910aeaac43a4bbc` (50,792 B), SDK 6.4.0.2 | `HEX_ARCH=v79 tools/docker/run.sh ./tools/hexagon/build_skel.sh` (gates rung 4) |
| Mac-side reference, not used: container `hexagon_e2e_test` / `hexagon_rpc_test` | `88996fee9a57113cf47793d877e4ae19` (1,380,160 B) / `dd05aaab4cbe916c2efabfb9b6757678` (528,168 B) | `tools/docker/run.sh ./tools/hexagon/build_host_test.sh` |

## Steps (workstation, phone `R3CY10WM83Y` on USB, single device)

```bash
git fetch && git checkout hvx/53-4096-unit-split            # do not rebase; commit the result on this branch
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/setup_sdk_env.source; export ANDROID_NDK=/opt/android-ndk-r26d
# 1. S1: is #23 B's binary still here? (look for aba0126e…, 50,920 B); else rebuild the #23 tree
md5sum build_hexagon/skel/*.so; ls -l build_hexagon/skel/*.so
cp build_hexagon/skel/<the aba0126e file> build_hexagon/skel/libnntr_htp_skel.v79-23b.so   # if present, else:
git worktree add /tmp/nntr-23b e086248a && (cd /tmp/nntr-23b && HEX_ARCH=v79 ./tools/hexagon/build_skel.sh && git rev-parse HEAD) \
  && cp /tmp/nntr-23b/build_hexagon/skel/libnntr_htp_skel.so build_hexagon/skel/libnntr_htp_skel.v79-23b.so   # expect 50,920 B
# 2. S2 + harness at this branch
HEX_ARCH=v79 ./tools/hexagon/build_skel.sh && cp build_hexagon/skel/libnntr_htp_skel.so build_hexagon/skel/libnntr_htp_skel.v79-head.so   # expect 50,984 B
./tools/hexagon/build_host_test.sh
md5sum build_hexagon/skel/libnntr_htp_skel.v79-23b.so build_hexagon/skel/libnntr_htp_skel.v79-head.so build_hexagon/host/hexagon_e2e_test build_hexagon/host/hexagon_rpc_test /tmp/t4096.i32
ls -l build_hexagon/skel/libnntr_htp_skel.v79-23b.so build_hexagon/skel/libnntr_htp_skel.v79-head.so          # sizes 50,920 / 50,984
./tools/hexagon/run_device_test.sh                          # RPC_TEST PASS (S2 is the current libnntr_htp_skel.so; pushes the new harness)
# 3. helpers: `use` selects a skel and prints its md5; `temp` = battery temperature (tenths of °C) / level
use()  { cp build_hexagon/skel/libnntr_htp_skel.$1.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so; }
temp() { date +%H:%M; adb shell dumpsys battery | grep -E 'level|status|temperature'; }
run4k() { ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64; }
# 4. O1 — phone idle >= 10 min first (screen off, USB attached, nothing running, >= 10 min after any #25 run)
temp; use v79-23b  && run4k          # S1(O1)  slot 1
temp; use v79-head && run4k          # S2(O1)  slot 2
# 5. O2 — idle >= 10 min again, then the reverse order
temp; use v79-head && run4k          # S2(O2)  slot 1
temp; use v79-23b  && run4k          # S1(O2)  slot 2
# 6. accuracy, after the speed runs (they must not heat them): one 4096 --eval per skel
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --eval          # S1 (still selected)
use v79-head && ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --eval   # S2
adb shell md5sum /data/local/tmp/nntr_htp/libnntr_htp_skel.so                                 # the last skel that ran (S2)
# 7. per speed log (logs/hexagon/e2e_<stamp>.log): prefill tok/s and first-ten / last-ten decode medians (the harness prints only the 63-step median)
dec() { grep -E '^E2E step .* n=1 ' "$1" | sed -E 's/.* pcycles=([0-9]+).*/\1/'; }
for L in logs/hexagon/e2e_<stamp>.log; do echo "$L"; grep -E '^E2E step .* n=128 ' $L | sed -E 's/.* us=([0-9]+).*/\1/' | awk '{s+=$1} END{printf "prefill tok/s %.2f\n",4096/(s/1e6)}'; dec $L | head -10 | sort -n | awk '{v[NR]=$1} END{printf "first10 median %.3f Mcyc\n",(v[5]+v[6])/2e6}'; dec $L | tail -10 | sort -n | awk '{v[NR]=$1} END{printf "last10 median %.3f Mcyc\n",(v[5]+v[6])/2e6}'; grep -E '^E2E (decode|wall_ms)' $L; done
```
Lines to look for: `RPC_TEST PASS`; per speed run `E2E init ok weights=… kv=… act=… n_ops=451`, 32 × `E2E step … n=128 …`,
63 × `E2E step … n=1 pcycles=<c> us=<t> …`, `E2E gen …`,
`E2E decode steps=63 median_us=<u> median_pcycles=<c> pcycles_per_us=<r>`, `E2E wall_ms <w>`; per `--eval`:
`E2E ppl <x> steps 4095 top1 <n> wall_ms <w>` then the same `E2E decode` line. If a run hangs or the DSP
restarts, note the last `E2E step` line and the `logs/hexagon/device_farf_<stamp>.log` tail in the Notes,
`adb reboot`, idle ≥ 10 min, and redo that whole order (both slots), not just the one run. No `adb`
options beyond the above; insert the serial after the image path only if more than one device is attached.

## Results (fill in)

Reading: DSP Mcyc/tok = `median_pcycles` ÷ 1e6; decode ms/tok = `median_us` ÷ 1000; prefill tok/s from
step 7. References in brackets: **#23 B** 4096 = 34.4 tok/s / 87.4 ms / **177.6 Mcyc** (`R3CY205ZMND`,
2026-09-17, malloc harness); **#35 B1** = 32.14 / 32.71 (cooled re-run) tok/s, 92.274 / 89.103 ms,
**188.029 / 186.753 Mcyc** (this unit, 2026-09-18, `8d8cdb26…`).

| order | slot | skel | md5 (from `use`) | start hh:mm | idle min before | battery °C / level | prefill tok/s (ref 34.4 / 32.1–32.7) | decode median host ms (87.4 / 89.1–92.3) | DSP Mcyc/tok (177.6 / 186.8–188.0) | `pcycles_per_us` | first-ten / last-ten Mcyc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| O1 | 1 | S1 (#23 tree) | | | ≥ 10 | | | | | | / |
| O1 | 2 | S2 (`hvx_impl`) | | | 0 | | | | | | / |
| O2 | 1 | S2 (`hvx_impl`) | | | ≥ 10 | | | | | | / |
| O2 | 2 | S1 (#23 tree) | | | 0 | | | | | | / |

| skel | ctx | token file | --eval PPL (ref) | top-1 (ref) | note |
|---|---|---|---|---|---|
| S1 (#23 tree) | 4096 | t4096.i32 (29684fbd…) | (1.5687 = #23 B) | (3758) | expected digit for digit: same tree, same image |
| S2 (`hvx_impl`) | 4096 | t4096.i32 | (1.5623 = #35 B1) | (3759) | expected digit for digit: #50 is validator-only |

`adb shell md5sum` after the last run: ` ` (must equal the S2 md5).

### Verdict (plan §1; read slot-matched: `S1(O1)` vs `S2(O2)` = slot 1, `S2(O1)` vs `S1(O2)` = slot 2)

`Δ_slot = (S2 − S1) / S1` in DSP Mcyc, per slot. Prefill tok/s is recorded with the same slot rule
(band ±3 %) but does not decide (host-wall based; #24 showed this unit's host clock moves on its own).

| outcome | gate (Mcyc, both slots) | consequence |
|---|---|---|
| **1 Unit / session** | `|Δ| ≤ 2 %` in both slots, i.e. S1 reads ≈ 186–188 like S2 | #23's 4096 cells re-baselined to this unit from the four readings; cross-unit 4096 band set from them; benchmark 4096 goal row cites the within-unit number; rule 9(e) closes; close #53 |
| **2 Code / build** | S1 ≤ 181.2 Mcyc (within 2 % of #23 B) **and** S2 ≥ 4 % above S1 in both slots | real regression between `e086248a` and `hvx_impl`; follow-up issue bisects with a v79 skel at `1bbd8d53`, then per commit (`813a6b7a` #35, `e77a2275` #50, `7c8efb61` #33) |
| **3 Inconclusive** | anything else (S1 ≈ 178 with S2 only 2–4 % above; slots disagree in sign; same skel differs > 2 % between its two slots) | one more O1 + O2 pair on a second day before any re-baseline; the 4096 cell stays "within one unit and session only" |

Also void (re-run, do not read): a skel md5 in the table that is not the one in the artifact rows / sizes
other than 50,920 / 50,984 B; a `--eval` line that is not digit-for-digit its reference (then the skel
is not the tree it claims to be).

## Hand back
1. Fill the two tables, the `adb shell md5sum` line and the Notes (SDK version, `hexagon-clang --version`
   if S1 was rebuilt and its size is not 50,920 B, the `git rev-parse HEAD` of `/tmp/nntr-23b`, phone
   state before O1).
2. `git add docs/measurements/53-4096-unit-split.md && git commit -s -m "[docs] Fill the #53 4096 unit-split measurement (R3CY10WM83Y, SDK 6.4.0.1)" && git push`
3. `gh issue edit 53 -R dlwlzzero/nntrainer --remove-label state:needs-measurement --add-label state:measured`
4. `git worktree remove /tmp/nntr-23b` (if created).

## Notes from the run
<phone state before O1 (minutes since the last #25 run, battery °C), S1 provenance (stored binary or
rebuild + worktree HEAD), anything odd: thermal, FARF errors, stale-file pushes, DSP restarts>
