# Measurement 35: v79-native skel after the quantiser decode fix (B1 gate, B2 IEEE-vs-qf, A v75 control)

Branch `hvx/35-v79-skel` — DSP artifacts built in the container from `305c51f1` (the last commit that
changes DSP sources; later commits on the branch are tools/docs only) — estimated device time: **25 min**
(B1 + B2 ≈ 18 min; A, optional, ≈ 7 min)

## Why
Issue #35 replaced the quantiser's qf32 magic-add decode with an integer decode of the sf bits, which
fixes the four v79 simulator failures of #23 and makes the int16 rounding sign-symmetric on v75
(HEXAGON.md §7 rule 6). The simulators now pass 13/13 on both arches; this run decides whether the
v79 skel (B1) can become the shipping skel (its speed is 10–27 % above v75 per #23) — pass rules below —
and answers ledger ④ / §7 rule 1 with B2, a v79 skel built with the qf-format helpers instead of the
IEEE ones. A (v75) is **optional**: v79 is the primary arch since 2026-09-17 (the device is the S25
Ultra) and v75 is wrapped up; run A only if the 25 min allow — it is the last v75 control row and the
only silicon reading of the new decode on v75.

## 0. Machine setup (any Linux x86_64 box + S25 / S25 Ultra on USB; skip what you already have)

| need | how | check |
|---|---|---|
| Hexagon SDK 6.4.x with HEXAGON_Tools 19.0.04 | Qualcomm Software Center or `qpm-cli` (`tools/docker/setup_wizard.sh sdk` shows the commands) | `source <SDK>/setup_sdk_env.source && echo $DEFAULT_TOOLS_VARIANT` → `toolv19` |
| Android NDK r26d | https://dl.google.com/android/repository/android-ndk-r26d-linux.zip | `export ANDROID_NDK=<path>` |
| adb | `apt install android-tools-adb` | `adb devices` lists the phone as `device` |
| python3 + `transformers>=4.40 safetensors tokenizers sentencepiece numpy huggingface_hub` | `pip install ...` (tools/docker/Dockerfile) | — |
| gcc/g++, md5sum | distro packages | — |

```bash
git clone https://github.com/dlwlzzero/nntrainer && cd nntrainer
git checkout hvx/35-v79-skel            # do not rebase; commit the result on this branch
source <SDK>/setup_sdk_env.source; export ANDROID_NDK=<ndk path>
```

## 1. Model artifacts (only if `/tmp/qwen3_full*` and the token files are not already on the box)

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='models/hf', allow_patterns=['*.json','*.txt','model.safetensors'])"
python3 tools/hexagon/make_w8cx_bin.py models/hf models/nntr_qwen3_0.6b_w8cx_DEFAULT.bin   # md5 7562313bb4cc70410450d0ab3a4fa563
./tools/hexagon/build_host_x86.sh
./build_x86_hexagon/nntr_hexpack models/nntr_qwen3_0.6b_w8cx_DEFAULT.bin /tmp/qwen3_full                 # .hexw 5abf61bef086368559423daa3bea9a99 / .hexcfg 5926be5703f85531c9c46c1978c25287
./build_x86_hexagon/nntr_hexpack models/nntr_qwen3_0.6b_w8cx_DEFAULT.bin /tmp/qwen3_full4k --max-seq 4224 # e5c92fad696fb35405918941d1062be3 / b0d8aca9a6c7626b0af8dc81e1de59eb
# prompt: the "Prompt text" block at the end of docs/measurements/23-sdk64-baseline.md, saved as models/eval_23.txt (md5 a838b25d115aab83e29d0605cb570a33)
python3 tools/hexagon/make_tokens.py models/hf models/eval_23.txt /tmp/t512_23.i32 --limit 512   # md5 10bb428f92c792aa7733339f8e6b0da1
python3 tools/hexagon/make_tokens.py models/hf <long.txt> /tmp/t1024.i32 --limit 1024             # speed rows only
python3 tools/hexagon/make_tokens.py models/hf <long.txt> /tmp/t4096.i32 --limit 4096
./build_x86_hexagon/hexagon_ref_run /tmp/qwen3_full --tokens /tmp/t512_23.i32 --eval             # your x86 reference for this prompt (container clang 41.2365 / 161, workstation gcc 41.5466 / 164)
```
The P4 prompt `/tmp/t512.i32` (md5 `9da5d7b4…`, x86 reference PPL 33.0195 / top-1 184, HEXAGON.md §5.1)
exists only on the original workstation; it is the G3 accuracy gate when available. On another box use
`t512_23.i32` against your own `hexagon_ref_run --eval` line and say so in the Notes.

**Rebase note (orchestrator, 2026-09-17 23:40).** The branch was rebased onto `hvx_impl` @ `1bbd8d53`
(after PRs #43–#46, including #33's validator / token-id changes in `nntr_htp_common.h` and
`htp_graph.c`). The three skels and the harnesses in the table were rebuilt from the rebased tree
(`305c51f1`); on that tree `HEX_ARCH=v79 run_sim_test.sh graph` and `quant` print STATs bit-identical to
the pre-rebase G1 run (`graph_prefill 0.0218946/6.88818`, `graph_decode 0.0197323/23.2374`,
`quant16_generic 184/25600`). The full v79 13-test + `profile acc` sweep is re-run once at PR time,
after this measurement (contract §7 budget); #33 and #35 touch disjoint DSP files.

## 2. Artifacts (built in the container, SDK 6.4.0.2 / hexagon-clang 19.0.04 `toolv19`, @ `305c51f1`)

| file | md5 (size) | built with |
|---|---|---|
| build_hexagon/skel/libnntr_htp_skel.v79.so (**B1**) | `af69e880d9b2181113030e0427117778` (50,728 B) | `HEX_ARCH=v79 ./tools/hexagon/build_skel.sh` |
| build_hexagon/skel/libnntr_htp_skel.v79qf.so (**B2**) | `9b50b2c2fe11a94683c0939b8f5a6508` (50,728 B) | `HEX_ARCH=v79 HEX_EXTRA_CFLAGS=-DHTP_FORCE_QF_HELPERS ./tools/hexagon/build_skel.sh` |
| build_hexagon/skel/libnntr_htp_skel.v75.so (**A**, optional) | `83182a7cd9bcdd780cf30c4e5e61e12a` (46,632 B) | `HEX_ARCH=v75 ./tools/hexagon/build_skel.sh` |
| build_hexagon/host/hexagon_rpc_test | `c269f67ebcf79f8c30e42abad9fa9f16` (524,984 B) | `./tools/hexagon/build_host_test.sh` (unchanged since #23) |
| build_hexagon/host/hexagon_e2e_test | `bbd5d1b15294bfffa95d6aedee7c247e` (1,262,128 B) | same |
| /tmp/qwen3_full.hexw / .hexcfg | `5abf61be…` / `5926be57…` | packer unchanged since P4 |
| /tmp/qwen3_full4k.hexw / .hexcfg | `e5c92fad…` / `b0d8aca9…` | `--max-seq 4224` |
| /tmp/t512_23.i32 | `10bb428f92c792aa7733339f8e6b0da1` | see §1 |

If the box did not receive the Mac binaries, rebuild them there with the three `build_skel.sh` lines
above plus `build_host_test.sh` (≈ 3 min). **The skel link is not byte-reproducible**: two builds of
the same tree differ in the order of ~100 dynamic-relocation entries (code and data identical; checked
with `cmp` on two v79 builds of `305c51f1`), so a rebuild has a different md5 at the same size — write
the md5 you get in the table and keep the size. Copy each skel to its variant name before the next build.

## 3. Measure (≈ 25 min; scripts push only files whose md5 changed)

Insert the serial after the image path (`run_e2e_test.sh /tmp/qwen3_full <serial> -- ...`) only if
more than one device is attached. Order: B1, then B2, then A if time allows.

```bash
# B1 (v79, the gate)
cp build_hexagon/skel/libnntr_htp_skel.v79.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so   # af69e880...
./tools/hexagon/run_device_test.sh                                                                    # RPC_TEST PASS
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t512_23.i32 --chunk 128 --steps 64
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t512_23.i32 --eval
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t512.i32 --eval                   # P4 prompt, if the file exists
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t1024.i32 --chunk 128 --steps 64
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t1024.i32 --eval
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --eval
# B2 (v79 with qf helpers): ledger ④ (3), recorded
cp build_hexagon/skel/libnntr_htp_skel.v79qf.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so   # 9b50b2c2...
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --chunk 128 --steps 64
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --eval
adb shell md5sum /data/local/tmp/nntr_htp/libnntr_htp_skel.so                                          # the skel that ran last (B2, or A below)
# A (v75, new decode) — OPTIONAL last v75 control row
cp build_hexagon/skel/libnntr_htp_skel.v75.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so   # 83182a7c...
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --chunk 128 --steps 64
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --eval
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512.i32 --eval                     # P4 prompt, if the file exists
adb shell md5sum /data/local/tmp/nntr_htp/libnntr_htp_skel.so
```
Lines to look for: `RPC_TEST PASS`; per run `E2E init ok weights=598623744 ...`, one
`E2E step <i> pos=<p> n=<n> pcycles=<c> us=<t> top1=<id>` per chunk / decode step, `E2E gen ...`,
`E2E wall_ms <w>`; for `--eval`: `E2E ppl <x> steps 511 top1 <n> wall_ms <w>`. If a run hangs or the
DSP restarts, note the last `E2E step` line and the `logs/hexagon/device_farf_<stamp>.log` tail in the
Notes, `adb reboot`, continue with the next command.
Reading the numbers (HEXAGON.md §8.2): prefill tok/s = prompt tokens ÷ (sum of `us` of the `n=128`
steps / 1e6); decode ms/tok = median host `us` over the 63 `n=1` steps ÷ 1000; decode tok/s =
1000 ÷ that; DSP Mcyc/tok = median `pcycles` of the same steps ÷ 1e6.

## 4. Hand back
1. Fill the tables and the Notes below (phone model, SDK version, your x86 reference line).
2. `git add docs/measurements/35-v79-skel.md logs/hexagon/ && git commit -s -m "[docs] Fill the #35 v79-skel measurement (<phone>, SDK <ver>)" && git push`
3. `gh issue edit 35 -R dlwlzzero/nntrainer --remove-label state:needs-measurement --add-label state:measured`

## Results (fill in)
Reference in brackets: B1 and B2 against #23 variant B (v79 skel, same silicon), A against #23 variant A
(`docs/measurements/23-sdk64-baseline.md`). v79 is the primary arch; A is the optional last v75 row. **Pass (B1 only)**: prefill tok/s within ±5 % of 207.4 /
131.2 / 34.4 and decode DSP Mcyc/tok within ±5 % of 60.1 / 76.5 / 177.6 at 512 / 1024 / 4096; `--eval`
on the P4 prompt inside the P4 band against x86 33.0195 / 184 (PPL 33.07–33.45), or on `t512_23.i32`
within +0.3 % of your own `hexagon_ref_run` line; no hang,
SSR or FARF fatal. B2 and A are recorded, not gated. The 512 decode column carries a ±5 %-class
spread by construction (#23 Notes), so read the 1024 / 4096 rows first.

| variant | skel md5 (from `md5sum` before the run) | ctx | image | prefill tok/s (ref) | decode median host ms (ref) | decode tok/s (ref) | DSP Mcyc/tok (ref) |
|---|---|---|---|---|---|---|---|
| B1 (v79) | | 512 | qwen3_full | (207.4) | (31.9) | (31.36) | (60.1) |
| B1 (v79) | | 1024 | qwen3_full | (131.2) | (39.6) | (25.27) | (76.5) |
| B1 (v79) | | 4096 | qwen3_full4k | (34.4) | (87.4) | (11.44) | (177.6) |
| B2 (v79qf) | | 512 | qwen3_full | (207.4, B1's row is the closer reference) | (31.9) | (31.36) | (60.1) |
| A (v75, optional) | | 512 | qwen3_full | (189.0) | (31.3) | (31.92) | (64.3) |

| variant | ctx | token file (md5) | --eval PPL (ref) | top-1 (ref) | x86 ref PPL / top-1 | gap % |
|---|---|---|---|---|---|---|
| B1 (v79) | 512 | t512_23.i32 (10bb428f…) | (41.2623) | (161) | 41.5466 / 164 (workstation gcc) or your own line | |
| B1 (v79) | 512 | t512.i32 (P4 prompt, if present) | (—, #23 B did not run it) | | 33.0195 / 184 | |
| B1 (v79) | 1024 | t1024.i32 | (5.9509) | (692) | — | |
| B1 (v79) | 4096 | t4096.i32 | (1.5687) | (3758) | — | |
| B2 (v79qf) | 512 | t512_23.i32 | (B1's value) | | same as above | |
| A (v75, optional) | 512 | t512_23.i32 | (40.7596) | (162) | same as above | |
| A (v75, optional) | 512 | t512.i32 (P4 prompt, if present) | (33.0884) | (189) | 33.0195 / 184 | |

`adb shell md5sum` after the last run: ______ (expected `9b50b2c2…` after B2, `83182a7c…` if A ran).

Reading B2 (HEXAGON.md §7 rule 1): B2 = B1 in PPL / top-1 and speed within noise → the 2026-08 v79
failures were a toolchain-8.x artefact, rule 1 closes; B2 correct but slower → the IEEE helpers are a
genuine v79 speed source; B2 wrong → the qf helpers have a v79-specific problem the v75 skel does not
see: record, file, do not block.

## Notes from the run
<phone model / serial, SDK version, x86 reference line, thermal, warm-up, FARF errors, stale-file pushes>

## Simulator record for these artifacts (container, SDK 6.4.0.2, Rosetta; relative signal only)
* v79 (B1 sources): 13/13 PASS; `profile acc` PASS, `profile_prefill_acc STAT max_abs=0.0659682
  max_rel=92.7791`, `workers=6`, `total_pcycles=6174196`; `quant_generic pm1=0/65536`,
  `quant16_generic pm1=184/25600`, `tie`/`tie2`/`i16_tie`/`i16_tie2` byte-identical; `matmul_w8a8_* 0/0`,
  `matmul_dma ref_* 0/0`, `logits 0/0`, `graph_prefill 0.0218946/6.88818`, `graph_decode 0.0197323/23.2374`.
* v79 + `-DHTP_FORCE_QF_HELPERS` (B2 sources), the kinds whose helpers switch: `quant PASS`
  (`0/65536`, `184/25600`), `matmul PASS` (`w8a8_* 0/0`, `w8a16_m8 0.03125/0.000974659`), `attn PASS`
  (`prefill 0.000488281/0.208165`, `decode 0.000244141/0.186483` = the v75 values), `rmsnorm PASS`
  (`general 0.000976563/0.000670241`), `eltwise PASS` (`add 0/0`, `silu_mul 0/0`).
* v75 (A sources): 13/13 PASS; `profile acc` PASS, `profile_prefill_acc STAT max_abs=0.0757427
  max_rel=66.5909`, `workers=4`, `total_pcycles=3488445` (P4 / #23:
  `0.077216/98.2637`; moves because the int16 decode is now sign-symmetric); `quant_generic pm1=0/65536`,
  `quant16_generic pm1=162/25600` (was 167), `matmul_w8a8_m8 0.0078125/0.000788644` (unchanged),
  `graph_prefill 0.0273907/8.00437`, `graph_decode 0.0174583/24.125`, `logits 7.62939e-06/2.36832e-07`
  (unchanged).
