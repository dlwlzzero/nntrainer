# Measurement 23: SDK 6.4 rebuild of the M6 P4 baseline (v75 skel) and first v79-skel data point

Branch `hvx/23-sdk64-baseline` @ `4b25ac3c` (artifacts built from that commit; kernels unchanged since `82eacd7e`) — estimated device time: **45 min**

## Why
Every Hexagon build moved from the workstation's SDK 6.0.0.2 / toolchain 8.7.08 to the
container's SDK 6.4.0.2 / hexagon-clang 19.0.04 (`toolv19`). On the simulator the v75 build
is bit-identical to the P4 record (HEXAGON.md §8.3); this run shows whether the device agrees
(closes follow-up ⑭) and gives the v79-native skel its first silicon numbers (ledger ④: on the
v79 simulator `quant`, `matmul`, `matmul_dma` and `logits` fail, the rest pass — see §5.2).
Variant **A = v75 skel** decides #23; variant **B = v79 skel** is recorded, never pass/fail.

## Artifacts (built in the container, SDK 6.4.0.2 / hexagon-clang 19.0.04, `-mhvx-ieee-fp` probe accepted on both arches)
All on the Mac at `/Users/dlwlzzero/Projects/nntrainer/`; none is tracked by git.

| file | md5 | built with |
|---|---|---|
| build_hexagon/skel/libnntr_htp_skel.v75.so | `b9b1e3615d20959482efd51ad65b7320` | `HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/build_skel.sh` @ 4b25ac3c (46,632 B) |
| build_hexagon/skel/libnntr_htp_skel.v79.so | `bdfdf143f84fa158ebccba35862d9c04` | `HEX_ARCH=v79 tools/docker/run.sh ./tools/hexagon/build_skel.sh` @ 4b25ac3c (50,728 B) |
| build_hexagon/host/hexagon_rpc_test | `c269f67ebcf79f8c30e42abad9fa9f16` | `tools/docker/run.sh ./tools/hexagon/build_host_test.sh` @ 4b25ac3c (NDK r26d, API 31) |
| build_hexagon/host/hexagon_e2e_test | `bbd5d1b15294bfffa95d6aedee7c247e` | same |
| build_x86_hexagon/qwen3_full.hexw / .hexcfg | `5abf61bef086368559423daa3bea9a99` / `5926be5703f85531c9c46c1978c25287` | `nntr_hexpack /model/nntr_qwen3_0.6b_w8cx_DEFAULT.bin` @ 4b25ac3c (max_seq 2048; weights 598,623,744 B; source .bin md5 `7562313bb4cc70410450d0ab3a4fa563`, PR #31) |
| build_x86_hexagon/t512.i32 | `10bb428f92c792aa7733339f8e6b0da1` | `make_tokens.py /model/hf eval.txt --limit 512` (512 tokens; text below) |
| build_x86_hexagon/eval.txt | `a838b25d115aab83e29d0605cb570a33` | the prompt, reproduced at the end of this file |

The packer and the ABI (`nntr_htp_common.h` v4, `.hexcfg` `weight_layout=tiled32`) are unchanged
by #23, so the `qwen3_full` / `qwen3_full_4224` images already on the workstation are usable **if**
their source `.bin` matches the one above (issue #32 tracks that comparison); if the workstation
image's md5 differs, either copy the Mac image (600 MB) or repack from the same `.bin`. The
4096-token run needs the workstation's `qwen3_full_4224` image (`--max-seq 4224`, HEXAGON.md §8.2).

Getting the artifacts to the workstation (from the workstation, Mac reachable as `<mac>`):
```
cd <nntrainer checkout> && git fetch && git checkout hvx/23-sdk64-baseline
mkdir -p build_hexagon/skel build_hexagon/host build_x86_hexagon
scp <mac>:/Users/dlwlzzero/Projects/nntrainer/build_hexagon/skel/libnntr_htp_skel.v7{5,9}.so build_hexagon/skel/
scp <mac>:/Users/dlwlzzero/Projects/nntrainer/build_hexagon/host/hexagon_{rpc,e2e}_test build_hexagon/host/
scp <mac>:/Users/dlwlzzero/Projects/nntrainer/build_x86_hexagon/t512.i32 build_x86_hexagon/
md5sum build_hexagon/skel/libnntr_htp_skel.v7*.so build_hexagon/host/hexagon_*_test build_x86_hexagon/t512.i32   # must match the table
```
If scp is inconvenient, `t512.i32` can be regenerated on the workstation from the prompt text at
the end of this file: `python3 tools/hexagon/make_tokens.py <hf tokenizer dir> eval.txt /tmp/t512.i32 --limit 512`
(must print `tokens=512` and md5 `10bb428f...`; the HF tokenizer files are the Qwen3-0.6B ones).
The 1024- and 4096-token files are the workstation's existing `t1024.i32` / `t4096.i32` from the
P4 run (they only feed the speed rows; no x86 reference exists for them).

## Steps (workstation, phone on USB)
1. `git fetch && git checkout hvx/23-sdk64-baseline`, copy the artifacts as above, verify the md5s.
2. `adb devices` shows `R3CY10WM83Y device`. Copy the images and tokens to `/tmp` as in the P4 run
   (`/tmp/qwen3_full.{hexw,hexcfg}`, `/tmp/qwen3_full_4224.{hexw,hexcfg}`, `/tmp/t512.i32`, `/tmp/t1024.i32`, `/tmp/t4096.i32`).
3. **Variant A (v75)** — run first so the baseline is never lost:
   ```
   cp build_hexagon/skel/libnntr_htp_skel.v75.so build_hexagon/skel/libnntr_htp_skel.so
   md5sum build_hexagon/skel/libnntr_htp_skel.so                       # -> b9b1e361..., write it in the table
   ./tools/hexagon/run_device_test.sh R3CY10WM83Y                       # RPC_TEST PASS (skel loads)
   ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY10WM83Y -- --tokens /tmp/t512.i32 --chunk 128 --steps 64
   ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY10WM83Y -- --tokens /tmp/t512.i32 --eval
   ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY10WM83Y -- --tokens /tmp/t1024.i32 --chunk 128 --steps 64
   ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full R3CY10WM83Y -- --tokens /tmp/t1024.i32 --eval
   ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full_4224 R3CY10WM83Y -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64
   ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full_4224 R3CY10WM83Y -- --tokens /tmp/t4096.i32 --eval
   adb -s R3CY10WM83Y shell md5sum /data/local/tmp/nntr_htp/libnntr_htp_skel.so   # the skel that actually ran
   ```
   Expected log lines: `E2E init ok weights=598623744 ...`, one `E2E step <i> pos=<p> n=<n> pcycles=<c> us=<t> top1=<id>`
   per chunk / decode step, `E2E gen ...`, `E2E wall_ms <w>`; for `--eval`: `E2E ppl <x> steps 511 top1 <n> wall_ms <w>`.
   Read the numbers as in HEXAGON.md §8.2: prefill tok/s = prompt tokens ÷ (sum of `us` of the `n=128` steps / 1000);
   decode ms/tok = median host `us` over the 63 `n=1` steps ÷ 1000; DSP Mcyc/tok = median `pcycles` of the same steps ÷ 1e6.
4. **Variant B (v79)** — same eight commands after
   `cp build_hexagon/skel/libnntr_htp_skel.v79.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so`
   (-> `bdfdf143...`). If a run hangs or the DSP restarts, note the last `E2E step` line and the
   `logs/hexagon/device_farf_<stamp>.log` tail in "Notes", `adb reboot`, and continue with the next command;
   nothing in B blocks #23.
5. Paste the summary lines into the tables below, commit this file on the same branch, push, and set
   the issue label to `state:measured` (`gh issue edit 23 --remove-label state:needs-measurement --add-label state:measured`).

## Results (fill in)
Reference = HEXAGON.md §8.2 M6 P4 (v75 skel, SDK 6.0.0.2, 2026-09-16). Pass for A: DSP Mcyc/tok within ±3 %, tok/s within ±5 %.

| variant | skel md5 (from `md5sum` before the run / device) | ctx | image | prefill tok/s (ref) | decode median host ms (ref) | decode tok/s (ref) | decode median DSP Mcyc/tok (ref) |
|---|---|---|---|---|---|---|---|
| A (v75) | | 512 | qwen3_full | (192.1) | (36.1) | (27.7) | (68.9) |
| A (v75) | | 1024 | qwen3_full | (118.2) | (43.7) | (22.9) | (85.2) |
| A (v75) | | 4096 | qwen3_full_4224 | (27.3) | (106.1) | (9.4) | (216.9) |
| B (v79) | | 512 | qwen3_full | (192.1) | (36.1) | (27.7) | (68.9) |
| B (v79) | | 1024 | qwen3_full | (118.2) | (43.7) | (22.9) | (85.2) |
| B (v79) | | 4096 | qwen3_full_4224 | (27.3) | (106.1) | (9.4) | (216.9) |

Accuracy (`--eval`). The 512 row uses **this handoff's `t512.i32`**, whose x86 reference was
measured in the container at `4b25ac3c`: **PPL 41.2365 / top-1 161** (511 steps). Pass for A: the
DSP-vs-reference gap inside the P4 band (+0.16 … +1.29 %; on the workstation's own 512-token
prompt P4 measured DSP 33.3132 / 189 against x86 33.0195 / 184, +0.89 %). If the workstation's
`t512.i32` from the P4 run is also at hand, add that row for continuity. The 1024 / 4096 rows
have no x86 reference (P4 did not record them either); fill PPL / top-1 only.

| variant | ctx | token file (md5) | --eval PPL | top-1 | x86 ref PPL / top-1 | gap % | v79 behaviour (B only: matches / garbage / inf / crash) |
|---|---|---|---|---|---|---|---|
| A (v75) | 512 | t512.i32 (10bb428f...) | | | 41.2365 / 161 | | — |
| A (v75) | 512 | workstation P4 prompt (optional) | | | 33.0195 / 184 | | — |
| A (v75) | 1024 | t1024.i32 | | | — | | — |
| A (v75) | 4096 | t4096.i32 | | | — | | — |
| B (v79) | 512 | t512.i32 (10bb428f...) | | | 41.2365 / 161 | | |
| B (v79) | 1024 | t1024.i32 | | | — | | |
| B (v79) | 4096 | t4096.i32 | | | — | | |

`find_divergence.py` bound: not required for this run; if A's gap leaves the band, run
`python3 tools/hexagon/find_divergence.py /tmp/qwen3_full /tmp/t512.i32 --serial R3CY10WM83Y --chunk 8`
on a 1-layer image (HEXAGON.md §6) and record the `FIRST_DIVERGENCE` line here.

## Simulator record for this build (Mac, Rosetta emulation; relative signal only)
* v75: 13/13 PASS, `profile acc` PASS, `profile_prefill_acc STAT max_abs=0.077216 max_rel=98.2637`
  (= P4), `total_pcycles=3454962` (P4 3,415,503), `MATMUL_W8A16 per_call=697913` (P4 683,954),
  `MATMUL_W8A8 per_call=120224`, `ATTN per_call=91475`, `barrier_empty_x1000=4121706`;
  `quant_generic pm1=0/65536`, `quant16_generic pm1=167/25600`.
* v79: 9 PASS (`smoke pool exp rmsnorm rope eltwise embed attn graph`), 4 FAIL (`quant`: tie row
  `x=-1.5 -> -1` vs ref `-2`, ±1 in `rand0`/`rand15`/`k3072`, `quant_generic 13/65536`,
  `quant16_generic 229/25600`; `matmul`: `w8a8_m1 max_abs=0.03125 max_rel=0.045677`;
  `matmul_dma`: `ref_4096k 0.0625/0.273998`; `logits`: `0.0195827/0.273973`).
  `profile acc` v79: PASS, `profile_prefill_acc STAT max_abs=0.0899355 max_rel=49.2742` (moved from the v75 value, still inside the 0.1 bound); the v79 simulator exposes 6 HVX units so it ran `workers=6`, `total_pcycles=6105728`, `MATMUL_W8A8 per_call=212250`, `MATMUL_W8A16 per_call=1252328`, `ATTN per_call=174095`, `barrier_empty_x1000=8319592` — not comparable with the 4-worker v75 numbers; wall 16m34s.
* x86 reference gate: `LOWER_TEST PASS`, `W8CX_BIN_TEST PASS`, oplist header PASS; `hexagon_ref_run --eval`
  on t512.i32 = PPL 41.2365 / top-1 161 (160 s in the container).

## Notes from the run
<thermal, warm-up, FARF errors, stale-file pushes, the last E2E line before any v79 hang>

## Prompt text (`eval.txt`, 519 words; tokenizes to 512 tokens with `--limit 512`)
```
The Hexagon digital signal processor inside a modern smartphone is a curious piece of hardware. It was designed, in the first place, to handle the steady stream of audio and radio samples that a phone must process without waking the main application cores, and for that purpose it was given wide vector registers, a small but very fast local memory, and a scheduler that can keep several hardware threads busy at once. Over the years the same design has turned out to be a good fit for neural networks, because a network is, at bottom, a long sequence of multiply and accumulate operations over arrays that fit comfortably in those wide registers. The interesting part of moving a language model onto such a processor is not the arithmetic itself but everything around it: how the weights are laid out in memory so that each vector load brings in exactly the bytes the next instruction needs, how the activations are quantized on the fly without losing the small differences that decide which token comes next, and how the host processor and the signal processor agree on who owns which buffer at which moment. A single mistake in any of these arrangements does not usually produce a crash. Instead it produces a model that still speaks fluent sentences but chooses slightly wrong words, and the only way to notice is to measure the perplexity of a known text and compare it against a reference implementation that runs, slowly but exactly, on an ordinary computer. That is why every change to the kernels is followed by the same ritual: build the reference, build the simulator image, run the golden tests, and only then push the library to the phone and read the numbers back. The simulator does not model memory bandwidth, so it can tell you that the answer is right but not how long it will take; for the timing you need the device, a stopwatch in the harness, and a good deal of patience while the thermal throttling settles. The reward, when everything lines up, is a model that generates text at a rate the phone's own processor cannot match, using a fraction of the power, and leaving the application cores free to draw the screen and answer the network. It is a modest reward measured in tokens per second, but it is the kind of result that only appears after every layer of the stack has been checked twice.
There is also a quieter lesson in all of this. Tools change underneath a project: a new compiler arrives, a vendor ships a new software development kit, the simulator gains a processor generation it did not have before. Each of those changes is an opportunity for the numbers to drift without anyone touching the kernels, and so the same golden tests that guard against a careless edit also guard against a careless upgrade. Running them again after the toolchain moves, and writing down exactly which version produced which figure, is dull work, but it is the difference between knowing that a result still holds and merely hoping that it does.
```
