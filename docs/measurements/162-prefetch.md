# Measurement 162 (b): the prefetch overlap, verdict sitting (A vs B, bit-preserving)

Branch `htp/162-prefetch`, code @ `a160adfc` (plan
`docs/plans/162-g1024-attention-prefetch.md` §4 step 2 and a reduced step 3,
issue #162). Estimated device time: **≈ 35 min** (plus up to 20 min cool
start). Run by the orchestrator on the workstation (unit `R3CY10WM83Y`).

## Why

Step 0 (2026-09-29, `logs0/`) rejected lever (a) (coverage C = 0.50; no
microbench variant beat `as_is`) and passed P′ for lever (b) at S = 4 MiB
under the bypass: net +1.584 ms/token cold, +1.262 warm (touch_threads = 7),
`dmm_pct` −3.1 / −1.4, touch_late 0/64. This sitting asks whether the
in-app overlap lifts G = 1024 decode from A's ≈ 51 to ≥ 50.00 with margin,
with every byte unchanged. Only A and B are run (no (a), so no C).

| variant | env |
|---|---|
| **A** | nothing (committed default) |
| **B** | `NNTR_MOE_PREFETCH_MIB=4` |

What B does: after load, each MoE layer N gets the first 4 MiB of layer
N+1's first FC layer's Q4_0 weights (conv in_proj or qkv; 21 of the 22 MoE
layers have one). In each decode MoE call, after the dspq request is
written and before the read spin, all 8 pool threads (7 workers and the
main thread, `parallel_for`) read one word per 64-byte line of those
bytes. Nothing is written but a sink.

## Artifacts (`/local/mnt/workspace/htp_moe/162/set_b/`, `md5.txt` in it)

| file | md5 | built with |
|---|---|---|
| `libnntr_hvx_skel.so` | `3a8b00f418888a70a693725aaeee8e0a` | `test/htp/build.sh` @ `a160adfc` (v79, `UNDEFINED SYMBOLS OK (51 runtime imports)`). No DSP source change |
| `nntrainer_causallm` | `6f4cc4b9843de2e9f510477ac50acb5f` | `build_android.sh --htp --cache` @ `a160adfc` |
| `libcausallm_core.so` | `6409c1405517599042f6127e0e3d4dbd` | same (`moe prefetch: on` 1, `NNTR_HTP_FORWARD_KINDS` 2) |
| `libnntrainer.so` | `8f74303dad5704e23fb7c6891a730831` | same, `jni/obj/local/arm64-v8a/` (NEEDED `libsdkl.so`, `libcdsprpc.so`; `dspq: on` 1, `window_jobs=` 1) |
| `libccapi-nntrainer.so` | `8e537ba50101dae491bd0754544fe484` | same |
| `libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 sysroot |
| `libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL `lib/6.4.0.1/armv8_android26` |
| `prompt512.txt` | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt` (prompt p01) |
| `bitset-0{2..8}-*.txt` | see `md5.txt` | `docs/measurements/prompts/` (the #152 set, p02–p08) |
| model `q40-qs4cx-wh` | — | on the phone since #100 |

Not staged: `libcdsprpc.so` (rule 5).

## Steps (orchestrator; **device measurement unavoidable**)

```
bash /local/mnt/workspace/htp_moe/162/run_162b.sh R3CY10WM83Y
```

Logs go to `/local/mnt/workspace/htp_moe/162/logs_b/`, the summary to
`logs_b/sitting.out`; the last line reads `expectation mismatches: 0` when
every expected line was seen. The script:

0. Records `adb devices`, screen off, waits for zone0 ≤ 35 °C (stops after
   20 min), checkpoint t0.
1. Pushes `set_b/` to `/data/local/tmp/nntrainer/causallm/s162b`, device
   md5 vs `md5.txt` (`MD5 OK` or stop), the #158 config block (greedy,
   `bad_word_ids [124900]`, `moe_engine: htp`).
2. Sanity at G = 8, A then B. **Stops** if A lacks `dma_bypass=1
   source=default` or `dspq: on` (void, rules 36 / 43), or if B's close
   line lacks `window_jobs=168` (the touch never ran). Checkpoint t1.
3. tok/s at prompt 512, `NNTR_NUM_THREADS=8`, no `NNTR_OP_TIME`, mirrored
   A B | B A at G = 64, 512, 1024; checkpoint after each G.
4. MoE dumps A1, B, A2 at G = 64 → `htp_dump_eval.py`: A1 vs A2 and A1 vs
   B `bit_identical=1`. Checkpoint t3.
5. `NNTR_PPL_DECODE=cont.ids` at G = 512: A self (writes the continuation),
   B and A2 forced; every `[PPL] decode step=` line of B and A2 equal to
   A's. Checkpoint t4.
6. The 8 prompts at G = 64, A then B; `strip` text compared. Checkpoint t5.
7. Summary: text per prompt, text of every tok/s cell vs A r1 of its G,
   prefill / decode / last-64 tok/s per cell with means and B vs A %.

Every log is checked for (rule 36): `moe m1 gemv: on (applied=0x703e1)
lead=192KB rows1=1 feed=vtcm dma_bypass=1 source=default` once, `dspq: on`
once, `dspq: close calls=22G served=22G bad=0 … window_jobs=J` with J = 0
for A and 21·G for B, and for B `[CausalLM] moe prefetch: on mib=4
layers=21` once (A: no such line).

## Gates (plan §1, for this sitting)

| # | pass |
|---|---|
| G1 | B decode at G = 1024, mean of r1 / r2, **≥ 50.00**; G 64 / 512 not below A by more than A's own r1/r2 spread |
| G2 | the banner checks above on every log |
| G3 | dumps A1 vs A2 and A1 vs B `bit_identical=1` |
| G4 | `[PPL] decode step=` lines of B and A2 identical to A's |
| G5 | text ≡ A on 8/8 prompts and in every tok/s cell |
| G6 | B prefill ≥ −5 % of A per G |

## Results (fill in)

Reference: step 0's A at G = 1024 (with `NNTR_OP_TIME=1`) 52.36 / 50.90
tok/s; attention 1.68–1.87 ms/token, MoE wait 9.5 ms/token.

| G | A r1 / r2 prefill · decode | B r1 / r2 prefill · decode | mean decode A → B (%) | text ≡ A |
|---|---|---|---|---|
| 64 | | | | |
| 512 | | | | |
| 1024 | | | | |

| check | result |
|---|---|
| unit, zone0 at t0–t5, MD5 OK | |
| G2 banners (mismatch count) | |
| G3 A2 / B `bit_identical` | |
| G4 nll A2 / B | |
| G5 8-prompt text | |

## Notes from the run

<serials seen, thermal, anything void>
