# Measurement 225: the LFM2.5 table of record (9 cells × 3 cases) on the config of record with the FC WH sidecar

Branch `htp/225-fcwh-e2e` (PR #233) — code @ `7efd22adf` (the staged set was
built from it; this file and the runner sit on top) — estimated device time:
**≈ 95 min** (reboot + 5 min idle, 68 runs, cool waits; up to +3 min if the
hybrid falls back), plus **≈ 10 min** of workstation steps. Device: an S25
Ultra (v79) on the farm, run by the user.

## Why

#222's closing sitting failed three ways on the config of record: one PD
(Q28) could not load the keys' re-quantized FC copies, the hybrid ran out
of DSP heap at P1024, and the hybrid's accuracy failed (stacked Q4_0 →
qs4cx quantization; LEDGER rule 63). #225 packs the 67 FC weights once
from f32 into a `QS4CX_WH` sidecar (`nntr_lfm2_8b_a1b_q40_arm_fcwh.bin`):
HTP prefill (PR 1, #230) and the one-PD decode's FC / dense FFN ops (PR 2,
#233) read that one copy; the hybrid's prefill runs in 512-token chunks
(the way of record, user 2026-10-06). This sitting fills the table the
user fixed on 2026-10-06 — **P64 / P512 / P1024 × G64 / G512 / G1024 for
CPU only, hybrid and E2E one PD** — and is plan 225's gate G1–G4:

| gate | read from | pass |
|---|---|---|
| G1 fit | the 27 cells of A, B (or Bfb), Q | every cell has `prefill:` and `generation:` lines, no VOID |
| G2 E2E | Q cells | `calls/token=1.00`; `cpu fc skipped` = 32 × tokens; `[HTP] fc wh: … heap_kib=0 requant=0`; `heap_used_kib` < #222's 223 003; `mapped_mib` ≤ 3840 (the PD's ceiling) |
| G3 hybrid P1024 | `B_P1024_*` | no `AEE_ENOMEMORY` (`0x80000402`); if B cannot hold it, Bfb (below) is the hybrid case |
| G4 accuracy, **T2** (user 2026-10-06) | texts + `loops.txt` + PPL block | PPL is **recorded, not gated**; a run fails only on a repetition loop (`loop_check.py`: the variant loops where A does not) or off-context text; the user approves the texts |

Prefill ≥ −5 % of A is **not** a gate here (the CPU is a different case,
not a reference binary of the same path); read cells side by side.

**PR 3 (`NNTR_HTP_FC_M1`, the hybrid's M = 1 FCs on the WH GEMV) is not
built, so there is no Bm1 cell.** It is added (B + `NNTR_HTP_FC_M1=1` at
P512 × G64 / G512 / G1024, 3 runs, ≈ 5 min) when PR 3 lands.

## What to expect (arithmetic, not measured)

* **Q (one PD, C = 28):** pool 22 × 28 × 5.25 = 3234 MiB in 13 chunks
  (3328 mapped) + the sidecar's images in their own arena chunk (≈ 216 MiB
  of WH, less what fits the pool's tails) + the lm_head Q4M1 chunk (147 →
  192 MiB mapped) = **3712–3776 MiB mapped of 3840**; no FC bytes on the
  heap (`heap_kib=0`), so `heap_used_kib` should sit near Qold's 92 058 of
  #222, not 223 003. If the map is refused, Q is VOID with its
  `e2e: fc arena` / `arena:` lines (PR #233's `ponytail:` upgrade is to
  put the lm_head slices in the FC chunk's tail, −64 MiB).
* **B (hybrid, full residency):** 3696 MiB of experts in 15 chunks leave
  ≈ 144 MiB of tails; the ≈ 216 MiB of WH images fill them and the rest
  (≈ 72–86 MiB) goes to the DSP heap by the bit-exact unpack (`fc wh: …
  heap_kib=<n>`, not 0). P1024 now runs in 512-row chunks, so the conv
  block's scratch is the P512 size. **If B does not load or dies at a
  prompt length** (user decision (b), 2026-10-06), the runner records B as
  VOID there with its error and arena / heap lines and re-runs that cell
  and B's remaining cells at that length as **Bfb** = `attn_proj_engine`
  and `dense_ffn_engine` cpu, `conv_block_engine` htp kept — **Bfb is then
  the hybrid case of record** for that length. Not pool C = 31.
* **Texts:** B / Q / Aoff run other arithmetic than the CPU (rule 39);
  differing text is expected and is not a fail under T2. B and Q differ
  from each other too (B's decode row is the CPU's Q4_0 FCs, Q's the DSP
  WH GEMV). Run 2 must equal run 1 of the same case (determinism).
* **First generated token:** `causal_lm.cpp` prints the prefill's token
  only when the prompt is shorter than `init_seq_len`. A (q40 config,
  `init_seq_len 512`) drops it at P512, the NPU configs (1024) keep it;
  at P1024 all drop it. The runner marks a text that is the other plus
  its first token as `same+1st`. The `[PPL] decode` lines are
  position-indexed and unaffected.

## Variants (one binary set; config and env only; ≤ 4 + the fallback)

| variant | model dir / config | env | cells |
|---|---|---|---|
| **A** (CPU only, reference, first in each block) | `models/q40`, its own `nntr_config.json` (`init_seq_len 512`; set to 1024 for P1024 only, as #222 did) | nothing | 9 + PPL (writes the continuations) |
| **Aoff** | `models/q40-qs4cx-wh`, `cfg_off.json` = the config of record minus `conv_block_engine` / `dense_ffn_engine` / `attn_proj_engine` (no HTP prefill FC, sidecar never opened) | nothing | P512 × G64 / G512 + PPL (the NPU model's own reference) |
| **B** (hybrid) | `models/q40-qs4cx-wh`, `cfg_new.json` = the config of record (md5 `3f6808e3…`, `fc_wh_file_name`, `init_seq_len 1024`) | nothing | 9 (G64 twice, mirrored) + PPL |
| **Bfb** (only after a B VOID) | `cfg_fb.json` = the config of record with `attn_proj_engine` / `dense_ffn_engine` `cpu` | nothing | B's cells at the VOID length |
| **Q** (E2E one PD) | as B | `NNTR_HTP_E2E=1 NNTR_MOE_CACHE_EXPERTS=28` (model file pre-read + `page_cache_evict -1`) | 9 (G64 twice, mirrored) + PPL |

Every config gets the sittings' overlay (the runner applies it):
`num_to_generate` = G, `bad_word_ids [124900]`; `do_sample false` in both
`generation_config.json`. `NNTR_NUM_THREADS=8` everywhere. The runner
restores the device's generation configs and the `q40` config at the end
and leaves the config of record (pristine) in `models/q40-qs4cx-wh`.

Prompts: **P64** = `p64.txt` (p01's first 64 tokens), **P512** = `p01.txt`
(`77-prompt512.txt`), **P1024** = `p1024.txt` (p01 + p07 + p05 + p03 cut
at 1024 tokens); the recipe and md5s are in `prompts/README.md`.

Order (`zone0` ≤ 35 °C at every block start): for P in 64, 512, 1024:
G64 `A B Q`, cool, `Q B` (run 2, mirrored), [P512: `Aoff`]; G512 `A B Q`
[P512: `Aoff`]; G1024 `A B Q`. Then the two profile runs, then the PPL
block (8 prompts at G 256: `A` self, `A` p01 forced once more = the null
check, then `Aoff`, `B`, `Q` forced on A's continuation).

Run count: A 9 + B 11 + Q 11 + Aoff 2 + profiles 2 + PPL 33 = **68**
(+1 failed B and its Bfb re-run per VOID length).

## Artifacts (workstation, SDK 6.4.0.1, HexKL 6.4.0.1 (`hexkl-1.0-beta.2`), NDK r30; built from `7efd22adf`)

Staged at `/local/mnt/workspace/htp_moe/225/` by `225-stage.sh` (`md5.txt`
= `dd24b4fc379656e8c29150ba8c1e0eb5`; the runner checks it on both ends),
plus `run_225.sh` and `extra/` (step 2; md5s below, held by the runner).

| file | md5 | built with |
|---|---|---|
| `app/libnntr_hvx_skel.v79.so` (S25) | `58e3a85fe1f983b64705128f52607649` | `test/htp/build.sh` (`ARCH OK (V79)`, `UNDEFINED SYMBOLS OK (62 runtime imports)`) |
| `app/libnntr_hvx_skel.v81.so` (S26, not used here) | `ca25ec2d68e6830a5f18b621b1554048` | `HEX_ARCH=v81 test/htp/build.sh` |
| `app/nntrainer_causallm` | `f30349be4c7b87d708878f41ca692900` | `build_android.sh --htp --cache` |
| `app/libcausallm_core.so` | `b1b7ed3b11420d149052db702c2902c4` | `NNTR_HTP_FORWARD_KINDS` strings: 2 |
| `app/libnntrainer.so` (`jni/obj/local`) | `ebf3556ec9aeae5665ca07847816e8fa` | NEEDED `libsdkl.so`, `libcdsprpc.so` |
| `app/libccapi-nntrainer.so` (`jni/obj/local`) | `ad46760cde21a9617ada13e0f310092a` | |
| `app/libc++_shared.so` (NDK r30) / `app/libsdkl.so` | `b1586b9b512712800fd36a24abac1c0a` / `0ad4e22a70e4f135bce38ad8fd1e001b` | |
| `app/page_cache_evict` | `42595651ef514e155887eb31f421b2dc` | NDK clang, `tools/htp/page_cache_evict.c` |
| `app/cfg_new.json` = `config/q40-qs4cx-wh.nntr_config.json` | `3f6808e30b6e16e1c8592fa39977814c` | the config of record (#225 PR 2) |
| `app/p01.txt` … `p08.txt` | `fc65c158…` … `67b657c1…` | `prompts/README.md` |
| `model/nntr_lfm2_8b_a1b_q40_arm_fcwh.bin` (228,188,160 B) | `71812a91d5acdbe9e026c479db8e275e` | `nntr_quantize_stream --fc_wh_sidecar` (#225 PR 1) |
| `extra/p64.txt` / `extra/p1024.txt` | `c0d3e9ffb0c9565e51fa044c53ac71d7` / `2e47c5f45f538babcc7b4a7bb48a4e70` | `prompts/p64.txt`, `prompts/p1024.txt` |
| `extra/cfg_off.json` / `extra/cfg_fb.json` | `01b492ad749d2a315ae242a276e886b6` / `49fcc38aa665a3f4c6825d937ff33eb5` | derived from `cfg_new.json` by the runner (one `sed` each) |
| `extra/loop_check.py` | `e607c5f7d4638a6eb199ac0d572867c1` | `tools/htp/loop_check.py` (host only) |
| `run_225.sh` (= `225-run.sh`) | `2d06483d8771d1a85f9877e3c9fd92da` | this branch |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm.bin` | `7b7867fab51845664c0050c0a837073e` | NPU model (#78); the config of record names it `nntr_lfm2.5_8b_a1b_q40_arm.bin` = a symlink (the runner makes it, idempotent) |
| `/local/mnt/workspace/models/lfm2.5-8b-a1b/q40/nntr_lfm2_8b_a1b_q40_arm.bin` | `d28f55c5bd7adeb8bf73b02de582eb88` | CPU model (#78) |

## Steps (workstation → farm session, unit on USB)

1. **The set (workstation).** It is already staged; to rebuild instead:
   `git fetch && git checkout htp/225-fcwh-e2e`, `source tools/htp/env.sh`,
   `export HEXKL_ROOT=$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon
   HEXKL_SDK_VER=6.4.0.1`, `HEX_ARCH=v79 ./test/htp/build.sh && cp
   test/htp/build/libnntr_hvx_skel.so test/htp/build/libnntr_hvx_skel.v79.so`
   (the same for v81), `(cd Applications/CausalLM && ./build_android.sh
   --htp --cache)`, then `bash docs/measurements/225-stage.sh`. Either way:
   ```
   cd /local/mnt/workspace/htp_moe/225 && md5sum md5.txt   # dd24b4fc379656e8c29150ba8c1e0eb5
   md5sum -c md5.txt | grep -vc ': OK$'                      # 0
   ```
   A differing `md5.txt` means the binaries differ from this table: say so
   under Notes and paste the new `md5.txt`.
2. **The runner and the extras (workstation, from this branch; already
   done on 2026-10-06, repeat if in doubt):**
   ```
   W=/local/mnt/workspace/htp_moe/225; mkdir -p $W/extra
   cp docs/measurements/225-run.sh $W/run_225.sh
   cp docs/measurements/prompts/p64.txt docs/measurements/prompts/p1024.txt tools/htp/loop_check.py $W/extra/
   ```
   If the farm session runs on another machine, copy the whole
   `/local/mnt/workspace/htp_moe/225/` there (≈ 260 MB incl. the sidecar).
   The farm machine needs `adb`, `bash`, `python3` (stdlib only).
3. The unit must already hold `models/q40-qs4cx-wh/` (with
   `nntr_lfm2_8b_a1b_q40_arm.bin`, its tokenizer and configs) and
   `models/q40/` under `/data/local/tmp/nntrainer/causallm/` (the #207 /
   #208 / #222 farm units do; otherwise `(cd Applications/CausalLM &&
   ./install_android.sh --model=<model dir>)` for both). The runner pushes
   the sidecar into `models/q40-qs4cx-wh/` itself (once, 228 MB, skipped
   when the device md5 already matches). `adb devices` shows exactly one
   unit; note its serial and SoC (S25 = SM8750 = v79).
4. **Reboot the unit, wait 5 min idle** (LEDGER rule 61), then
   ```
   bash /local/mnt/workspace/htp_moe/225/run_225.sh <serial>
   ```
   (`v79` is the default arch.) It checks `md5.txt` on the workstation,
   pushes the set, checks the device md5s of the app, the extras, the
   sidecar and the configs (`MD5 OK (app, extra, sidecar)`;
   `logs/md5_device.log`, `logs/md5_models.log`), then runs the blocks
   above. On `STOP` (stale skel `0x8000040e`, A did not generate, unit
   gone): reboot and run the same command; it resumes at the first
   unfinished run. **A VOID is not a STOP**: the sitting goes on.
5. Expected lines per run: `prefill: <P> tokens, <ms> ms, <tps> TPS`,
   `generation: <G> tokens, …`, `generation(last 64): …`, `Max Resident
   Set Size`, the text. The runner's own checks (`OK` / `BAD` lines; the
   last line counts the BADs):
   * every run: `prefill: <P> tokens` (speed cells);
   * A: no HTP banner;
   * NPU runs: one `moe m1 gemv: on (applied=` banner;
   * Aoff: `dspq: on`, no token driver, **no** `[HTP] fc wh:` banner;
   * B / Bfb: `dspq: on`, no token driver, one `[HTP] fc wh: file=…fcwh.bin
     handles=<n> arena_kib=<n> heap_kib=<n> requant=0` (printed; B's
     `heap_kib` is the overflow);
   * Q: `[HTP] fc wh: … heap_kib=0 requant=0`; `graph: q4m1 weights=… wh_handles=<n>`;
     `token driver: close … hops/token=0.00 … timeouts=0 stale=0 …
     id_mismatch=0`; `e2e: close … unmap_fail=0 detach_fail=0`;
     `calls/token=1.00`; `cpu fc skipped` = 32 × tokens; `heap_used_kib`
     < 223 003; `e2e: fc arena … mapped_mib` ≤ 3840 (printed with
     `s1_arena_mib`, `s1_heap_kib`, the pool line);
   * profiles (`prof_B` or `prof_Bfb`, `prof_Q`; P512 G64,
     `NNTR_HTP_PROFILE=2`, not speed cells): the `convert to registry`
     row (it includes the experts' file reads, so it is not 0; the FC
     requant is read from `requant=0`), the `M>1` rows (B: conv, dense,
     FC; Bfb: conv), Q's per-kind `graph: tokens=… pcyc/token` lines;
   * summary: `logs/speed.txt` (text vs A r1 of the same P and G: `same` /
     `same+1st` / `DIFF`), `every r2 text == its r1`, `logs/loops.txt`,
     `logs/ppl.txt`, `logs/texts.txt`, the VOID list, `logs/therm.log`.
6. Paste `logs/speed.txt`, the Q check lines of one cell per P, the
   profile rows, `logs/ppl.txt`, `logs/loops.txt` and `logs/texts.txt`
   below, fill the approval column, commit this file on the branch, push,
   and comment on #225 (`state:measured`). Logs stay in
   `/local/mnt/workspace/htp_moe/225/logs/` — say on which machine.

## Results (fill in)

Reference (other sittings, not comparable cell by cell; prefill / decode
tok/s): #222 sitting 2026-10-02 farm `R3CY205ZMND` (S25): CPU `q40` P64
244.3 / 53.6, 210.5 / 53.4, 210.5 / 49.4; P512 310.5 / 49.0, 307.7 / 46.9,
294.4 / 48.9; P1024 (`init_seq_len 1024`) 311.8 / 49.7, 317.8 / 49.1,
313.8 / 48.5 (G 64 / 512 / 1024). Anew (the keys with the re-quantized
copies, accuracy failed) P64 294.9 / 56.1, 254.0 / 55.6, 266.7 / 52.9;
P512 786.5 / 56.5, 761.9 / 56.4, 739.9 / 51.7; P1024 VOID. Qold (C = 28,
no keys) P512 556 / 33.7, 455 / 43.3, 531 / 43.3. Record sitting
2026-09-30 (old config): NPU hybrid decode 53.97 / 52.16 / 51.41. Goal:
decode ≥ 50 and above the CPU of the same sitting.

### The table of record (decode tok/s all / prefill tok/s; G64 = run 1)

| case | P | G 64 | G 512 | G 1024 |
|---|---|---|---|---|
| CPU only (A) | 64 | | | |
| CPU only (A) | 512 | | | |
| CPU only (A) | 1024 | | | |
| hybrid (B or Bfb: say which) | 64 | | | |
| hybrid | 512 | | | |
| hybrid | 1024 | | | |
| E2E one PD (Q28) | 64 | | | |
| E2E one PD | 512 | | | |
| E2E one PD | 1024 | | | |

### Every run

| run | prefill tok/s | decode tok/s (all) | decode (last 64) | peak RSS (KB) | text vs A r1 | loop (L1 / L2) | Q: calls/token, cpu fc skipped / token | fc wh arena_kib / heap_kib | Q: heap_used_kib, mapped_mib |
|---|---|---|---|---|---|---|---|---|---|
| A_P64_G64_r1 | | | | | (ref) | | — | — | — |
| B_P64_G64_r1 | | | | | | | — | | — |
| Q_P64_G64_r1 | | | | | | | | | |
| Q_P64_G64_r2 | | | | | | | | | |
| B_P64_G64_r2 | | | | | | | — | | — |
| A_P64_G512_r1 | | | | | (ref) | | — | — | — |
| B_P64_G512_r1 | | | | | | | — | | — |
| Q_P64_G512_r1 | | | | | | | | | |
| A_P64_G1024_r1 | | | | | (ref) | | — | — | — |
| B_P64_G1024_r1 | | | | | | | — | | — |
| Q_P64_G1024_r1 | | | | | | | | | |
| A_P512_G64_r1 | | | | | (ref) | | — | — | — |
| B_P512_G64_r1 | | | | | | | — | | — |
| Q_P512_G64_r1 | | | | | | | | | |
| Q_P512_G64_r2 | | | | | | | | | |
| B_P512_G64_r2 | | | | | | | — | | — |
| Aoff_P512_G64_r1 | | | | | | | — | (none) | — |
| A_P512_G512_r1 | | | | | (ref) | | — | — | — |
| B_P512_G512_r1 | | | | | | | — | | — |
| Q_P512_G512_r1 | | | | | | | | | |
| Aoff_P512_G512_r1 | | | | | | | — | (none) | — |
| A_P512_G1024_r1 | | | | | (ref) | | — | — | — |
| B_P512_G1024_r1 | | | | | | | — | | — |
| Q_P512_G1024_r1 | | | | | | | | | |
| A_P1024_G64_r1 | | | | | (ref) | | — | — | — |
| B_P1024_G64_r1 | | | | | | | — | | — |
| Q_P1024_G64_r1 | | | | | | | | | |
| Q_P1024_G64_r2 | | | | | | | | | |
| B_P1024_G64_r2 | | | | | | | — | | — |
| A_P1024_G512_r1 | | | | | (ref) | | — | — | — |
| B_P1024_G512_r1 | | | | | | | — | | — |
| Q_P1024_G512_r1 | | | | | | | | | |
| A_P1024_G1024_r1 | | | | | (ref) | | — | — | — |
| B_P1024_G1024_r1 | | | | | | | — | | — |
| Q_P1024_G1024_r1 | | | | | | | | | |

VOID (`logs/void_*`, error line + arena / heap lines; a B VOID names its
Bfb rows): ___

Device md5s (`logs/md5_device.log`, `logs/md5_models.log`): skel ___,
`libnntrainer.so` ___, sidecar ___, `cfg_new.json` ___, `cfg_off.json`
___, `cfg_fb.json` ___, NPU bin (both names) ___, CPU bin ___. Unit ___
(SoC ___), `MD5 OK` line seen: ___.

Profile rows (`prof_B` / `prof_Bfb`, `prof_Q`: `convert to registry`,
`M>1`, `M==1`, Q's per-kind `graph: tokens=… pcyc/token`; against #222's
Q28 FC + DENSE_FFN + LM_HEAD 9.2 ms a token, rule 59a): <paste>

## Accuracy: PPL (8 prompts, G 256; recorded, not gated: T2) and text approval

Prefill PPL (`[PPL] prompt … ppl=`) and decode PPL forced on A's
continuation (`[PPL] decode … source=file`; A's own run is `source=self`,
its p01 forced re-run `ppl_Af_p01` is the null check). Information for
the record and for the Gemma work; the pass / fail is the loop column and
the user's approval.

| prompt | A prefill PPL | Aoff prefill PPL | B prefill PPL | Q prefill PPL | A decode PPL (self) | Aoff decode (forced) | B decode (forced) | Q decode (forced) | loops (variant loops, A does not) |
|---|---|---|---|---|---|---|---|---|---|
| p01 | | | | | | | | | |
| p02 | | | | | | | | | |
| p03 | | | | | | | | | |
| p04 | | | | | | | | | |
| p05 | | | | | | | | | |
| p06 | | | | | | | | | |
| p07 | | | | | | | | | |
| p08 | | | | | | | | | |
| pooled | | | | | | | | | |

Null check (`ppl_Af_p01` decode PPL == `ppl_A_p01`'s): ___

Texts (G 64, run 1; `logs/texts.txt`). T2: approve unless the text loops
or leaves the prompt's context; differing from A is expected.

| variant | P | generated text | text approved (user: y/n) |
|---|---|---|---|
| A | 64 | <paste> | (reference) |
| B / Bfb | 64 | <paste> | |
| Q | 64 | <paste> | |
| A | 512 | <paste> | (reference) |
| Aoff | 512 | <paste> | |
| B / Bfb | 512 | <paste> | |
| Q | 512 | <paste> | |
| A | 1024 | <paste> | (reference) |
| B / Bfb | 1024 | <paste> | |
| Q | 1024 | <paste> | |

## Notes from the run

<machine the runner ran on, serial, uptime at start, thermal per block,
STOPs / reboots, VOIDs and whether Bfb became the hybrid case, anything
stale>
