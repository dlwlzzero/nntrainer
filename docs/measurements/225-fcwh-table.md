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

## Results (2026-10-06, farm `R3CY205ZMND`)

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
| CPU only (A) | 64 | 54.05 / 257.0 | 52.54 / 255.0 | 51.48 / 252.0 |
| CPU only (A) | 512 | 52.07 / 328.0 | 50.70 / 249.6 | 49.47 / 231.2 |
| CPU only (A) | 1024 | 47.58 / 248.3 | 48.82 / 311.2 | 47.41 / 275.7 |
| hybrid (B at P64 / P512; **Bfb** at P1024, B VOID there) | 64 | 59.59 / 277.1 | 58.01 / 272.3 | 54.74 / 283.2 |
| hybrid (B) | 512 | 55.17 / 727.3 | 55.72 / 728.3 | 51.43 / 738.8 |
| hybrid (Bfb) | 1024 | 52.03 / 663.2 | 51.51 / 650.6 | 48.74 / 649.7 |
| E2E one PD (Q28) | 64 | 42.81 / 275.9 | 49.34 / 248.1 | 48.49 / 248.1 |
| E2E one PD (Q28) | 512 | 42.78 / 751.8 | 45.94 / 705.2 | 45.61 / 724.2 |
| E2E one PD (Q28) | 1024 | 37.83 / 699.0 | 43.93 / 742.0 | 43.52 / 685.4 |

### Every run

"text vs A r1" is recomputed with every `[HTP]` banner removed from the
generated text: the `dspq: on queue=0x…` and `graph: init …` banners print
in the middle of the text (stdout not line-terminated), and the runner's
`gen()` kept them, so its `speed.txt` reads DIFF for every B / Bfb / Aoff /
Q run and its `every r2 text == its r1` check reports `BAD … got '2'` (B
P64 / P512: only the queue address differs; with the banner removed both
r2 texts are byte-identical to r1). "1st token shifted" = one side starts
with the prefill's token (init_seq_len, see What to expect). Loop columns
are `logs/loops.txt` as written (banners included; one line per run).

| run | prefill tok/s | decode tok/s (all) | decode (last 64) | peak RSS (KB) | text vs A r1 | loop (L1 / L2) | Q: calls/token, cpu fc skipped / token | fc wh arena_kib / heap_kib | Q: heap_used_kib, mapped_mib (fc + pool) |
|---|---|---|---|---|---|---|---|---|---|
| A_P64_G64_r1 | 257.028 | 54.0541 | 54.0541 | 5296776 | (ref) | 1 / 1.00 loop | — | (none) | — |
| B_P64_G64_r1 | 277.056 | 59.5903 | 59.5903 | 5272764 | DIFF (common 9 of 291 chars) | 1 / 1.00 loop | — | 145408 / 75776 | — |
| Q_P64_G64_r1 | 275.862 | 42.8094 | 42.8094 | 816828 | DIFF (common 9 of 291 chars) | 1 / 0.72 loop | 1.00, 32 | 221184 / 0 | 94604, 192 + 3520 |
| Q_P64_G64_r2 | 280.702 | 49.3066 | 49.3066 | 817396 | DIFF (common 9 of 291 chars) | 1 / 0.72 loop | 1.00, 32 | 221184 / 0 | 94604, 192 + 3520 |
| B_P64_G64_r2 | 284.444 | 59.0951 | 59.0951 | 5066440 | DIFF (common 9 of 291 chars) | 1 / 1.00 loop | — | 145408 / 75776 | — |
| A_P64_G512_r1 | 254.98 | 52.5398 | 51.8639 | 5906272 | (ref) | 1 / 1.00 loop | — | (none) | — |
| B_P64_G512_r1 | 272.34 | 58.0104 | 53.6463 | 4914184 | DIFF (common 9 of 2307 chars) | 1 / 1.00 loop | — | 145408 / 75776 | — |
| Q_P64_G512_r1 | 248.062 | 49.3446 | 49.5356 | 811348 | DIFF (common 9 of 2307 chars) | 1 / 1.00 loop | 1.00, 32 | 221184 / 0 | 94604, 192 + 3520 |
| A_P64_G1024_r1 | 251.969 | 51.4832 | 49.4973 | 5913876 | (ref) | 1 / 1.00 loop | — | (none) | — |
| B_P64_G1024_r1 | 283.186 | 54.7359 | 50.1567 | 5269308 | DIFF (common 9 of 4607 chars) | 1 / 1.00 loop | — | 145408 / 75776 | — |
| Q_P64_G1024_r1 | 248.062 | 48.4917 | 47.3723 | 810340 | DIFF (common 9 of 4607 chars) | 1 / 1.00 loop | 1.00, 32 | 221184 / 0 | 94604, 192 + 3520 |
| A_P512_G64_r1 | 327.995 | 52.0749 | 52.0749 | 5915348 | (ref) | 1 / 0.00 | — | (none) | — |
| B_P512_G64_r1 | 727.273 | 55.1724 | 55.1724 | 5294312 | DIFF (common 1 of 259 chars) | 1 / 0.00 | — | 145408 / 75776 | — |
| Q_P512_G64_r1 | 751.836 | 42.7807 | 42.7807 | 828604 | DIFF (common 2 of 259 chars, 1st token shifted) | 1 / 0.23 | 1.00, 32 | 221184 / 0 | 94604, 192 + 3520 |
| Q_P512_G64_r2 | 752.941 | 44.4753 | 44.4753 | 828408 | DIFF (common 2 of 259 chars, 1st token shifted) | 1 / 0.23 | 1.00, 32 | 221184 / 0 | 94604, 192 + 3520 |
| B_P512_G64_r2 | 733.524 | 55.3633 | 55.3633 | 5117268 | DIFF (common 1 of 259 chars) | 1 / 0.00 | — | 145408 / 75776 | — |
| Aoff_P512_G64_r1 | 527.291 | 53.9629 | 53.9629 | 5005808 | DIFF (common 216 of 259 chars, 1st token shifted) | 1 / 0.00 | — | (none) | — |
| A_P512_G512_r1 | 249.634 | 50.6981 | 49.961 | 5926536 | (ref) | 1 / 1.00 loop | — | (none) | — |
| B_P512_G512_r1 | 728.307 | 55.7188 | 54.5145 | 5296620 | DIFF (common 1 of 2106 chars) | 1 / 0.61 loop | — | 145408 / 75776 | — |
| Q_P512_G512_r1 | 705.234 | 45.9399 | 46.4104 | 828580 | DIFF (common 2 of 2106 chars, 1st token shifted) | 17 / 1.00 loop | 1.00, 32 | 221184 / 0 | 94604, 192 + 3520 |
| Aoff_P512_G512_r1 | 559.563 | 52.5829 | 50.3145 | 5291684 | DIFF (common 216 of 2106 chars, 1st token shifted) | 1 / 0.95 loop | — | (none) | — |
| A_P512_G1024_r1 | 231.151 | 49.4686 | 47.7612 | 5957744 | (ref) | 1 / 1.00 loop | — | (none) | — |
| B_P512_G1024_r1 | 738.817 | 51.4263 | 47.7612 | 5292360 | DIFF (common 1 of 4221 chars) | 1 / 1.00 loop | — | 145408 / 75776 | — |
| Q_P512_G1024_r1 | 724.187 | 45.6064 | 44.9123 | 827448 | DIFF (common 2 of 4221 chars, 1st token shifted) | 35 / 1.00 loop | 1.00, 32 | 221184 / 0 | 94604, 192 + 3520 |
| A_P1024_G64_r1 | 248.303 | 47.5836 | 47.5836 | 5907328 | (ref) | 1 / 0.00 | — | (none) | — |
| B_P1024_G64_r1 | VOID | | | | | | — | 145408 / 75776 | — |
| Bfb_P1024_G64_r1 | 663.212 | 52.0325 | 52.0325 | 5294484 | DIFF (common 8 of 260 chars) | 6 / 1.00 loop | — | 139264 / 8192 | — |
| Q_P1024_G64_r1 | 698.976 | 37.8251 | 37.8251 | 860632 | DIFF (common 3 of 260 chars) | 1 / 0.00 | 1.00, 32 | 221184 / 0 | 94604, 192 + 3520 |
| Q_P1024_G64_r2 | 754.606 | 38.7175 | 38.7175 | 860184 | DIFF (common 3 of 260 chars) | 1 / 0.00 | 1.00, 32 | 221184 / 0 | 94604, 192 + 3520 |
| Bfb_P1024_G64_r2 | 686.327 | 51.4469 | 51.4469 | 5354684 | DIFF (common 8 of 260 chars) | 6 / 1.00 loop | — | 139264 / 8192 | — |
| A_P1024_G512_r1 | 311.246 | 48.8177 | 47.69 | 5905860 | (ref) | 1 / 0.16 | — | (none) | — |
| Bfb_P1024_G512_r1 | 650.572 | 51.5091 | 47.5483 | 5304768 | DIFF (common 8 of 2319 chars) | 62 / 1.00 loop | — | 139264 / 8192 | — |
| Q_P1024_G512_r1 | 742.029 | 43.9296 | 44.2294 | 860736 | DIFF (common 3 of 2319 chars) | 1 / 1.00 loop | 1.00, 32 | 221184 / 0 | 94604, 192 + 3520 |
| A_P1024_G1024_r1 | 275.714 | 47.4118 | 45.7797 | 5903956 | (ref) | 1 / 0.82 loop | — | (none) | — |
| Bfb_P1024_G1024_r1 | 649.746 | 48.7364 | 45.2617 | 5323028 | DIFF (common 8 of 4627 chars) | 126 / 1.00 loop | — | 139264 / 8192 | — |
| Q_P1024_G1024_r1 | 685.408 | 43.5245 | 42.0499 | 859036 | DIFF (common 3 of 4627 chars) | 1 / 1.00 loop | 1.00, 32 | 221184 / 0 | 94604, 192 + 3520 |

VOID: `B_P1024_G64_r1` (`logs/void_B_P1024`), so B's P1024 cells ran as
**Bfb** (`Bfb_P1024_G64_r1` / `_r2`, `Bfb_P1024_G512_r1`,
`Bfb_P1024_G1024_r1`) and Bfb is the hybrid case at P1024 (user decision
(b)); B did not die at P64 / P512, and `prof_B` is B:

```
[!] FATAL ERROR: nntr_hvx_mm_u8i4_layer failed: err=-2147482112 (M=1024 K=2048 N=3072 handles=3)
    [HTP] fc wh: file=../models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm_fcwh.bin handles=112 arena_kib=145408 heap_kib=75776 requant=0
    [HTP] arena: chunks unmapped 15/15 mib=3840 (release rc=0x0 put=15)
```

`0x80000600` is `AEE_ERPC` (LEDGER rule 38), not #222's `AEE_ENOMEMORY`
`0x80000402`. The failing call is M = 1024 at K = 2048, N = 3072, the
attention qkv projection: PR 1 chunks the conv / dense / MoE prefill calls
at 512 rows, and this FC call reached the DSP unchunked (reading of the
log line, not checked in code). Q ran the same keys at P1024 and loaded.
Bfb's banner: `fc wh: … arena_kib=139264 heap_kib=8192 requant=0` (conv
images only).

Device md5s (`logs/md5_device.log`, `logs/md5_models.log`): skel v79
`5508180c…` (pushed as `libnntr_hvx_skel.so`), `libnntrainer.so`
`701cd42e…`, sidecar `71812a91…`, `cfg_new.json` `3f6808e3…`,
`cfg_off.json` `01b492ad…`, `cfg_fb.json` `49fcc38a…`, NPU bin (both names)
`7b7867fa…`, CPU bin `d28f55c5…`. Unit `R3CY205ZMND` = SM-S938N (SoC
SM8750, v79); `MD5 OK (app, extra, sidecar)` seen. The app set is a
rebuild (see Notes), so its md5s are not the staged `dd24b4fc…` set's.

Profile rows (`prof_B`, `prof_Q`, P512 G64, `NNTR_HTP_PROFILE=2`; full
lines in `logs/prof_*.log`):

```
prof_B: 709.141 TPS 51.9903 TPS 51.9903 TPS  rss=5169748KB
[HTP] fc wh: file=../models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm_fcwh.bin handles=112 arena_kib=145408 heap_kib=75776 requant=0
prof_Q: 673.684 TPS 40.9469 TPS 40.9469 TPS calls/token=1.00 rss=828296KB
[HTP] fc wh: file=../models/q40-qs4cx-wh/nntr_lfm2_8b_a1b_q40_arm_fcwh.bin handles=112 arena_kib=221184 heap_kib=0 requant=0
[HTP] graph: q4m1 weights=67 handles=8 feed=vtcm wh_handles=112
[HTP] e2e: fc arena weights=67 handles=8 attach_mib=140.6 chunks=1 mapped_mib=192 feed=vtcm load_ms=432.4 lanes=6,3 s1_arena_mib=3520 s1_heap_kib=61211
[HTP] token driver: pool misses=103 misses/token=1.61 miss_wait_us/token=549.0 rounds=68 arm_ms/round=0.933 pgpgin_mib=5.2
[HTP] e2e: close mapped_mib=192.61 unmap_fail=0 detach_fail=0 heap_used_kib=94604 (info rc 0x0)
prof_B M>1 rows present: conv dense FC  (B: conv dense FC; Bfb: conv)
[HTP-PROFILE]   convert to registry:        0.0 ms  (0.00 ms/weight)
[HTP-PROFILE] layer calls (M==1 is decode's shape)
[HTP-PROFILE]   K=2048  N=2048  M>1        calls=23      rows=11776    host=    362.2 ms (15746.1 us/call)  dsp=14921.8 us/call (94.8%) transport=  824.3 us/call  [quant 322.9 gather 34.0 requant 98.6 swiglu(hidden) 28216.8 dequant 322.4 acc 3133.1 drain 148.1+28.5 push 46.3 scatter 69.0 alloc 148.1 stage 435.1 mm 9844.5 | rest<=291.1 (1.8% of host) blocks=1104 m1_gemv=0/23 feed=0/23 dmaq=0.00]
[HTP-PROFILE]   K=2048  N=2048  M>1 dense  calls=3       rows=1536     host=     31.7 ms (10556.0 us/call)  dsp=10126.7 us/call (95.9%) transport=  429.3 us/call  [quant 285.3 gather 62.0 requant 68.7 swiglu(hidden) 24628.3 dequant 357.7 acc 2085.3 drain 2.0+3.3 push 4.3 scatter 54.3 alloc 0.3 stage 392.3 mm 6633.3 | rest<=177.7 (1.7% of host) blocks=96 m1_gemv=0/3 feed=0/3 dmaq=0.00]
[HTP-PROFILE]   K=2048  N=2048  M>1 conv   calls=19      rows=9728     host=     96.9 ms ( 5098.6 us/call)  dsp= 4624.1 us/call (90.7%) transport=  474.6 us/call  [quant 246.1 gather 244.3 requant 0.4 swiglu(hidden) 3750.7 dequant 511.5 acc 753.6 drain 0.9+36.7 push 1.1 scatter 0.0 alloc 0.7 stage 295.9 mm 2465.2 | rest<=67.6 (1.3% of host) blocks=152 m1_gemv=0/19 feed=0/19 dmaq=0.00]
[HTP-PROFILE]   K=2048  N=2048  M>1 FC     calls=7       rows=3584     host=     12.0 ms ( 1719.4 us/call)  dsp= 1300.9 us/call (75.7%) transport=  418.6 us/call  [quant 433.9 gather 0.0 requant 0.0 swiglu 0.0 dequant 6.3 acc 192.7 drain 29.0+0.0 push 0.0 scatter 0.0 alloc 0.0 stage 0.0 mm 0.0 | rest<=639.0 (37.2% of host) blocks=0 m1_gemv=0/7 feed=0/7 dmaq=0.00]
[HTP-PROFILE]   K=2048  N=2048  M==1       calls=1408    rows=1408     host=    572.3 ms (  406.5 us/call)  dsp=  395.1 us/call (97.2%) transport=   11.4 us/call  [quant 6.5 gather 0.0 requant 5.9 swiglu(hidden) 1312.1 dequant 0.0 acc 0.0 drain 0.0+0.0 push 0.0 scatter 4.2 alloc 0.1 stage 3.9 mm 371.7 | rest<=2.9 (0.7% of host) blocks=0 m1_gemv=1408/1408 feed=1408/1408 dmaq=1.00]
[HTP-PROFILE]   K=2048  N=3072  M>1 FC     calls=6       rows=3072     host=     13.8 ms ( 2307.5 us/call)  dsp= 1780.2 us/call (77.1%) transport=  527.3 us/call  [quant 416.5 gather 0.0 requant 0.0 swiglu 0.0 dequant 6.7 acc 288.3 drain 93.7+0.0 push 0.0 scatter 0.0 alloc 0.0 stage 0.0 mm 0.0 | rest<=975.0 (42.3% of host) blocks=0 m1_gemv=0/6 feed=0/6 dmaq=0.00]
prof_Q M>1 rows present: conv dense FC  (B: conv dense FC; Bfb: conv)
[HTP-PROFILE]   convert to registry:      618.5 ms  (0.85 ms/weight)
[HTP-PROFILE] layer calls (M==1 is decode's shape)
[HTP-PROFILE]   K=2048  N=2048  M>1        calls=23      rows=11776    host=    371.5 ms (16153.2 us/call)  dsp=15053.3 us/call (93.2%) transport= 1100.0 us/call  [quant 301.7 gather 59.0 requant 137.9 swiglu(hidden) 28735.3 dequant 333.1 acc 3135.4 drain 114.6+25.3 push 47.4 scatter 70.2 alloc 134.8 stage 453.3 mm 9947.0 | rest<=293.4 (1.8% of host) blocks=1104 m1_gemv=0/23 feed=0/23 dmaq=0.00]
[HTP-PROFILE]   K=2048  N=2048  M>1 dense  calls=3       rows=1536     host=     33.0 ms (10987.3 us/call)  dsp=10348.3 us/call (94.2%) transport=  639.0 us/call  [quant 297.3 gather 65.0 requant 77.0 swiglu(hidden) 24654.7 dequant 357.7 acc 2095.0 drain 24.0+118.3 push 5.7 scatter 57.7 alloc 1.0 stage 400.0 mm 6672.3 | rest<=177.3 (1.6% of host) blocks=96 m1_gemv=0/3 feed=0/3 dmaq=0.00]
[HTP-PROFILE]   K=2048  N=2048  M>1 conv   calls=19      rows=9728     host=     98.1 ms ( 5164.0 us/call)  dsp= 4613.8 us/call (89.3%) transport=  550.2 us/call  [quant 282.7 gather 150.8 requant 0.4 swiglu(hidden) 3808.5 dequant 555.7 acc 765.7 drain 1.5+36.4 push 1.0 scatter 0.0 alloc 0.6 stage 302.9 mm 2444.9 | rest<=71.1 (1.4% of host) blocks=152 m1_gemv=0/19 feed=0/19 dmaq=0.00]
[HTP-PROFILE]   K=2048  N=2048  M>1 FC     calls=7       rows=3584     host=     12.6 ms ( 1800.0 us/call)  dsp= 1303.7 us/call (72.4%) transport=  496.3 us/call  [quant 447.6 gather 0.0 requant 0.0 swiglu 0.0 dequant 6.7 acc 191.1 drain 10.6+0.0 push 0.0 scatter 0.0 alloc 0.0 stage 0.0 mm 0.0 | rest<=647.7 (36.0% of host) blocks=0 m1_gemv=0/7 feed=0/7 dmaq=0.00]
[HTP-PROFILE]   K=2048  N=3072  M>1 FC     calls=6       rows=3072     host=     14.7 ms ( 2453.2 us/call)  dsp= 1819.7 us/call (74.2%) transport=  633.5 us/call  [quant 465.2 gather 0.0 requant 0.0 swiglu 0.0 dequant 5.7 acc 287.2 drain 86.3+0.0 push 0.0 scatter 0.0 alloc 0.0 stage 0.0 mm 0.0 | rest<=975.3 (39.8% of host) blocks=0 m1_gemv=0/6 feed=0/6 dmaq=0.00]
[HTP] graph: tokens=64 pcyc/token=43973168 wait_us/token=0.0
[HTP] graph per-kind pcyc/token: RMSNORM=690096(0.331ms) FC=7953470(3.819ms) CONV1D_GATE=995059(0.478ms) QK_NORM=195707(0.094ms) ROPE=58268(0.028ms) ATTN_M1=1291267(0.620ms) ADD=188622(0.091ms) ROUTER_TOPK=1409490(0.677ms) MOE=22798520(10.946ms) DENSE_FFN=2200554(1.057ms) LM_HEAD=6192115(2.973ms) | wall_ms/token=21.275 mhz=2083 spin_us=0
```

## Accuracy: PPL (8 prompts, G 256; recorded, not gated: T2) and text approval

Prefill PPL (`[PPL] prompt … ppl=`) and decode PPL forced on A's
continuation (`[PPL] decode … source=file`; A's own run is `source=self`,
its p01 forced re-run `ppl_Af_p01` is the null check); top1 in brackets.
B in this block is B (P512 loads). Pooled = exp of the token-weighted mean
nll over the 8 prompts.

| prompt | A prefill PPL | Aoff prefill PPL | B prefill PPL | Q prefill PPL | A decode PPL (self) | Aoff decode (forced) | B decode (forced) | Q decode (forced) | loops (variant loops, A does not) |
|---|---|---|---|---|---|---|---|---|---|
| p01 | 109.759 | 115.095 | 94.5355 | 94.5355 | 1.17242 (256/256) | 1.19862 (250/256) | 1.1753 (249/256) | 1.20026 (241/256) | none |
| p02 | 32.6117 | 26.1919 | 30.4323 | 30.4323 | 1.13027 (256/256) | 1.13134 (253/256) | 1.15269 (251/256) | 1.17916 (248/256) | none |
| p03 | 64.9281 | 111.365 | 183.353 | 183.353 | 1.06884 (256/256) | 1.06488 (256/256) | 1.1326 (256/256) | 1.09162 (255/256) | none |
| p04 | 270.551 | 264.051 | 355.64 | 355.64 | 1.28385 (256/256) | 1.36414 (236/256) | 1.67123 (222/256) | 1.79102 (218/256) | none |
| p05 | 175.76 | 167.568 | 220.315 | 220.315 | 1.13511 (256/256) | 1.20173 (246/256) | 1.34009 (231/256) | 1.26154 (240/256) | none |
| p06 | 68.3631 | 76.4529 | 54.6975 | 54.6975 | 1.10811 (256/256) | 1.14267 (250/256) | 1.19381 (247/256) | 1.32659 (243/256) | none |
| p07 | 29.6229 | 32.2619 | 35.0692 | 35.0692 | 1.58811 (256/256) | 1.68107 (226/256) | 1.85281 (209/256) | 1.98865 (198/256) | none |
| p08 | 4114.37 | 4370.54 | 4704.72 | 4704.72 | 1.04105 (256/256) | 1.03717 (255/256) | 1.05043 (255/256) | 1.07486 (254/256) | none |
| pooled (token-weighted) | 86.44 | 90.31 | 94.08 | 94.08 | 1.1809 | 1.2139 | 1.2965 | 1.3318 | none |

Null check: `ppl_Af_p01` decode PPL 1.17242 top1 256/256 == `ppl_A_p01` 1.17242 top1 256/256: **equal**

Texts (G 64, run 1, `[HTP]` banners removed; ⏎ = newline). T2: approve
unless the text loops or leaves the prompt's context; differing from A is
expected. Approval column: the user's.

| variant | P | generated text | text approved (user: y/n) |
|---|---|---|---|
| A | 64 | s in the early afternoon, and the smell of salt and diesel hangs in the early afternoon, and the smell of salt and diesel hangs in the early afternoon, and the smell of salt and diesel hangs in the early afternoon, and the smell of salt and diesel hangs in the early afternoon, and the smell | (reference) |
| B | 64 | s in the air, and the smell of the smell of the smell of the smell of the smell of the smell of the smell of the smell of the smell of the smell of the smell of the smell of the smell of the smell of the smell of the smell of the smell of the smell of the smell of the smell |  |
| Q | 64 | s in the air, and for most of its history it has lived by the tide, and for most of its history it has lived by the tide, and for most of its history it has lived by the tide, and for most of its history it has lived by the tide, and for most of its history it has |  |
| A | 512 |  town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum, a lifeboat station, and a lifeboat itself | (reference) |
| Aoff | 512 |  The town has a single main street that climbs from the harbour to a stone church at the top of the hill, and along it stand a bakery, a hardware shop, two pubs, a post office that also sells fishing line, a small museum that opens only on summer weekends, and a lifeboat station |  |
| B | 512 |  In winter the wind comes straight off the water and the streets empty by four in the afternoon, but in summer the population nearly doubles as visitors arrive to walk the cliff paths, watch the seabirds, and eat fish and chips on the harbour wall while the gulls circle overhead hoping for scraps. The people of Ardley |  |
| Q | 512 |  In the same style, add more detail about its history, its people, its weather and the seasons, and do not stop until you are told to. In the same style, add more detail about its people, its weather and the seasons, and do not stop until you are told to. In the same style, add |  |
| A | 1024 | . ⏎  ⏎ Now, we have a huge amount of text. The user has asked for a single JSON object with those keys. We need to extract the information from the order note below. The order note is the entire text. The keys: "customer_name" (string), "email" (string), "items" ( | (reference) |
| Bfb | 1024 | . ⏎  ⏎ Now, I have a huge amount of data to process. I will have to do it all. I will have to do it all. I will have to do it all. I will have to do it all. I will have to do it all. I will have to do it all. I will have |  |
| Q | 1024 | . ⏎  ⏎ Thus, the final output is a JSON object with these keys: "customer_name", "email", "items", "delivery_date", "express", "total_eur". The values are to be filled from the order note. The order note does not provide explicit values for these keys, so we must infer |  |

Loops where the variant loops and A does not (`logs/loops.txt`):
Bfb P1024 G64 (r1, r2: L1run 6), Bfb P1024 G512 (L1run 62), Q P1024 G512
(L2 1.00; A 0.16). None at P512 G64, none in the PPL block; P64's prompt
asks to continue without stopping and A loops there too (no variant
fails at P64 by this rule).

## Notes from the run

* Runner on the workstation (`/local/mnt/workspace/htp_moe/225/`, logs in
  `logs/`), unit `R3CY205ZMND` (S25 Ultra, SM8750) through the ADF SSH
  bridge (an `adb` shim: `shell` pushed as a script and run with `sh`;
  reboot refused by the shim). 2026-10-06 15:56:15 – 16:36:33 KST, one
  invocation, no STOP, 1 BAD (the banner artifact above).
* **Not 5 min idle (rule 61):** the user rebooted the unit; the runner
  started at uptime 52 s. Q's first G64 runs read lower than their r2
  (P64 42.81 vs 49.31, P512 42.78 vs 44.48).
* **Set rebuilt, `md5.txt` `23000b96…` (not `dd24b4fc…`):** this
  workstation had no staged set and no sidecar. Built from
  `htp/225-fcwh-e2e` @ `51b2b4477` (code = `7efd22adf`) in a fresh
  worktree: skels `HEX_ARCH=v79/v81 ./test/htp/build.sh` (SDK 6.4.0.1,
  HexKL 6.4.0.1, `ARCH OK`, `UNDEFINED SYMBOLS OK (62 runtime imports)`),
  `build_android.sh --htp` (`-DENABLE_HEXKL=1`), then `225-stage.sh`.
  The sidecar was re-packed from the fp32 bin (`nntr_quantize_stream fp32
  --fc_dtype Q4_0 --moe_dtype QS4CX_WH --embd_dtype Q4_0 --lmhead_dtype
  Q4_0 --isa ARM --fc_wh_sidecar`): sidecar `71812a91…` and main file
  `7b7867fa…`, both byte-identical to the files of record. The models
  were pushed to the unit fresh (it held none). New `md5.txt`:

```
3f6808e30b6e16e1c8592fa39977814c  app/cfg_new.json
254485ff2f016526798b4c114be91b0a  app/libcausallm_core.so
9e7ad51b5696edd53500a6da4f24ae66  app/libccapi-nntrainer.so
b1586b9b512712800fd36a24abac1c0a  app/libc++_shared.so
701cd42e260f0a7612a5a5f062dbfef5  app/libnntrainer.so
5508180c2449fdb210538acea99d4fa0  app/libnntr_hvx_skel.v79.so
02021bd855955b73b8dedad1049747b4  app/libnntr_hvx_skel.v81.so
0ad4e22a70e4f135bce38ad8fd1e001b  app/libsdkl.so
2938a3c644d3754af2a4fe9d79413fb6  app/nntrainer_causallm
fc65c1588dc66dd764c7013fe96cbb75  app/p01.txt
7d2a17d3cc097d0567c9463c5eaeff88  app/p02.txt
c804b037d3bb87bf09259504fcae1c79  app/p03.txt
c5d4595b859a4e6abc3a035353aac180  app/p04.txt
a21ecdeea9d43d1b1366bde523347a62  app/p05.txt
6efd387197d6078366152f83a9b72a69  app/p06.txt
db0251e1bc57f072f815802cd93c260d  app/p07.txt
67b657c1261c2c12022907c2117a9a54  app/p08.txt
42595651ef514e155887eb31f421b2dc  app/page_cache_evict
71812a91d5acdbe9e026c479db8e275e  model/nntr_lfm2_8b_a1b_q40_arm_fcwh.bin
```

* Thermal (`logs/therm.log`, battery °C×10 / zone0 m°C): t0 270 / 35600,
  after P64 325 / 59300, after P512 329 / 58100, after P1024 331 / 58100,
  end 346 / 55800; every block started at zone0 ≤ 35 °C (`cool`).
