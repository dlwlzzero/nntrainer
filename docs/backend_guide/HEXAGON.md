# Hexagon Backend

The Hexagon (cDSP) backend runs a decoder-only LLM on the DSP at **graph
granularity**: the host hands over an op-list and every large buffer once,
and each `forward()` call executes the whole list — one FastRPC round-trip
(~0.26 ms) per prefill chunk or decode token, so RPC overhead is
negligible even at M=1.

Status: qwen3-0.6b (W8_CX int8 weights) runs end-to-end on an 8 Elite
cDSP from a packed weight image and matches the x86 reference executor at
the model level (+0.04 % PPL). Nothing in the nntrainer engine calls the
backend yet (`engine="htp"` is planned, section 9); the entry points are
the standalone harnesses in section 5.

---

## 1. Architecture

```mermaid
graph LR
  subgraph Host
    runner["HexagonRunner"]
    stub["QAIC stub (nntr_htp_stub.c)"]
    rbuf["RpcmemBuffer (dma-buf)"]
    runner --> stub
  end
  subgraph cDSP
    skel["QAIC skel (libnntr_htp_skel.so)"]
    exec["executor.c (glue, n_ops==0 dummy path)"]
    graphx["htp_graph (validate, scratch, dispatch)"]
    ops["9 HVX op kernels (ops/, hvx/)"]
    wp["QuRT worker pool"]
    vtcm["VTCM + user-DMA"]
    hmap["HAP_mmap view"]
    skel --> exec
    exec --> graphx
    graphx --> ops
    ops --- wp
    ops --- vtcm
    exec --- hmap
  end
  stub -->|FastRPC| skel
  rbuf -.-|shared zero-copy| hmap
```

The host side is arm64 Android; the DSP side is hexagon v75/v79.

* **`nntr_htp.idl`** defines the wire interface. QAIC generates a host
  stub and a DSP skel from it at build time; neither is committed.
* **`nntr_htp_common.h`** is the op-list wire format (section 1.4), plain
  C compiled by both toolchains. Both sides pin `NNTR_HTP_ABI_VERSION`;
  `init()` performs a version handshake and rejects a mismatch with
  `AEE_EUNSUPPORTED` before touching anything else.
* **`RpcmemBuffer`** (host) is a move-only RAII wrapper around one rpcmem
  (dma-buf) allocation — memory the CPU and DSP share zero-copy.
* **`HexagonRunner`** (host) owns one `remote_handle64` session:
  `create()` → `init()` → `forward()`* → destructor closes. `create()`
  returning `nullptr` means "no usable DSP"; callers take the CPU path.
* **`executor.c`** (DSP) is the FastRPC glue: validates the op-list, maps
  the handed-off buffers persistently, and builds an `htp_graph` that
  `forward()`/`forward_debug()` delegate to. With `n_ops == 0` it keeps
  the M1 dummy pattern (a deterministic fill mixing in `weights[0]`) so
  the round-trip test can prove host-written data is visible through the
  mapping.
* **`htp_graph`** (DSP) owns what the kernels share: validates the
  op-list once at `init()`, sizes quantization/attention scratch from an
  op scan, acquires VTCM best-effort (DDR fallback), and per call
  dispatches the ops sequentially through the kind table. Each kernel
  fans out over the **QuRT worker pool** (one worker per HVX unit) and
  returns at a barrier.

### 1.1 Buffer strategy: hand off once, map forever

WEIGHTS / KV / ACT cross the boundary **once**, at `init()`, as dma-buf
fds; the DSP maps them with `HAP_mmap` for the session lifetime.
`forward()` carries only `token_ids` in and `logits` out — the FastRPC
driver manages coherency for those small sequence arguments.

Two non-obvious mechanics, both learned during bring-up:

1. **A raw fd number is meaningless to the DSP.** The host must first
   register it with `fastrpc_mmap(domain, fd, addr, 0, size,
   FASTRPC_MAP_FD_DELAYED)`; otherwise `HAP_mmap` fails with
   `AEE_ENOMEMORY`. Re-registering returns `AEE_EALREADY`, which
   `HexagonRunner::init()` treats as success so re-init works.
2. **The weights buffer is additionally passed as an in-sequence** in the
   same `init()` call: the driver performs the one-time CPU cache flush
   for in-parameters, the DSP ignores the sequence and keeps only the fd
   mapping. Weight content must be final before `init()`.

598 MB WEIGHTS + 224 MB KV + 4 MB ACT allocate fine as three rpcmem
buffers; `init()` takes ~1 s.

### 1.2 Session setup: unsigned PD

`create()` requests an unsigned protection domain via
`remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, ...)` before
opening the session — HVX needs no privileges, and this avoids per-device
testsig installation. The call is best-effort; `remote_handle64_open` is
the real gate.

### 1.3 Error convention

DSP methods return AEE codes (`AEEStdErr.h`) unchanged to the host. **rout
parameters are not copied back on failure** — `dsp_abi_version` reads 0
when `init()` fails; that is expected, not a marshalling bug.

### 1.4 Op-list wire format (ABI v3)

The op-list passed to `init()` is one 64-byte `nntr_htp_oplist_header`
(magic, version, `n_ops`, model shape — layers/heads/dims/`max_seq`/
`max_chunk`) followed by `n_ops` × 64-byte `nntr_htp_op_desc` records.
Each descriptor names an op kind, its m/k/n shape (`m == 0` means "the
per-call token count"), and up to four tensor references — a buffer id
(WEIGHTS / KV / ACT / TOKENS / LOGITS) plus a 128-byte-aligned offset.

| kind | computes |
|------|----------|
| `EMBED` | int8 embedding row gather + dequant → fp16 |
| `RMSNORM` | RMS norm, optional per-head QK-Norm (`FLAG_PER_HEAD`) |
| `MATMUL_W8A8` | per-token dynamic-quant int8×int8 matmul → fp16 |
| `MATMUL_W8A16` | fp16 activation × int8 weight, fp32 accumulate → fp16 (no activation quant; `down_proj`) |
| `ROPE` | rotary embedding on q/k in place, precomputed cos/sin |
| `ATTN` | causal GQA attention against the persistent KV cache |
| `SILU_MUL` | SiLU(gate) ⊙ up |
| `ADD` | elementwise residual add |
| `MATMUL_LOGITS` | last-token int8 matmul → fp32 logits |

`init()` validates everything up front (header, kinds, alignment, every
tensor ref bounds-checked against the real buffer sizes) and rejects a
bad list with `AEE_EBADPARM`; `forward()` only checks runtime arguments
(token count ≤ `max_chunk`, position < `max_seq`, logits length).

v3 additions: `forward()` returns the DSP cycle count of the op loop
(`HAP_perf_get_pcycles`), and `forward_debug()` runs ops `[0, n)` only
and returns a byte slice of WEIGHTS/KV/ACT — the primitive behind the
divergence bisect (section 6). A partial run still updates KV, so debug
sessions restart from pos 0.

---

## 2. Graph lowering and weight image

Lowering is host-side, SDK-free C++: it turns a model's shape into the
op-list plus WEIGHTS/ACT layout plans, and packs the weight image at
those offsets. `graph_lowering.h` (backend) holds only model-agnostic
vocabulary — `HexModelConfig`, `HexModelWeights`, `HexLoweredGraph`,
`align128()`, `pack_weights()`; the qwen3 recipe `lower_qwen3()` lives
with the app under `Applications/CausalLM/hexagon/`. A second model adds
its own `*_lowering.cpp` and reuses `pack_weights()` unchanged.

`lower_qwen3(cfg)` is pure shape computation (reads no weights);
`pack_weights(g, cfg, w, dst)` copies/converts each tensor of `w` to the
offsets `g.woff` already holds.

### 2.1 WEIGHTS image

A bump cursor lays out the image with every tensor 128B-aligned:

1. `embed` — int8 `[vocab][hidden]`, tied and reused as the
   `MATMUL_LOGITS` weight;
2. `embed_scale` — fp32 `[vocab]`;
3. `rope_table` — fp16 `[max_seq][cos64||sin64]`, angle
   `p * theta^(-2i/128)`, precomputed on the host;
4. `final_norm` — fp16 `[hidden]`;
5. per layer: `wq/wq_s`, `wk/wk_s`, `wv/wv_s`, `wo/wo_s`, `gate/gate_s`,
   `up/up_s`, `down/down_s`, then `attn_norm/ffn_norm/q_norm/k_norm`.

Projections are int8 N-major (`[N][K]`) with one fp32 scale per output
channel, copied byte-exact; norm gammas and the RoPE table are converted
to fp16. For qwen3-0.6b (28 layers, hidden 1024, 16/8 heads, head_dim
128, ffn 3072, vocab 151936, max_seq 2048) the image is exactly
**598,623,744 bytes** with no alignment padding.

### 2.2 ACT buffer

Nine 128B-aligned slots, each sized for `max_chunk` tokens, reused by
every layer (so `act_size` does not scale with `n_layers`):

| slot | per token | role |
|------|-----------|------|
| `x` | `hidden` fp16 | residual stream |
| `t` | `hidden` fp16 | post-norm scratch |
| `q` | `n_heads*head_dim` fp16 | query projection |
| `kb` / `vb` | `n_kv_heads*head_dim` fp16 | this layer's k / v projection |
| `ao` | `n_heads*head_dim` fp16 | attention output |
| `h2` | `hidden` fp16 | matmul-out / residual scratch |
| `g` / `u` | `ffn` fp16 | gate / up projection |

The KV cache is separate: `kv_size = 2 * n_layers * n_kv_heads * max_seq
* head_dim * 2` bytes (fp16 key + value).

### 2.3 Op sequence

`lower_qwen3()` emits `1 + 16 * n_layers + 2` ops (451 for qwen3-0.6b):
`EMBED`, then per layer

| # | kind | |
|---|------|--|
| 1 | RMSNORM | `x * attn_norm -> t` |
| 2–4 | MATMUL_W8A8 | `t * wq/wk/wv -> q/kb/vb` |
| 5–6 | RMSNORM | `q * q_norm`, `kb * k_norm` in place, `FLAG_PER_HEAD` |
| 7 | ROPE | q, kb in place via `rope_table` |
| 8 | ATTN | `q, kb, vb -> ao`, tagged with the layer's KV index |
| 9 | MATMUL_W8A8 | `ao * wo -> h2` |
| 10 | ADD | `x + h2 -> x` |
| 11 | RMSNORM | `x * ffn_norm -> t` |
| 12–13 | MATMUL_W8A8 | `t * gate/up -> g/u` |
| 14 | SILU_MUL | `g, u -> g` in place |
| 15 | MATMUL_W8A16 | `g * down -> h2` |
| 16 | ADD | `x + h2 -> x` |

then a final `RMSNORM` and `MATMUL_LOGITS` (weight refs point at
`embed`). Op 15 is W8A16 because the SwiGLU output is outlier-heavy:
per-token int8 there alone costs ~6 % PPL on qwen3-0.6b (x86 breakdown
on 128 tokens, reference 20.32: all-int8 21.72, down fp32 20.38, wo fp32
21.11, q/k/v/gate/up fp32 21.42, lm_head fp32 21.79).

### 2.4 Checkpoint and packed image files

`nntr_quantize` writes the W8_CX `.bin` (header-less, 598,230,528 B for
qwen3-0.6b): embedding, then per layer `attn_norm, wq, q_norm, wk,
k_norm, wv, wo, ffn_norm, up, gate, down`, then `output_norm` (2-D
tensors as int8 `[N][K]` + fp32 `[N]`, norms fp32). `Qwen3W8cxBin` mmaps
it and hands out non-owning pointers as a `HexModelWeights`.

`nntr_hexpack <bin> <prefix> [--layers N]` writes `<prefix>.hexw` (the
WEIGHTS image; 172,498,944 B for the 1-layer bring-up image) and
`<prefix>.hexcfg` (11 `key=value` lines = `HexModelConfig`). Every
consumer re-runs `lower_qwen3()` from the `.hexcfg`, so image and
op-list cannot drift.

---

## 3. Source layout

```
nntrainer/tensor/hexagon/
├── htp/                      # hexagon-clang (DSP side)
│   ├── nntr_htp.idl          # FastRPC interface (init/forward/forward_debug)
│   ├── nntr_htp_common.h     # op-list wire format v3 + validation (shared with host)
│   ├── executor.c            # FastRPC glue -> htp_graph (or n_ops==0 dummy path)
│   ├── htp_graph.{h,c}       # executor: pool, scratch, VTCM, dispatch, forward_upto
│   ├── worker_pool.{h,c}     # QuRT worker pool + barrier
│   ├── ops/                  # one kernel per op kind
│   │   ├── hvx-matmul.c      # W8A8 (DDR + VTCM/DMA), W8A16, LOGITS
│   │   ├── hvx-attn.c / hvx-rmsnorm.c / hvx-rope.c / hvx-embed.c
│   │   └── hvx-eltwise.c     # ADD + SILU_MUL
│   ├── hvx/                  # vector helpers: f16 math, quant; exp/inverse from ggml-hexagon
│   ├── hex/                  # scalar utils (ggml-hexagon)
│   └── dma/                  # user-DMA queue (ggml-hexagon)
├── host/                     # NDK clang, part of libnntrainer (enable-hexagon)
│   ├── rpcmem_allocator.{h,cpp}
│   ├── hexagon_runner.{h,cpp}
│   └── graph_lowering.{h,cpp}   # HexModelConfig/Weights, pack_weights(), f32_to_f16_bits
└── meson.build

Applications/CausalLM/hexagon/
├── qwen3_lowering.{h,cpp}    # lower_qwen3(): the op sequence of section 2.3
├── qwen3_w8cx_bin.{h,cpp}    # mmap reader for the W8_CX .bin
├── hex_image.{h,cpp}         # .hexcfg read/write + raw file helpers
└── hex_pack.cpp              # nntr_hexpack (meson target, host-only)

test/hexagon/
├── test_oplist_header.c      # x86: wire-format self-check
├── test_lowering.cpp         # x86: op-list, layouts, pack_weights, hexcfg round trip
├── test_w8cx_bin.cpp         # x86: .bin reader sanity on the real checkpoint
├── hexagon_ref_run.cpp       # x86: scalar reference executor of a packed image
├── hexagon_rpc_test.cpp      # device: M1 round-trip test (n_ops == 0)
├── hexagon_e2e_test.cpp      # device: runs a packed image through HexagonRunner
└── sim/                      # simulator golden tests
    ├── ref_ops.{h,c}         # scalar reference kernels (also used by hexagon_ref_run)
    ├── ref_fp16_x86.h        # __fp16 stand-in for gcc < 12 on x86
    └── test_*.c              # one file per test

tools/hexagon/
├── build_host_x86.sh         # x86 tests + nntr_hexpack + hexagon_ref_run (no SDK)
├── build_skel.sh             # qaic + hexagon-clang -> libnntr_htp_skel.so
├── build_sim_test.sh / run_sim_test.sh
├── build_host_test.sh        # NDK cross-build: hexagon_rpc_test + hexagon_e2e_test
├── run_device_test.sh / check_rpc_log.py / plot_rpc_latency.py   # M1 round-trip
├── run_e2e_test.sh           # push image + harness, run, pull dumps, capture FARF
├── make_tokens.py            # text -> int32 LE token file
└── find_divergence.py        # bisect the first op where DSP != x86 reference
```

`hvx/hvx-exp.h`, `hvx-inverse.h`, `hvx-base.h`, `hvx-floor.h`,
`hvx-types.h`, `hex/` and `dma/` are imported from llama.cpp's
ggml-hexagon backend (MIT); they keep their headers and are listed in
`NOTICE`.

---

## 4. Build integration (`enable-hexagon`)

```bash
./tools/package_android.sh . -Denable-hexagon=true -Dhexagon-sdk-root=$HEXAGON_SDK_ROOT
```

Prerequisites: Hexagon SDK 6.3.0.0+ (bring-your-own, as for QNN) and an
Android NDK. The option (default `false`, strict no-op when off):

* errors out unless `platform=android` and an SDK root is given;
* runs QAIC at **configure time** into `<builddir>/nntr_htp_generated/`
  (the Android build is meson-configure → ndk-build, so the stub path
  must be a plain string) and registers the `.idl` as a reconfigure
  trigger;
* adds `host/*.cpp` + the stub to `nntrainer_sources` and ships
  `libcdsprpc.so` as a prebuilt — a link stub only; at runtime the
  device's `/vendor/lib64/libcdsprpc.so` resolves under the same soname.

The DSP skel is **not** part of the meson build (section 5.3).

---

## 5. Build and run

### 5.1 x86 (no SDK, no device)

```bash
./tools/hexagon/build_host_x86.sh        # -> build_x86_hexagon/{test_lowering,test_w8cx_bin,nntr_hexpack,hexagon_ref_run}
./build_x86_hexagon/test_lowering                        # LOWER_TEST PASS
./build_x86_hexagon/test_w8cx_bin $W8CX.bin              # W8CX_BIN_TEST PASS
gcc -Wall -Werror -o /tmp/t test/hexagon/test_oplist_header.c && /tmp/t
ninja -C build_x86 Applications/CausalLM/nntr_hexpack    # same tool via meson

./build_x86_hexagon/nntr_hexpack $W8CX.bin /tmp/qwen3_full            # ~2 s
python3 tools/hexagon/make_tokens.py $HF_DIR text.txt /tmp/t.i32 --limit 512
./build_x86_hexagon/hexagon_ref_run /tmp/qwen3_full --tokens /tmp/t.i32 --eval
```

`hexagon_ref_run` interprets a packed image with the scalar `ref_*`
kernels — the same fp16 + per-token int8 math as the DSP — and is the
accuracy oracle. Modes: default (prefill in chunks, then greedy
`--steps N`), `--eval` (teacher-forced PPL, one token per step),
`--dump-op i --dump-out f` (run ops `[0, i)`, write op `i-1`'s output),
`--list-ops`. `ref_ops.c` is C shared with the simulator; on x86 it is
compiled as C++ so `__fp16` can be a conversion struct
(`ref_fp16_x86.h`) on gcc 11.

### 5.2 Simulator golden tests

```bash
source $HEXAGON_SDK_ROOT/setup_sdk_env.source
HEX_ARCH=v75 ./tools/hexagon/build_sim_test.sh    # -> build_hexagon/sim/libnntr_sim_test.so
HEX_ARCH=v75 ./tools/hexagon/run_sim_test.sh <name>
```

`hexagon-sim` boots a QuRT image and dispatches one test by name; a pass
prints `SIM_TEST <name> PASS`. The 13 tests (`smoke pool exp quant
matmul matmul_dma rmsnorm rope eltwise embed attn logits graph`) compare
each kernel — and `graph`, a full 2-layer prefill/decode plus partial
execution — against `ref_ops.c` with a mixed bound `|d| <= atol + rtol *
|ref|`; the integer paths are bit-exact by construction. All pass on
v75 and v79 (SDK 6.3.0.0, toolchain 8.8). The `run_main_on_hexagon`
image is picked per `HEX_ARCH`; `HEX_EXTRA_CFLAGS` appends compiler
flags to both sim and skel builds.

### 5.3 Device

```bash
source $HEXAGON_SDK_ROOT/setup_sdk_env.source          # + ANDROID_NDK
HEX_ARCH=v75 ./tools/hexagon/build_skel.sh             # -> build_hexagon/skel/libnntr_htp_skel.so (see section 7)
./tools/hexagon/build_host_test.sh                     # -> build_hexagon/host/{hexagon_rpc_test,hexagon_e2e_test}

./tools/hexagon/run_device_test.sh [serial]            # RPC_TEST PASS
python3 tools/hexagon/check_rpc_log.py logs/hexagon/device_test_<stamp>.log

./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full [serial] -- --tokens /tmp/t.i32 --eval
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full [serial] -- --tokens /tmp/t.i32 --chunk 128 --steps 64
```

The binaries link only host sources + stub with `-static-libstdc++`
(`libc++_shared.so` does not exist under `/data/local/tmp`). The run
scripts write `0x1f` into `<binary>.farf` on the device — without it
DSP FARF lines never reach logcat — and capture them into
`logs/hexagon/device_farf_<stamp>.log`. `run_e2e_test.sh` pushes the skel,
harness, image and token file only when the device copy differs in size
(the image is ~600 MB) and pulls back `--dump-out` files.

`hexagon_rpc_test` drives `init()` with `n_ops == 0` and verifies the
RPC/mapping contract: session open, rpcmem, ABI-mismatch rejection, fd
registration + `HAP_mmap`, the dummy pattern `token_ids[i%3] + pos + i +
weights[0]` (proving host-written data is visible), and 32 timed
round-trips. `hexagon_e2e_test` mirrors `hexagon_ref_run`'s modes so
the two outputs compare 1:1; every line starts with `E2E ` and each step
reports DSP pcycles and host wall time.

---

## 6. Debugging a divergence

```bash
python3 tools/hexagon/find_divergence.py /tmp/qwen3_full /tmp/t.i32 --serial <serial> --chunk 8
# -> FIRST_DIVERGENCE op=<i> kind=<k> layer=<l> max_abs=.. max_rel=..   or NO_DIVERGENCE
```

Both runners execute ops `[0, i)` for the same chunk at pos 0 and dump
op `i-1`'s output (`hexagon_ref_run --dump-op`, `hexagon_e2e_test
--dump-op --dump-buf --dump-off --dump-bytes` over `forward_debug`).
"Outputs agree" is monotone in `i`, so 451 ops take 9 comparisons.
The last op writes LOGITS, which `forward_debug` cannot dump; judge it by
the `--eval` PPL of both runners. Default tolerance is 0.1: per-token
int8 amplifies ~1e-4 fp16 noise 5–8× per matmul (measured on the 1-layer
image: ATTN rel-RMS 1e-4 → next W8A8 3e-3), so a tight tolerance flags
two correct implementations as diverged. Use a 1-layer image
(`nntr_hexpack --layers 1`) for fast iteration.

---

## 7. HVX kernel rules (8 Elite silicon)

Found with the 1-layer image + `find_divergence.py`; the v75/v79
simulators pass either way, so a device pass is not optional.

1. **IEEE-format HVX float instructions are not trustworthy on this
   part.** `Q6_Vhf_vadd_VhfVhf` returned all zeros on silicon; in the
   v79-native build `Q6_Wsf_vmpyacc_WsfVhfVhf` and chained
   `Q6_Vqf32_vadd_Vqf32Vqf32` produced inf in ATTN and the W8A16 dot
   (also on the v79 simulator, printf-sensitive). Kernels use only
   qf-format ops — `Wqf32_vmpy_VhfVhf`, `Vqf32_vadd/vsub_VsfVsf`,
   `Vqf32_vmpy_VsfVsf`, `Vsf_equals_Vqf32`, `Vhf_equals_Wqf32` — and the
   device skel is built with **`HEX_ARCH=v75`**, which runs unchanged on
   v79 silicon. The v79-native build remains a to-do.
2. **`Vhf_equals_Vqf16` after a qf16 multiply rounds badly** (v75 sim
   probe: correct RNE 57 %, truncation 21 %, worse 21 %) while
   `Vhf_equals_Wqf32` is exact RNE. RMSNORM, SILU_MUL, ROPE, ADD and ATTN
   compute in fp32 inside the op and narrow to fp16 once — the same
   contract as the reference. Truncating instead at every op boundary
   costs 2.7 % PPL.
3. **SILU clamps the exp argument** (`-g <= 80`): qwen3 layer 27 has
   |g| > 250, `exp(-g)` overflowed and the HVX reciprocal turned
   `1/(1+inf)` into NaN instead of 0.
4. The kernels require `-mhvx-ieee-fp` (toolchain 8.8) for the fp16
   intrinsics; both build scripts pass it.

---

## 8. Results

Device: Galaxy S25 (SM-S931N, SM8750 = 8 Elite, cDSP v79), v75 skel,
SDK 6.3.0.0, 2026-08-31. Dummy `forward()` round-trip (M1 test, 32
iterations): 231 / 255 / 284 µs min / median / max.

### 8.1 Accuracy

Text: `eval.txt` (387 tokens) and its first 128; long = *Pride and
Prejudice*, first 70k chars in 8 × 2048-token windows (torch proxy).

| baseline | 128 tok | 387 tok | 16,376 tok |
|---|---|---|---|
| ① fp32 (nntrainer `--eval` / torch) | 19.893 | 19.1652 | 26.4673 |
| ② fake-quant W8_CX | 20.3183 | 19.5977 | 26.5031 |
| ②' x86 reference, packed image | 20.3758 | **20.2718** (top-1 155) | — |
| ③ DSP, same image | 21.3479 | **20.2802** (top-1 154) | — |

* ③ vs ②' on 387 tokens: **+0.04 % PPL**. On 128 tokens they swing by
  several percent against each other for the amplification reason in
  section 6, so short-text logit gates between two correct
  implementations are not meaningful; the model-level number is.
* ③ vs ② (+3.5 %) is the activation-quantization cost of the w8a8
  design — ②' shows the same +3.4 % — not a kernel error.
* 1-layer image: device vs reference top-1 8/8, generated ids identical;
  per-op rel-RMS ≤ 1e-4 up to ATTN, bit-exact for EMBED / RMSNORM / W8A8.

### 8.2 Performance (untuned, no HAP_power vote)

`--chunk 128 --steps 64`; medians over decode steps 2–64, host wall time
around each RPC:

| input | prefill | Mcycles/tok | decode (n=1) | Mcycles/tok |
|---|---|---|---|---|
| 512 tok | 24.5 s → **20.9 tok/s** | 101 | 102.3 ms → **9.8 tok/s** | 211 |
| 1024 tok | 62.1 s → **16.5 tok/s** | 128 | 138.6 ms → **7.2 tok/s** | 290 |
| teacher-forced, 387 steps | — | — | ~85 ms → 11.7 tok/s | ~150 |

Prefill chunk time grows with position (128-token chunks at pos 0 / 384 /
896: 5.1 / 7.2 / 10.8 s) and so does decode: the attention score loop is
scalar over `hvx_dot_fp16` per (query, position), so ATTN — not weight
bandwidth (598 MB/token would allow ~60 tok/s) — dominates. The previous
per-op HexKL path decoded at ~0.16 tok/s.

---

## 9. Planned work

* **M5 performance**: HVX attention score/V kernel first, then
  `HAP_power` votes, DMA double-buffering measurement, prefill chunk
  sweep, VTCM streaming for W8A16 (`ponytail:` note in `hvx-matmul.c`).
* **`engine="htp"`**: Context/ComputeOps integration per
  `ARCHITECTURE.md` — `HexagonBackend` reads `Qwen3W8cxBin`, packs into
  rpcmem directly, invalidates KV when a `forward()` fails
  mid-generation, and routes DSP errors into the nntrainer logger (host
  messages currently go to stderr, DSP messages to FARF/logcat).
* **v79-native skel**: find why the IEEE/qf32-chain paths misbehave
  (section 7) so `HEX_ARCH=v79` can be the default again.
