# Hexagon Backend

This document describes the Hexagon (cDSP) backend: a host ↔ cDSP
FastRPC layer with persistent buffer mapping, an on-device
round-trip test, and build integration behind the `enable-hexagon`
meson option.

On top of that infrastructure the DSP now carries a real graph
executor: eight HVX op kernels behind a 64-byte op-descriptor wire
format (ABI v2), driven by a QuRT worker pool and verified against
scalar golden references on the hexagon simulator. When `init()`
receives an op-list with `n_ops == 0`, the executor keeps the
original deterministic dummy `forward()` pattern used by the
round-trip test. Nothing in the nntrainer engine calls the backend
yet — the host-side lowering and engine hookup are planned work; see
section 6.

---

## 1. Design goal

Offloading individual tensor ops to the DSP pays a FastRPC
round-trip per op. For autoregressive decode (M=1) the ops are tiny
and numerous, so per-call overhead dominates and offload can end up
slower than staying on the CPU.

This backend therefore offloads at **graph granularity**: the DSP
receives an op-list once, keeps every large buffer persistently
mapped, and executes the whole list per call — so decode costs **one
RPC per token**. The infrastructure validates the expensive parts of
that contract — one-time buffer handoff, persistent DSP-side
mapping, and the RPC round-trip itself — on device, and measures the
round-trip at ~0.26 ms median (see section 5). At one RPC per token
this overhead is negligible.

---

## 2. Architecture

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
    ops["8 HVX op kernels (ops/, hvx/)"]
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

* **`nntr_htp.idl`** defines the wire interface. QAIC generates a
  host stub and a DSP skel from it; both sides are regenerated at
  build time and never committed.
* **`nntr_htp_common.h`** is the op-list wire format (header + op
  descriptors, see section 2.4), plain C compiled by both
  toolchains. Both sides pin `NNTR_HTP_ABI_VERSION`; `init()`
  performs a version handshake and the DSP rejects a mismatch with
  `AEE_EUNSUPPORTED` before touching anything else.
* **`RpcmemBuffer`** (host) is a move-only RAII wrapper around one
  rpcmem (dma-buf) allocation — memory the CPU and DSP can share
  zero-copy. `valid()` is the single health check.
* **`HexagonRunner`** (host) owns one `remote_handle64` session:
  `create()` → `init()` → `forward()`* → destructor closes.
  `create()` returning `nullptr` means "no usable DSP" and callers
  must take the CPU fallback path.
* **`executor.c`** (DSP) is the FastRPC glue: it validates the
  op-list, persistently maps the handed-off buffers, and when the
  op-list carries ops (`n_ops > 0`) builds an `htp_graph` that
  `forward()` delegates to. With `n_ops == 0` it instead keeps the
  M1 dummy pattern — a deterministic fill that mixes in `weights[0]`
  so the round-trip test can prove host-written data is visible
  through the mapping.
* **`htp_graph`** (DSP) owns everything the kernels share: it
  validates the op-list once at `init()`, sizes the quantization and
  attention scratch from an op scan, acquires VTCM best-effort (DDR
  fallback), and per `forward()` call dispatches the ops
  sequentially through the kind table. Each op kernel fans out over
  the **QuRT worker pool** (one worker per HVX unit, detected at
  runtime) and returns at a barrier, so the executor itself stays a
  plain loop.

### 2.1 Buffer strategy: hand off once, map forever

Large buffers (weights / KV cache / activation scratch) cross the
boundary **once**, at `init()`, as dma-buf fds. The DSP maps them
with `HAP_mmap` and keeps the mapping for the session lifetime.
`forward()` carries only `token_ids` in and `logits` out — the
FastRPC driver manages coherency for those small sequence arguments
automatically.

Two non-obvious mechanics, both learned the hard way during
bring-up:

1. **A raw fd number is meaningless to the DSP.** The fd is a host
   process handle; passing it as an `int32` gives the DSP nothing to
   map, and `HAP_mmap` fails with `AEE_ENOMEMORY`. The host must
   first register the fd for the domain with
   `fastrpc_mmap(domain, fd, addr, 0, size, FASTRPC_MAP_FD_DELAYED)`.
   `FASTRPC_MAP_FD_DELAYED` defers the actual mapping to the
   DSP-side `HAP_mmap` call, which is exactly our pattern.
   Re-registering the same fd
   returns `AEE_EALREADY`, which `HexagonRunner::init()` treats as
   success so re-init works.
2. **The weights buffer is additionally passed as an in-sequence**
   in the same `init()` call. The driver performs the one-time CPU
   cache flush for in-parameters; the DSP ignores the sequence and
   keeps only the fd mapping. Weight content must therefore be final
   before `init()` is called.

### 2.2 Session setup: unsigned PD

`create()` requests an unsigned protection domain via
`remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, ...)` before
opening the session. HVX needs no privileges, and unsigned PD avoids
per-device testsig installation and production signing. The call is
best-effort — some firmwares reject it — and the subsequent
`remote_handle64_open` is the real success/failure gate.

### 2.3 Error convention

DSP methods return the AEE codes defined in `AEEStdErr.h`; the
driver passes them through to the host unchanged. Note that **rout
parameters are not copied back on failure** — e.g. `dsp_abi_version`
reads 0 when `init()` fails, which is expected, not a marshalling
bug.

### 2.4 Op-list wire format v2

The op-list buffer passed to `init()` is a fixed-layout byte stream
(`NNTR_HTP_ABI_VERSION` 2): one 64-byte `nntr_htp_oplist_header`
(magic, version, `n_ops`, and the model shape — layers/heads/dims/
`max_seq`/`max_chunk`), followed by `n_ops` × 64-byte
`nntr_htp_op_desc` records. Each descriptor names an op kind, its
m/k/n shape (`m == 0` means "substitute the per-call token count"),
and up to four tensor references — a buffer id (weights / KV /
activations / tokens / logits) plus a 128-byte-aligned byte offset.

Eight op kinds cover a qwen3-style decoder layer:

| kind | computes |
|------|----------|
| `EMBED` | int8 embedding row gather + dequant → fp16 |
| `RMSNORM` | RMS norm, optional per-head QK-Norm (`FLAG_PER_HEAD`) |
| `MATMUL_W8A8` | per-token dynamic-quant int8×int8 matmul → fp16 |
| `ROPE` | rotary embedding on q/k in place, precomputed cos/sin |
| `ATTN` | causal GQA attention against the persistent KV cache |
| `SILU_MUL` | SiLU(gate) ⊙ up |
| `ADD` | elementwise residual add |
| `MATMUL_LOGITS` | last-token int8 matmul → fp32 logits |

`init()` validates everything up front — header sanity, op kinds,
alignment, and every tensor reference bounds-checked against the
actual buffer sizes — and rejects a bad list with `AEE_EBADPARM`
before anything executes; `forward()` then only checks its runtime
arguments (token count vs `max_chunk`, position vs `max_seq`, logits
length) and runs the ops sequentially. Each op fans out internally
over a QuRT worker pool (one worker per HVX unit) and returns at a
barrier. An op-list with `n_ops == 0` is valid and selects the M1
dummy `forward()` path instead of building a graph.

---

## 3. Source layout

```
nntrainer/tensor/hexagon/
├── htp/                      # compiled by hexagon-clang (DSP side)
│   ├── nntr_htp.idl          # FastRPC interface
│   ├── nntr_htp_common.h     # op-list wire format v2 + validation,
│   │                         #   shared with host
│   ├── executor.c            # FastRPC glue → htp_graph (or dummy path)
│   ├── htp_graph.{h,c}       # op-list graph executor: pool, scratch,
│   │                         #   VTCM, sequential op dispatch
│   ├── worker_pool.{h,c}     # QuRT worker pool + barrier
│   ├── ops/                  # one HVX kernel per op kind
│   │   ├── htp_ops.h         # exec context + op fn signatures
│   │   ├── hvx-matmul.c      # MATMUL_W8A8 + MATMUL_LOGITS (VTCM/DMA)
│   │   ├── hvx-rmsnorm.c / hvx-rope.c / hvx-attn.c
│   │   ├── hvx-eltwise.c     # ADD + SILU_MUL
│   │   └── hvx-embed.c
│   ├── hvx/                  # HVX vector helpers (f16 math, quant,
│   │                         #   exp/inverse borrowed from ggml-hexagon)
│   ├── hex/                  # scalar utils (borrowed from ggml-hexagon)
│   └── dma/                  # user-DMA queue (borrowed from ggml-hexagon)
├── host/                     # compiled by NDK clang, part of libnntrainer
│   ├── rpcmem_allocator.{h,cpp}
│   └── hexagon_runner.{h,cpp}
└── meson.build               # folds host sources + stub into the lib

test/hexagon/
├── test_oplist_header.c      # x86 self-check for the wire format
├── hexagon_rpc_test.cpp      # on-device round-trip test (standalone)
└── sim/                      # simulator golden tests (see section 4.5)
    ├── sim_test_main.c       # dispatches tests by name
    ├── ref_ops.{h,c}         # scalar fp32 reference kernels
    └── test_*.c              # one file per test

tools/hexagon/
├── build_skel.sh             # qaic + hexagon-clang → libnntr_htp_skel.so
├── build_sim_test.sh         # cross-build of the sim test lib (v75)
├── run_sim_test.sh           # hexagon-sim + QuRT image, one test by name
├── build_host_test.sh        # NDK cross-build of the on-device test
├── run_device_test.sh        # adb push + run + log capture
├── check_rpc_log.py          # PASS/FAIL verdict + latency stats
└── plot_rpc_latency.py       # latency histogram (matplotlib)
```

The `hvx/hvx-exp.h`, `hvx-inverse.h`, `hvx-base.h`, `hvx-floor.h`, `hvx-types.h`,
the `hex/` utilities and the `dma/` queue are imported from the
llama.cpp ggml-hexagon backend (MIT license); the files keep their
original copyright headers and the borrowed set is listed in the
repository `NOTICE` file.

---

## 4. Building

Prerequisites: Hexagon SDK 6.3.0.0 or later and an Android NDK. The
SDK is
proprietary and follows the same bring-your-own-SDK pattern as the
QNN backend (see `QNN_BUILD.md`). The SDK's hexagon-clang (toolchain
8.8) requires `-mhvx-ieee-fp` for the IEEE fp16 HVX intrinsics the
kernels use; both `build_skel.sh` and `build_sim_test.sh` pass it.

### 4.1 DSP skel

```bash
source $HEXAGON_SDK_ROOT/setup_sdk_env.source
./tools/hexagon/build_skel.sh          # HEX_ARCH=v75 to override (default v79)
# → build_hexagon/skel/libnntr_htp_skel.so
```

The skel is cross-built by hexagon-clang and is **not** part of the
meson build; it is pushed to the device separately (or shipped with
the app and found via `ADSP_LIBRARY_PATH`). Besides the FastRPC glue
it now compiles the whole graph executor — `htp_graph.c`,
`worker_pool.c` and the `ops/`, `hvx/`, `dma/` sources.

### 4.2 Host library (`enable-hexagon`)

```bash
./tools/package_android.sh . \
  -Denable-hexagon=true \
  -Dhexagon-sdk-root=$HEXAGON_SDK_ROOT
```

What the option does:

* errors out unless `platform=android` and an SDK root is given;
* runs QAIC at **configure time** into
  `<builddir>/nntr_htp_generated/` so the stub path stays a plain
  string for the `Android.mk` source list (the Android build is
  meson-configure → ndk-build, so ninja targets can't be used here);
* registers the `.idl` as a reconfigure trigger (`configure_file`
  copy), so editing the IDL reruns QAIC instead of silently building
  against a stale stub;
* adds `host/*.cpp` and the generated stub to `nntrainer_sources`
  and ships `libcdsprpc.so` as a prebuilt. The SDK copy is a link
  stub only — at runtime the loader resolves the device's own
  `/vendor/lib64/libcdsprpc.so` under the same soname.

The option defaults to `false` and is a strict no-op when off.

### 4.3 Standalone on-device test

```bash
./tools/hexagon/build_skel.sh
./tools/hexagon/build_host_test.sh     # needs ANDROID_NDK; links -static-libstdc++
./tools/hexagon/run_device_test.sh     # [adb-serial]
python3 tools/hexagon/check_rpc_log.py logs/hexagon/device_test_<stamp>.log
```

The test binary links only the host sources + stub (no libnntrainer)
and is linked with `-static-libstdc++` because `libc++_shared.so`
does not exist under `/data/local/tmp`.

`run_device_test.sh` also writes `0x1f` into
`hexagon_rpc_test.farf` on the device before running — without that
file the DSP's FARF log lines never reach logcat, and DSP-side
failures are invisible. FARF output is captured into
`logs/hexagon/device_farf_<stamp>.log`.

### 4.4 What the test verifies

1. session open (skel loads, unsigned PD path works);
2. rpcmem allocation (3 × 4 KB dma-bufs);
3. ABI mismatch rejection (version 999 must fail `init`);
4. successful `init` (fd registration + `HAP_mmap` + handshake);
5. forward pattern match — logits must equal
   `token_ids[i%3] + pos + i + weights[0]`, where `weights[0]` was
   written by the **host**, proving the persistent mapping is real;
6. forward latency, 32 timed iterations (= per-token RPC cost).

Every output line starts with `RPC_TEST`; `check_rpc_log.py` turns a
captured log into a `VERDICT: PASS/FAIL` exit code.

The test drives `init()` with an empty op-list (`n_ops == 0`), so it
exercises the M1 dummy `forward()` path by design — it verifies the
RPC/mapping contract, not the kernels. The kernels are covered by
the simulator tests below.

### 4.5 Simulator golden tests

The HVX kernels and the graph executor are verified on the hexagon
simulator against scalar fp32 reference implementations
(`test/hexagon/sim/ref_ops.c`), with no device needed:

```bash
./tools/hexagon/build_sim_test.sh      # → build_hexagon/sim/libnntr_sim_test.so
./tools/hexagon/run_sim_test.sh <name>
```

All tests link into one `libnntr_sim_test.so`; `run_sim_test.sh`
boots a QuRT image on `hexagon-sim` and dispatches one test by name.
A passing test prints `SIM_TEST <name> PASS`. The 13 tests:

`smoke` `pool` `exp` `quant` `matmul` `matmul_dma` `rmsnorm` `rope`
`eltwise` `embed` `attn` `logits` `graph`

(`graph` runs a full multi-layer decode through `htp_graph_forward`
and compares final logits.)

The sim build targets **v75** while the device skel targets v79: the
v75 target was chosen when SDK 6.0.0.2 shipped QuRT sim images only
up to v75, and SDK 6.3.0.0 still provides and maintains the v75
image, so the target is kept. Moving the sim to v79 is a separate
decision. The kernels contain no v75-specific assumptions, and the
same sources build for both targets.

---

## 5. Measured results

### 5.1 Device RPC baseline

On a Galaxy S25 Ultra (8 Elite, SM-S938N), 32 dummy `forward()`
round-trips (SDK 6.3.0.0 skel, 2026-08-19):

| min | median | max |
|-----|--------|-----|
| 231 µs | 255 µs | 284 µs |

This is the floor cost of one graph-level call per token.

### 5.2 Simulator golden test results

All 13 simulator tests pass (SDK 6.3.0.0, toolchain 8.8.06, v75
sim). Accuracy is the worst `cmp_f` deviation against the scalar
fp32 reference; the pass criterion is the mixed bound
`|d| <= atol + rtol * |ref|` (so a large `max_rel` on near-zero
elements is expected and covered by `atol`). Runtime is the
simulator-estimated core cycle count converted at a nominal 1 GHz
DSP clock (1M cycles = 1 ms) for the whole run, which includes
~4.2 ms of QuRT boot (the `smoke` baseline); the DMA-heavy runs are
dominated by simulated DMA cost. Neither is a device-performance
indicator — device measurements are section 5.1 and later
milestones.

| test | max_abs | tolerance (rtol/atol) | est. runtime @ 1 GHz |
|------|---------|-----------------------|----------------------|
| smoke / pool / quant | exact | – | 4.2 / 4.3 / 4.9 ms |
| exp | 1.5e-05 | – | 4.4 ms |
| matmul | 0 (bit-exact) | 2e-3 / 1e-3 | 33.7 ms |
| matmul_dma | 0 (bit-exact) | 2e-3 / 1e-3 | 211.7 ms |
| rmsnorm | 3.9e-03 | 5e-3 / 2e-3 | 6.0 ms |
| rope | 2.0e-03 | 5e-3 / 2e-3 | 51.4 ms |
| add | 0 (bit-exact) | 1e-3 / 1e-4 | 12.1 ms |
| silu_mul | 3.1e-02 | 2e-2 / 5e-3 | (with add) |
| embed | 0 (bit-exact) | 1e-3 / 1e-4 | 5.8 ms |
| attn | 9.8e-04 | 2e-2 / 5e-3 | 10.3 ms |
| logits | 0 (bit-exact) | 5e-3 / 1e-2 | 43.0 ms |
| graph (prefill/decode) | 3.0e-02 / 3.2e-02 | 3e-2 / 5e-2 | 214.8 ms |

The integer paths (matmul, embed, logits, add) are bit-exact by
construction — the references share the same quantization
arithmetic. The `graph` deviation is the accumulated re-rounding
noise of 13 per-token quantization stages across the synthetic
2-layer model, not a kernel error; per-op wiring is covered by the
bit-exact and per-kernel rows above.

---

## 6. Current status and planned work

Implemented:

* M1 — the FastRPC skeleton described above: IDL, persistent fd
  mapping, on-device round-trip test, and the `enable-hexagon` build
  wiring;
* M2 — the DSP-side graph executor: the op-list wire format v2
  (section 2.4), eight HVX op kernels, the QuRT worker pool, the
  `htp_graph` executor wired into the FastRPC session, and the 13
  simulator golden tests (section 4.5). The `n_ops == 0` dummy
  `forward()` path is kept so the M1 round-trip test still passes
  unchanged.

Planned, in rough order:

* M3 — host-side lowering: building the op-list, quantized weight
  layouts, and the KV/activation buffer plan from an actual model on
  the host;
* engine hookup — a Context/ComputeOps integration per
  `ARCHITECTURE.md`, KV-cache invalidation when a `forward()` fails
  mid-generation, and routing DSP-side errors into the nntrainer
  logger (host messages currently go to stderr, DSP messages to
  FARF/logcat);
* power/latency tuning (`HAP_power` votes).
