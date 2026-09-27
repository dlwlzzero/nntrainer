# HTP backend code map (from origin/htp_moe, read 2026-09-27)

Gathered 2026-09-27 at `a7c66dec` by reading `origin/htp_moe` with `git show` / `git grep`; recheck anything you cite against the current ref (`git show origin/htp_moe:<path>`).

B = `nntrainer/tensor/htp_backend`

## ARM side

The call chain from the model down to the stub:

1. CausalLM
2. `Lfm2MoELayer`
3. `Tensor::getOps()`, which returns the `ComputeOps*` held in the layer's context
4. `HtpComputeOps`
5. the generated stub `nntr_hvx_*()`, using the session handle from `HtpBackend::global()`
6. FastRPC

### `HtpContext` (`nntrainer/htp_context.h/.cpp`)

- Inherits from `Context` and `Singleton<HtpContext>`.
- Registered under the name `"htp"`; a layer selects it with the property `engine="htp"`.
- `initialize()` calls `ensureComputeOps()`:
  - If `HtpBackend::global().enabled()`, it calls `setComputeOps(get_htp_ops())`.
  - Otherwise it calls `setComputeOps(get_cpu_ops())`.

### `HtpBackend` (`B/htp_backend.h/.cpp`)

A process-wide singleton, accessed through `global()`. Members: `enabled_`, `handle_` (a `remote_handle64`), `qos_mode_`.

The constructor, in order:

1. Calls `remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, {CDSP_DOMAIN_ID,1})`, which gives an unsigned PD.
2. Calls `nntr_hvx_open(nntr_hvx_URI "&_dom=cdsp")`.
3. Calls `remote_handle64_control(DSPRPC_CONTROL_LATENCY)`:
   - It first tries `RPC_POLL_QOS` with a 5000 us window (`NNTR_HTP_POLL_US` overrides the window).
   - If that fails, it falls back to `RPC_PM_QOS` at 100 us.

The destructor calls `nntr_hvx_close`. `HtpBackend` owns the single FastRPC session.

### `ComputeOps` → `CpuComputeOps` → `HtpComputeOps`

`ComputeOps` is abstract (`nntrainer/tensor/cpu_backend/compute_ops.h`). `HtpComputeOps` is defined in `B/htp_compute_ops.cpp`, and `get_htp_ops()` returns its singleton.

HTP-relevant virtual methods:

- `gemm_q4_0_accel_fp32`
- `gemm_q4_0_batch_fp32`
- `gemm_qs4cx_accel_fp32`
- `gemm_qs4cx_batch_fp32`
- `gemm_qs4cx_fused_swiglu_fp32`
- `gemm_qs4cx_moe_layer_fp32`
- `accelerates_q4_0_at_m1()` (returns false)
- `set_decode_graph_desc` (#119)
- `register_qs4cx_weight`
- `register_q4_0_weight`

Private invoke helpers. Each one wraps one IDL call:

- `invokeLayer`
- `invokeLayerU8In`
- `invokeFused`
- `invokeGateUpSwiglu`
- `invokeMoeLayer`
- `invokeForward` (#119)
- `sendMoeOptsOnce` (runs once through `call_once`, sends `moe_set_opts`)

Weight registration:

- `get_or_register`: Q4_0x4 → `htp_qs4cx_from_q4_0x4` → DSP heap.
- `get_or_register_fc`
- `get_or_register_qs4cx`: → `nntr_hvx_weight_register_u8i4`; the DSP heap copies ("bakes") the weights.
- `get_or_register_wh`: QS4CX_WH → arena → `registerFromArena` → `nntr_hvx_weight_register_u8i4_arena`.

Arena helpers:

- `ensureArena`
- `place`
- `newChunk`
- `releaseArmSource`: calls madvise `DONTNEED` on the ARM copy unless `NNTR_HTP_KEEP_ARM_WEIGHTS` is set.

Nested structs:

- `ArenaChunk{unique_ptr<HtpRpcBuffer> buf, dsp_id, used}`: a bump allocator; each chunk is at most 256 MiB, halved when an allocation is refused.
- `ArenaEntry{chunk, off, K, N, w_scale, colsum_w, bias}`
- `StagingPool{map<size_t, unique_ptr<HtpRpcBuffer>> by_class}`: one buffer per power-of-two size class, starting at 64 KiB.
- `FcHandles`

Members:

- `handle_cache_`: weight pointer → DSP handle.
- `arena_chunks_`
- `invoke_mutex_`: serializes every call.
- `act_pool_`, `out_pool_`: the staging pools.
- #119 graph state: `graph_words_`, `moe_ops_`, `graph_inited_`, and related fields.

Ownership: `HtpComputeOps` owns its ArenaChunks (and their `HtpRpcBuffer`s) and the staging pools. It refers to `HtpBackend` through the singleton.

### `HtpProfile` (anonymous namespace)

A singleton controlled by `NNTR_HTP_PROFILE`, levels 1, 2 and 3, and by `NNTR_HTP_DMA_TRACE`. It records per-call host time and stage slots.

### `HtpRpcMemApi` / `HtpRpcBuffer` (`B/htp_rpcmem.h`)

- The functions are resolved at runtime with dlsym: `rpcmem_init`, `rpcmem_alloc`, `rpcmem_free`, `rpcmem_to_fd`, `fastrpc_mmap`, `fastrpc_munmap`.
- `HtpRpcBuffer(bytes, flags)` allocates on rpcmem heap 25.
  - With the default flags (1) the buffer is cached; this is used for staging.
  - With the `UNCACHED` flag (0) it is uncached; this is used for the arena.
  - If rpcmem is not available, it falls back to malloc.

### Pure helpers

- `nntrainer/tensor/htp_q4_0_convert.h`
- `nntrainer/tensor/htp_wh_layout.h`: WH tiles, 32×32 int4 = 512 B.
- `nntrainer/tensor/htp_act_quant.h`

Header-only C99, shared by the ARM side, the skel and the host checks:

- `B/htp_moe_opts.h`: the `HTP_MOE_FLAG_*` bits and the flags resolver.
- `B/htp_graph_desc.h`: the #119 op table.

### CausalLM integration

`Applications/CausalLM/models/lfm2_moe/lfm2_moe_causallm.cpp` defines `Lfm2MoeCausalLM : Lfm2CausalLM`:

- `setupParameters` reads `moe_engine`, `moe_htp_layers` and `moe_layer_dtype` (QS4CX or QS4CX_WH).
- `createMoeLayer` creates an `lfm2_moe` layer and sets its `engine` per layer.

`lfm2_moe_layer.cpp` defines `Lfm2MoELayer : LayerImpl`:

- The router (FP32 gate plus `expert_bias`) runs on the ARM side through `dot()`.
- `buildExpertAssignments` produces the routing.
- `tryMoeLayerOnAccelerator` flattens the routing into `row_index`, `row_count` and `row_weight`, then calls `ops->gemm_qs4cx_moe_layer_fp32`.

`Applications/CausalLM/models/transformer.cpp`, at load time:

- Calls `register_qs4cx_weight` for every MoE weight.
- Runs one MoE warm-up at M=512.
- Also registers the Q4_0 FC weights.

## FastRPC boundary

### Building the interface

The IDL is `test/htp/nntr_hvx.idl`: `interface nntr_hvx : remote_handle64`.

- The ARM stub is generated by `B/generate_stub.sh` (runs qaic) into `B/generated/`.
- The skel is built by `test/htp/build.sh` (qaic plus hexagon-clang with `-mhvx -mhvx-length=128B`), producing `libnntr_hvx_skel.so`.
- The skel is loaded from `ADSP_LIBRARY_PATH` into the cDSP unsigned PD.

### Method groups

| group | methods |
|---|---|
| session | open, close (from `remote_handle64`) |
| weights | `weight_register_u8i4`, `weight_register_u8i4_arena(K,N,arena,wh_off,w_scale,colsum_w,bias → w_handle)`, `weight_release_u8i4`, u8i8 variants |
| arena | `arena_attach(fd, bytes → arena)`, `arena_detach` |
| matmul | `mm_u8i4_layer[_timed]`, `_u8in`, `_fused`, `gate_up_swiglu`, `mm_u8i8_layer` |
| MoE | `mm_u8i4_moe_layer[_timed](M,K,inter,N_out, seq h_gate_up, seq h_down, seq row_index,row_count,row_weight, seq act_f32 → rout out_f32[, stage_us])`, `moe_set_opts(flags → applied)` |
| attention / small ops | `attn_register`, `attn_kv_append`, `attn_forward`; `softmax`, `exp`, `swiglu` |
| debug / probe | `dma_probe`, `dma_replay`, `moe_dma_trace_read`, `arena_probe` |
| per-token (#119) | `graph_init(desc → n_ops)`, `graph_release`, `forward(start_op,pos,routing,act_in → act_out,resume_at)`, `forward_debug` |

### What crosses on each call

- The activation and the output travel as `in sequence<>` / `rout sequence<>` buffers. They are copied into `HtpRpcBuffer`s taken from the size-class staging pools; these buffers are cached ION.
- The driver maps these buffers on every call and does cache maintenance over the whole dma-buf. Because the cost covers the whole dma-buf, buffers are pooled by size class: before the fix, one largest-shape buffer made decode's 8 KB call pay for a flush of the whole buffer (decode fell from 20.8 to 11.6 tok/s).
- Small arrays go as plain heap sequences: the weight handles, the routing, and a 48-byte primitive block.

### Handles

- The session handle is a `remote_handle64`; on the DSP it is a `nntr_hvx_session*`.
- A weight handle is a uint32 slot index into `hexkl_weight_u8i4_table`, which has 2048 slots.
- An arena id is a uint32 slot into `session.arenas[32]`.

### Arena shared memory (weights)

1. On the ARM side: `rpcmem_alloc`, uncached, in chunks of at most 256 MiB.
2. `rpcmem_to_fd`.
3. `fastrpc_mmap(CDSP_DOMAIN_ID, fd, va, 0, size, FASTRPC_MAP_FD)`.
4. `nntr_hvx_arena_attach(fd, bytes)`. On the DSP this calls `HAP_mmap_get(fd)`, which returns the DSP virtual address.
5. Each weight is registered by `(arena, wh_off)` and borrowed in place (`borrowed=1`), with no copy.
6. Detach and close call `HAP_mmap_put`.

The weights are written once and the arena is uncached on the CPU, so no flush is needed. `dmahandle` and `remote_register_buf` are not used.

### Synchronization

- Every call is synchronous, and `invoke_mutex_` serializes them.
- The ARM side waits for the reply under FastRPC poll-QoS, with a 5 ms poll window.
- In `open()`, the DSP places `HAP_power` votes:
  - apptype COMPUTE
  - DCVS v2 performance mode with 100 us latency and NOM/TURBO corners
  - a 40 GB/s bus vote
- Stale-skel guard: if the IDL and the skel do not match, calls fail with `AEE_EBADPARM` (0x8000040E). The `moe_set_opts` echo must equal the flags that were sent.

## DSP side

### Entry points

- `test/htp/hvx_add_f32.c`: open and close.
- `nntr_hvx_mm_u8i4.c`: weights, arena, layer, MoE, `moe_set_opts`.
- `nntr_hvx_attn.c`
- `nntr_hvx_dma_probe.c`
- `nntr_hvx_graph.c` (#119)

### Session struct (`test/htp/nntr_hvx_session.h`)

`nntr_hvx_session` has these fields: `vtcm_base`, `vtcm_size`, `config_off`, `hmx_locked`, `weights_u8i4`, `weights_u8i8`, `hvx_worker_pool *quant_pool`, `arenas[32]{fd,va,bytes}`, `moe_scratch`, `moe_flags`, `hexkl_graph *graph`.

`open()` does, in order:

1. calloc
2. `qurt_hvx_get_units`
3. `hexkl_micro_hw_init`: takes the whole VTCM, about 8 MiB.
4. `hexkl_micro_hmx_lock`
5. `hmx_setup_acc_read_int32`
6. `hvx_worker_pool_create(n_hvx-1)`
7. the HAP_power votes

`close()` frees the graph, then the weights, the arenas, the scratch and the pool, then unlocks the HMX.

### Weight table (`B/hmx/hexkl_mm_u8i4_dma.h`)

- `hexkl_weight_u8i4{in_use, wh_bytes, w_scale, colsum_w, bias, K, N, borrowed}`
- `hexkl_mm_u8i4_layer_run` is the FC path. It double-buffers the weights into VTCM.

### MoE kernel (`B/hmx/hexkl_mm_u8i4_moe.c/.h`)

Entry: `hexkl_mm_u8i4_moe_layer_run(tbl, vtcm, …, h_gate_up, h_down, routing, act_f32, out_f32, pool, scratch, flags)`.

**HMX path (prefill; M > 4, or the GEMV flag off):**

1. A DMA ring copies the activation in.
2. The first expert's gate_up is pushed before the quant scan, as gate/up chunk lists; each list holds at most 16 chunks.
3. The pool's background lane does the AH pack.
4. For each 64-row block and each chunk: `hexkl_dma_ring_wait`, then `hexkl_micro_hmx_mm_u8i4`, then `hmx_acc_read_int32`.
5. Pool workers run dequant with SwiGLU, then requant, then down, then dequant, then scatter.

**M=1 GEMV path (decode):** taken when the `M1_GEMV` flag is set, M ≤ 4, and at most 16 experts are active.

1. The activation is packed inline.
2. Stage A: `moe_m1_pair_worker` computes the gate/up columns with SwiGLU fused, through `hvx_gemm_u8i4_wh_col` (HVX vrmpy over WH tiles; bit-identical to HMX).
3. Stage B: requant.
4. Stage C: `moe_m1_down_worker`.
5. The caller does the scatter inline.

Each stage is one `hvx_worker_pool_run` across 6 lanes (the n_hvx−1 workers plus the caller), with expert-major slices.

The two ways the GEMV gets its weights:

- **l2fetch lead** (#113, used on the arena-read path): before computing block b, a worker issues block b+1's 2D l2fetch box. The lead defaults to 192 KB, and at most 3 boxes are outstanding. The row loop is `rows1`.
- **VTCM feed** (#117, the default):
  - VTCM holds two gate_up-sized slabs: G0 at offset 0 and G1 at `gu_bytes`.
  - Expert i's gate_up goes into slab i&1, as one whole-matrix DMA descriptor. Once no gate_up remains to load, the down matrices land in the slabs.
  - Waits sit inside the MM bracket. A slab is reused only after the join of the pool run that read it.
  - The l2fetch lead is forced off under the feed.
  - The feed requires `2*gu_bytes ≤ VTCM` and `2*dn_bytes ≤ gu_bytes`.

### DMA ring (`B/hmx/hexkl_dma_ring.c/.h`)

- One dmlinked chain of `hexkl_dma_desc2d` descriptors, 256 slots, used only by the caller thread.
- API: `push2d(dst, src, dst_stride, src_stride, row_size, nrows, …)`, `wait(idx)` (spins with dmpoll; descriptors retire in order), `drain`.
- Low level: `dmlink`, `dmstart`.

### Worker pool (`B/hvx/hvx_worker_pool.c/.h`)

- QuRT threads using the seqn/futex pattern.
- `run(pool, func, ctx, n_units)`: fork/join, with the caller acting as worker 0.
- A background lane: `submit_bg` / `wait_bg`.

### Per-token op table (#119, behind `NNTR_HTP_FORWARD=1`, off by default)

Wire format, defined in `htp_graph_desc.h`:

- A 7-word header: magic "HTPG", version, n_layers, n_ops, hidden, vocab, max_seq.
- `layer_kind[]`, `ffn_kind[]`
- `ops[]`: one `htp_graph_op` per op, 76 words each.
- Op kinds: RMSNORM, FC, CONV1D_GATE, QK_NORM, ROPE, ATTN_M1, ADD, ROUTER_TOPK, MOE, DENSE_FFN, LM_HEAD.
- At most 256 ops (LFM2 uses 228) and 3 activation slots.

On the skel, `hexkl_graph.c` holds the kernel table; only MOE is filled so far. The model still makes 22 calls per token.

On the ARM side:

1. `bindMoeOp` binds the op table's MoE entries.
2. `ensureGraphInit` sends the table once (`graph_init`).
3. From then on, `invokeForward` makes the calls.

## Runtime switches

Environment variables:

- `NNTR_HTP_POLL_US`: default 5000.
- `NNTR_HTP_PROFILE`: 1, 2 or 3.
- `NNTR_HTP_DMA_TRACE`
- `NNTR_MOE_HTP_M1_GEMV`: on by default.
- `NNTR_MOE_HTP_GEMV_LEAD_KB`: default 192.
- `NNTR_MOE_HTP_GEMV_ROWS1`: default 1.
- `NNTR_MOE_HTP_GEMV_FEED`: default 1 (vtcm); 0 means arena. With no switches set, the startup banner reads `applied=0x303e1`.
- `NNTR_HTP_FORWARD`
- `NNTR_MOE_HTP_DECODE`
- `NNTR_HTP_KEEP_ARM_WEIGHTS`

Config keys in `nntr_config.json`:

- `moe_engine`, `moe_htp_layers`, `moe_layer_dtype`
- `attn_proj_engine`, `conv_in_proj_engine`, `dense_ffn_engine`, each with its `*_htp_layers` list

## Optimizations landed since the upstream freeze 2ce38d65

| PR (issue) | change | component | measured |
|---|---|---|---|
| #93 (#87) | DMA trace, dma_probe/replay | DSP ring, skel | measurement only |
| #86 (#80) | M=1 HVX GEMV dispatch behind flag, `sendMoeOptsOnce` | moe kernel, hvx_gemm_u8i4_wh, IDL | — |
| #103 (#88) transport fix | ION staging per power-of-two size class (was one largest-shape buffer → decode 8 KB call paid a whole-dma-buf cache flush, decode 20.8 → 11.6) ; poll QoS 5 ms default (20.6 → 23.7 tok/s) | HtpComputeOps StagingPool; HtpBackend | see BENCHMARK |
| #108 (#101) | M=1 GEMV default (+6.2 / +7.7 / +1.7 % decode) | ARM opts | |
| #115 (#105+#113) | one-row loop, l2fetch box one block ahead, ≤3 outstanding, defaults rows1=1 lead=192 KB (mm 937 us, decode +3–4 %) | hvx_gemm_u8i4_wh, moe kernel | |
| #118 (#117) | VTCM feed: whole expert matrices DMA'd into two VTCM slabs under the GEMV; default. M=1 mm 922 → 676 us (−27 %), decode +31 % | moe kernel, ARM opts | |
| #119 (#85) | per-token entry skeleton (op table, graph_init/forward), behind NNTR_HTP_FORWARD | htp_graph_desc.h, nntr_hvx_graph.c, hexkl_graph.c, HtpComputeOps, Lfm2MoeCausalLM | not measured |
