# HTP Tensor Formation Analysis for `HTP_OPS_MAT_MUL_PERMUTED_W4D16A32`

This document analyzes how activation and weight tensors are formed and processed
before executing the `HTP_OPS_MAT_MUL_PERMUTED_W4D16A32` kernel on the Hexagon
Tensor Processor (HTP), based on the
[haozixu/llama.cpp-npu](https://github.com/haozixu/llama.cpp-npu) reference
implementation and the nntrainer HTP backend.

## Overview

`HTP_OPS_MAT_MUL_PERMUTED_W4D16A32` performs quantized matrix multiplication:

```
Output(F32) = Activation(F32) x Weight(Q4_0, pre-permuted)
```

- **Activation**: float32, shape `[M x K]`
- **Weight**: Q4_0 quantized (4-bit with FP16 scales), pre-permuted into
  `my_block_q4_0` super-block layout, logical shape `[N x K]`
- **Output**: float32, shape `[M x N]`
- **Shape constraints**: `K % 32 == 0`, `N % 32 == 0`

The computation has two stages: host-side dispatch (ARM CPU) and DSP-side kernel
execution (Hexagon HTP).

---

## 1. Host-Side Dispatch

**Source**: `ggml/src/ggml-htp/htp-ops.cc`

### Trigger Conditions

The operation is selected in `htp_ops_support_op()` when:
- `dst->type == GGML_TYPE_F32`
- `weight->type == GGML_TYPE_Q4_0` (repacked)
- `activation->type == GGML_TYPE_F32`
- `K % 32 == 0 && N % 32 == 0`
- Both src and dst tensors are contiguous (no extra dimensions)

### Tensor Identification

```c
auto * weight     = dst->src[0];  // Q4_0, shape [K x N] (ggml convention)
auto * activation = dst->src[1];  // F32, shape [M x K]
```

### RPCMEM Buffer Mapping

Before dispatching to DSP, all tensor buffers must be mapped into DSP-accessible
shared memory via FastRPC:

1. `prepare_tensor_rpcmem_mapping(dst)` calls `RpcMemMapper::validate()`
2. For each RPCMEM buffer, `rpcmem_to_fd()` obtains a file descriptor
3. `fastrpc_mmap()` maps the buffer into DSP virtual address space
4. An LRU eviction policy manages the 3GB mapping limit

### Parameter Serialization

```c
MatMulParams params {
    .output     = { output_fd,     (int32_t) output_offset     },
    .activation = { activation_fd, (int32_t) activation_offset },
    .weight     = { weight_fd,     (int32_t) weight_offset     },
    .m          = m,   // number of activation rows
    .k          = k,   // reduction dimension (weight->ne[0])
    .n          = n,   // number of output columns (weight->ne[1])
};
```

The params are packed into an `OpComputeRequest` message with
`op = HTP_OPS_MAT_MUL_PERMUTED_W4D16A32`, written to the shared message channel,
and the DSP is notified via atomic flag.

---

## 2. DSP-Side Kernel: Activation Processing

**Source**: `mat_mul.c :: transfer_activation_chunk_fp32_to_fp16()`

### Input
- Float32 array in DDR, row-major `[M x K]`

### Processing Pipeline

```
DDR (F32, row-major) --> VTCM (FP16, HMX tile layout)
```

1. **Chunking**: The M dimension is split into chunks of `m_chunk_n_rows`
   (determined by `find_chunk_size()` to fit in 1MB activation VTCM area)

2. **F32 to FP16 conversion + tiling**: Two rows are processed simultaneously:
   ```c
   HVX_Vector v0 = *pv_in0++;  // 32 floats from row r
   HVX_Vector v1 = *pv_in1++;  // 32 floats from row r+1
   HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);  // interleaved FP16
   ```

3. **HMX tile layout**: Output is stored in tiles of
   `HMX_FP16_TILE_N_ROWS x HMX_FP16_TILE_N_COLS` (typically 2 x 32).
   Each tile stores `HMX_FP16_TILE_N_ELMS` FP16 elements.
   ```c
   int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;
   HVX_Vector *tile = (HVX_Vector *)(vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
   tile[r1 / 2] = v_out;
   ```

---

## 3. DSP-Side Kernel: Weight Processing (Q4_0)

**Source**: `mat_mul.c :: hmx_mat_mul_af32_pwqk0_of32()`

### Input Format: Pre-permuted Q4_0

The weight is stored as `my_block_q4_0` super-blocks:

```c
// Single quantization group (32 elements)
typedef struct {
    __fp16  scale;           // FP16 scale factor
    uint8_t quants[16];      // 32 x 4-bit values packed into 16 bytes
} block_q4_0;                // 18 bytes

// Super-block (8 groups = 256 elements = QK_K)
typedef struct {
    __fp16  scales[8];       // 8 FP16 scales (coalesced from 8 groups)
    uint8_t quants[128];     // 8 x 16 bytes of packed 4-bit quants
} my_block_q4_0;             // 144 bytes
```

The "permuted" layout reorganizes data for efficient HVX/HMX processing:
- Scales are coalesced at the beginning of each super-block
- Quants from 8 groups are contiguous
- Weight matrix arranged as `[N x K]` with N super-blocks per row

### Processing Pipeline (3-Stage with Double Buffering)

```
DDR (Q4_0 super-blocks)
  |
  | DMA transfer (async, double-buffered)
  v
VTCM scratch buffer (raw Q4_0 bytes)
  |
  | HVX dequantization (multi-threaded)
  v
VTCM weight area (FP16, HMX tile layout)
```

#### Stage A: DMA Transfer
```c
dma_issue_load_from_ddr(&desc, buf_curr, permuted_weight, chunk_size);
// While previous chunk is being dequantized, next chunk DMA starts
```

#### Stage B: Dequantization (`dequantize_permuted_weight_q4_0_to_fp16_hvx_task`)

For each `my_block_q4_0` super-block:

1. **Load packed quants**: `HVX_Vector qs = vmemu(src[i].quants)` (128 bytes)
2. **Split 4-bit values**:
   ```c
   HVX_Vector v_qs_lo = qs;                          // low nibbles
   HVX_Vector v_qs_hi = Q6_Vub_vlsr_VubR(qs, 4);    // high nibbles
   ```
3. **LUT conversion to FP16**:
   ```c
   // q4_0_to_fp16_lut: {-8, -7, ..., -1, 0, 1, ..., 7} as FP16
   HVX_VectorPair vp_q0 = Q6_Wh_vlut16_VbVhR_nomatch(v_qs_lo, vlut_cvt, 0);
   HVX_VectorPair vp_q1 = Q6_Wh_vlut16_VbVhR_nomatch(v_qs_hi, vlut_cvt, 0);
   ```
4. **Load and broadcast scales**:
   ```c
   HVX_Vector v_packed_scales = vmemu(src[i].scales);
   HVX_Vector vlut_scales = Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(v_packed_scales));
   ```
5. **Dequantize**: `dequantized = quant_fp16 * scale_fp16`
   ```c
   *pv_out++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(Q6_V_lo_W(vp_q0), vs0_c));
   // ... repeat for all 4 output vectors (256 FP16 values)
   ```

Result: FP16 values in HMX tile layout stored in VTCM weight area.

---

## 4. HMX Matrix Multiply

**Source**: `mat_mul.c :: core_dot_chunk_fp16()`

```c
hmx_unit_acquire();
asm volatile("mxclracc.hf");          // clear HMX accumulator
hmx_set_output_scales(scales);         // set to 1.0 (no additional scaling)

for (r = 0; r < n_row_tiles; ++r) {
    for (c = 0; c < n_col_tiles; ++c) {
        for (k = 0; k < n_dot_tiles; k += 32) {
            hmx_load_tiles_fp16(row_tiles + offset, col_tiles + offset, n_tiles);
        }
        hmx_consume_accumulator_fp16(out_tile);  // write result tile
    }
}
hmx_unit_release();
```

---

## 5. Output Conversion

**Source**: `mat_mul.c :: transfer_output_chunk_fp16_to_fp32()`

```
VTCM (FP16, HMX tile layout) --> DDR (F32, row-major)
```

Two rows at a time are converted back:
```c
HVX_Vector v_src = ((const HVX_Vector *)tile)[r1 / 2];
HVX_VectorPair vp = hvx_my_vhf_to_wsf(v_src);  // FP16 -> 2 x F32

*pv_out0 = Q6_V_lo_W(vp);   // row r
*pv_out1 = Q6_V_hi_W(vp);   // row r+1
```

---

## 6. VTCM Memory Layout

| Area       | Size  | Purpose                              |
|------------|-------|--------------------------------------|
| Weight     | 1 MB  | Dequantized FP16 weight tiles        |
| Activation | 1 MB  | Converted FP16 activation tiles      |
| Output     | 1 MB  | FP16 output tiles (before F32 conv)  |
| Scratch 0  | 1 MB  | DMA double buffer A                  |
| Scratch 1  | 1 MB  | DMA double buffer B                  |
| Scratch 2  | 1 MB  | Pipeline output double buffer        |
| Scales     | 256 B | HMX column scales (set to 1.0)       |

`find_chunk_size()` determines the optimal `m_chunk_n_rows` and `n_chunk_n_cols`
to maximize utilization within VTCM capacity.

---

## 7. End-to-End Data Flow Summary

```
Host (ARM CPU)                         DSP (Hexagon HTP)
--------------                         ------------------
1. Identify MUL_MAT op
   weight=Q4_0, act=F32
2. Map buffers via FastRPC
3. Pack MatMulParams
4. Send op request via              -->  5. Receive request
   shared message channel                6. Map fd -> pointers
                                         7. For each (m_chunk, n_chunk):
                                            a. DMA: activation F32 -> VTCM
                                            b. HVX: F32 -> FP16 tiled
                                            c. DMA: weight Q4_0 -> VTCM scratch
                                            d. HVX: dequant Q4_0 -> FP16 tiled
                                            e. HMX: FP16 tile matmul
                                            f. HVX: FP16 tiled -> F32 DDR
                                         8. Signal completion
9. Read completion flag          <--
10. Continue to next op
```
