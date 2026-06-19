# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file q4_0_x4x2_quant.py
## @brief Q4_0 -> DSP-native x4x2 quantizer (HTP target), pure numpy.
##
## Offline counterpart to the FP16 HMX converter's permute helper. Produces the
## exact x4x2 byte layout consumed by the HTP matmul kernel
## `htp_ops_mat_mul_af32_pwqk0_of32`, so the runtime
## (nntrainer/tensor/float_tensor.cpp FloatTensor::dotQnK, Tdatatype::Q4_0_X4X2
## case) can memcpy the weight straight into RPC shared memory with no further
## conversion -- mirroring the FP16 dotFloat32Float16 path.
##
## Byte-for-byte source of truth (ported here, pinned by _self_test):
##   * nntrainer/tensor/q4_0_utils.cpp::quantizeAndRepackQ4_0X4x2
##       = quantize_q4_0(fp32 -> block_q4_0 stream) + repackToX4x2_Q4_0
##   * nntrainer/tensor/q4_0_utils.cpp::repackToX4x2_Q4_0 / unpackX4x2_to_Q4_0
##
## x4x2 row layout (one row == one output neuron, N rows total):
##   [ packed_quants : K/2 bytes ] [ scale_blocks : (K/256)*16 bytes ]
##   row_stride = K/2 + (K/256)*16   (== the plain Q4_0 byte count)
## Within repackToX4x2_Q4_0 each 32-elem group g of a 256-elem super-block is
## copied to quant offset (g//4)*64 + (g%4)*16, which equals g*16 -- i.e. the
## quants region is just every group's 16 qs bytes in group order, and the
## scales region is every group's FP16 scale in group order. So x4x2 is a
## structure-of-arrays split of the (d, qs) block stream: all qs, then all d.

import numpy as np

QK4_0 = 32  # elements per Q4_0 block (nntrainer/tensor/q4_0_tensor.h)


def quantize_q4_0(x: np.ndarray) -> bytes:
    """Quantise fp32 row-major (..., K) -> raw Q4_0 block stream (18 bytes/block:
    2-byte FP16 scale d, then 16 bytes of packed 4-bit quants qs).

    Verbatim port of Applications/.../gguf_to_nntrainer.py::quantize_q4_0, which
    is itself a port of ggml's quantize_row_q4_0_ref. Kept local so this module
    has no cross-file dependency.
    """
    flat = x.reshape(-1, QK4_0).astype(np.float32)
    nb = flat.shape[0]
    amax_idx = np.argmax(np.abs(flat), axis=1)
    amax = flat[np.arange(nb), amax_idx]
    d = amax / -8.0
    id_ = np.where(d != 0.0, 1.0 / d, 0.0).astype(np.float32)
    # Quantise to [-8, 7], then store +8 to fit in 4 bits unsigned.
    q = np.clip(np.rint(flat * id_[:, None]) + 8.0, 0, 15).astype(np.uint8)
    low = q[:, :16]
    high = q[:, 16:]
    qs = (low | (high << 4)).astype(np.uint8)  # (nb, 16)
    d_fp16 = d.astype(np.float16).view(np.uint16).reshape(nb, 1)
    d_bytes = d_fp16.view(np.uint8).reshape(nb, 2)
    blocks = np.concatenate([d_bytes, qs], axis=1).astype(np.uint8)  # (nb, 18)
    return blocks.tobytes()


def x4x2_row_stride(K: int) -> int:
    """Bytes per output row in the x4x2 layout."""
    return K // 2 + (K // 256) * 16


def quantize_and_repack_q4_0_x4x2(w_fp32_NK: np.ndarray) -> bytes:
    """FP32 weight [N (out), K (in)] -> Q4_0 x4x2 row-strided bytes.

    Returns N * x4x2_row_stride(K) bytes. Requires K % 256 == 0 and N % 32 == 0
    (the pwqk0 kernel's VLEN/super-block constraints); raises ValueError
    otherwise so callers can fall back to a non-HTP layout.
    """
    if w_fp32_NK.ndim != 2:
        raise ValueError(f"expected 2D [N, K] weight, got {w_fp32_NK.shape}")
    N, K = w_fp32_NK.shape
    if K % 256 != 0 or N % 32 != 0:
        raise ValueError(
            f"N={N}, K={K} must satisfy N%32==0 and K%256==0 for x4x2"
        )

    blocks = quantize_q4_0(np.ascontiguousarray(w_fp32_NK, dtype=np.float32))
    n_groups = K // QK4_0
    arr = np.frombuffer(blocks, dtype=np.uint8).reshape(N, n_groups, 18)
    d_bytes = arr[:, :, 0:2]   # (N, n_groups, 2) FP16 scale, little-endian
    qs = arr[:, :, 2:18]       # (N, n_groups, 16) packed quants

    # x4x2 = per row: all qs in group order, then all scales in group order.
    quants = qs.reshape(N, n_groups * 16)        # K/2 bytes
    scales = d_bytes.reshape(N, n_groups * 2)    # (K/256)*16 bytes
    row = np.concatenate([quants, scales], axis=1)
    assert row.shape[1] == x4x2_row_stride(K)
    return np.ascontiguousarray(row, dtype=np.uint8).tobytes()


def unpack_x4x2_to_blocks(src: bytes, N: int, K: int) -> bytes:
    """Inverse of the repack step: x4x2 row-strided bytes -> Q4_0 block stream
    (18 bytes/block, group order). Port of
    Q4_0Utils::unpackX4x2_to_Q4_0; used only by _self_test for a round-trip.
    """
    n_groups = K // QK4_0
    stride = x4x2_row_stride(K)
    raw = np.frombuffer(src, dtype=np.uint8).reshape(N, stride)
    quants = raw[:, : K // 2].reshape(N, n_groups, 16)
    scales = raw[:, K // 2 :].reshape(N, n_groups, 2)
    blocks = np.concatenate([scales, quants], axis=2)  # (N, n_groups, 18)
    return np.ascontiguousarray(blocks, dtype=np.uint8).tobytes()


# --- Cross-language golden -------------------------------------------------
# Generated once by a C++ harness calling nntrainer::quantizeAndRepackQ4_0X4x2
# on the deterministic fixture below (see _fixture / _self_test). Pins this
# module's output to the C++ source of truth at every import; if the layout or
# quantization ever drifts, import fails rather than writing a corrupt binary.
_GOLDEN_N = 32
_GOLDEN_K = 256
_GOLDEN_HEX = (
    "603cf9c5824e1ad7936f2cf8b5714e0ad3074b8ec2f5396da0e4175b8fc2f63a074a8eb2f5296ca0d4175b8ec2f6397dc6925f1be7a4603df9c5824e1bd7a36f7eb1e5286c9fd3074a8eb2f5296ca0d45f2ce8b4703df9c6925f1be7a4603df9e4285c9fc3064a7eb1e5286c9fd3074af8b5714e0ad6935f2ce8b4703df9c692fd2fc7afffafc92ffbafcd2ff7afd12f5b8fc2f63a7da1e4285c9fc3064a7eb1814e0ad7936f2cf8b5714e0ad6935f2cc2f5396da0e4175b8fc2f63a7db0e418f5396ca0d4175b8fc2f6397da0e4185be7a4603df9c5824e1bd7a36f2cf8b5816c9fd3074a8eb2f5396da0d4175b8fc2703df9c6825f1be7a4603df9c5824e1bd3064a7eb1f5286c9fd3074a8eb2f539f3afd52fb7afefafd92febafdd2fe7af0ad6935f2be8b4703df9c6825f1be7a43a7db1e4285c9fd3064a7eb1f5286c9f936f2cf8b5714d0ad6935f2be8b4703da0e4185b8fc2f63a7db1e4285c9fd3063cf9b5814e0ad7936f2cf8b5714d0ad6f9c5814e0bd7a36f2cf8b5814e0ad6934b8eb2f5396da0e4175b8fc2f6397da0825f1be7a4603df9c5824e1ad7a36f2ce12fe3afe52fdfafe92fb12fdbafed2fb1f5296c9fd3074b8eb2f5396da0e4172be8a4703df9c6825f1be7a4603cf9c5285c9fd3064a7eb1f5296c9fd3074b8eb5714d0ac6925f2be8a4703df9c6825f9fc2f63a7db1e4285c9fd3064a7eb1f54e0ad7936f2ce8b5714d0ac6925f2be80ad7a36f2cf8b5714e0ad6935f2ce8b4396da0e4175b8fc2f6397da0e4185c9fd7aff12fd3aff52fcfaff92fc12fcbafa4603cf9c5824e1ad7936f2cf8b5714eafd3074b8ec2f5396da0e4175b8fc2f6d3074a8eb2f5296ca0d4175b8ec2f639f9c6925f1be7a4603df9c5824e1bd7a34a7eb1e5286c9fd3074a8eb2f5296ca0935f2ce8b4703df9c6925f1be7a4603da1e4285c9fc3064a7eb1e5286c9fd3072cf8b5714e0ad6935f2ce8b4703df9c6fd2fc7afffafc92ffbafcd2ff7afd12f175b8fc2f63a7da1e4285c9fc3064a7ec5814e0ad7936f2cf8b5714e0ad6935f8ec2f5396da0e4175b8fc2f63a7db0e4b2f5396ca0d4175b8fc2f6397da0e4181be7a4603df9c5824e1bd7a36f2cf8b5286c9fd3074a8eb2f5396da0d4175b8fb4703df9c6825f1be7a4603df9c5824e9fd3064a7eb1f5286c9fd3074a8eb2f5f3afd52fb7afefafd92febafdd2fe7af4e0ad6935f2be8b4703df9c6825f1be7f63a7db1e4285c9fd3064a7eb1f5286cd7936f2cf8b5714d0ad6935f2be8b470a36f2cf8b5814e0ad6935f2cf8b4703d6f3cf9b5814e0ad7936f2cf8b5714d0a3df9c5814e0bd7a36f2cf8b5814e0ad6074b8eb2f5396da0e4175b8fc2f6397dc6825f1be7a4603df9c5824e1ad7a36fe12fe3afe52fad2fe92fb12fdbafed2f7eb1f5296c9fd3074b8eb2f5396da0e45f2be8a4703df9c6825f1be7a4603cf9e4285c9fd3064a7eb1f5296c9fd3074bf8b5714d0ac6925f2be8a4703df9c6825b9fc2f63a7db1e4285c9fd3064a7eb1814e0ad7936f2ce8b5714d0ac6925f2b4e0ad7a36f2cf8b5714e0ad6935f2ce8f5396da0e4175b8fc2f6397da0e4185cd7aff12fd3aff52fcfaff92fc12fcbafe7a4603cf9c5824e1ad7936f2cf8b5716cafd3074b8ec2f5396da0e4175b8fc29fd3074a8eb2f5296ca0d4175b8ec2f63df9c6925f1be7a4603df9c5824e1bd7064a7eb1e5286c9fd3074a8eb2f5296cd6935f2ce8b4703df9c6925f1be7a4607da1e4285c9fc3064a7eb1e5286c9fd36f2cf8b5714e0ad6935f2ce8b4703df9fd2fc7afffafc92ffbafcd2ff7afd12fe4175b8fc2f63a7da1e4285c9fc3064af9c5814e0ad7936f2cf8b5714e0ad6934b8ec2f5396da0e4175b8fc2f63a7db08eb2f5396ca0d4175b8fc2f6397da0e45f1be7a4603df9c5824e1bd7a36f2cf8f5286c9fd3074a8eb2f5396da0d4175be8b4703df9c6825f1be7a4603df9c5825c9fd3064a7eb1f5286c9fd3074a8eb2f3afd52fb7afefafd92febafdd2fe7af714e0ad6935f2be8b4703df9c6825f1bc2f63a7db1e4285c9fd3064a7eb1f5280ad7936f2cf8b5714d0ad6935f2be8b4d7a36f2cf8b5814e0ad6935f2cf8b4706da0e4175b8fc2f6397da0e4185b9fc3603df9c5824e1ad7a36f2cf8b5814e0ad3074b8eb2f5396da0e4175b8fc2f639f9c6825f1be7a4603df9c5824e1ad7a3e12fe3afe52fad2fdfafe92fdbafed2f4a7eb1f5296c9fd3074b8eb2f5396da0935f2be8a4703df9c6825f1be7a4603cb1e4285c9fd3064a7eb1f5296c9fd3072cf8b5714d0ac6925f2be8a4703df9c6185b9fc2f63a7db1e4285c9fd3064a7eb5814e0ad7936f2ce8b5714d0ac6925f814e0ad7a36f2cf8b5714e0ad6935f2cc2f5396da0e4175b8fc2f6397da0e418d7aff12fd3aff52fd0aff92fc12fccaf1be7a4603cf9c5824e1ad7936f2cf8b5296cafd3074b8ec2f5396da0e4175b8f6c9fd3074a8eb2f5296ca0d4175b8ec2703df9c6925f1be7a4603df9c5824e1bc3064a7eb1e5286c9fd3074a8eb2f5290ad6935f2ce8b4703df9c6925f1be7a4397da1e4285c9fc3064a7eb1e5286c9f936f2cf8b5714e0ad6935f2ce8b4703dfd2fc8af00b0c92ffbafcd2ff8afd12fa0e4175b8fc2f63a7da1e4285c9fc3063cf9c5814e0ad7936f2cf8b5714e0ad6074b8ec2f5396da0e4175b8fc2f63a7d4a8eb2f5396ca0d4175b8fc2f6397da0825f1be7a4603df9c5824e1bd7a36f2cb1f5286c9fd3074a8eb2f5396da0d4172be8b4703df9c6825f1be7a4603df9c5285c9fd3064a7eb1f5286c9fd3074a8ef4afd52fb8aff0afd92fecafdc2fe8afb5714e0ad6935f2be8b4703df9c6825f8fc2f63a7db1e4285c9fd3064a7eb1f54e0ad7936f2cf8b5714d0ad6935f2be80bd7a36f2cf8b5814e0ad6935f2cf8b4396da0e4175b8fc2f6397da0e4185b9fa4603df9c5824e1ad7a36f2cf8b5814e9fd3074b8eb2f5396da0e4175b8fc2f63df9c6825f1be7a4603df9c5824e1ad7e02fe4afe42fad2fe0afe82fdcafec2f064a7eb1f5296c9fd3074b8eb2f5396dc6935f2be8a4703df9c6825f1be7a4607db1e4285c9fd3064a7eb1f5296c9fd36f2cf8b5714d0ac6925f2be8a4703df9e4185b9fc2f63a7db1e4285c9fd3064af9b5814e0ad7936f2ce8b5714d0ac692c5814e0ad7a36f2cf8b5714e0ad6935f8ec2f5396da0e4175b8fc2f6397da0e4d8aff02fd4aff42fd0aff82fc02fccaf5e1be7a4603cf9c5824e1ad7936f2cf8f5296cafd3074b8ec2f5396da0e4175b286c9fd3074a8eb2f5296ca0d4175b8eb4703df9c6925f1be7a4603df9c5824e9fc3064a7eb1e5286c9fd3074a8eb2f54e0ad6935f2ce8b4703df9c6925f1be7f6397da1e4285c9fc3064a7eb1e5286cd7936f2cf8b5714e0ad6935f2ce8b470fc2fc8af00b0c82ffcafcc2ff8afd02f6da0e4175b8fc2f63a7da0e4185c9fd3a0d4175b8fc2f6397da0e4185b9fc2f6d3074b8ec2f5396da0e4175b8fc2f63a074a8eb2f5396ca0d4175b8fc2f6397dc6925f1be7a4603df9c5824e1bd7a36f7eb1f5286c9fd3074a8eb2f5396da0d45f2be8b4703df9c6825f1be7a4603df9e4285c9fd3064a7eb1f5286c9fd3074abcaff4afb8aff0afd82fecafdc2fe8aff8b5714e0ad6935f2be8b4703df9c6825b8fc2f63a7db1e4285c9fd3064a7eb1814e0ad7936f2cf8b5714d0ad6935f2b4e0bd7a36f2cf8b5814e0ad6935f2cf8f5396da0e4175b8fc2f6397da0e4185be7a4603df9c5824e1ad7a36f2cf8b5816c9fd3074b8eb2f5396da0e4175b8fc2703df9c6825f1be7a4603df9c5824e1ae02fe4afe42fac2fe0afe82fdcafec2fd3064a7eb1f5296c9fd3074b8eb2f5390ac6935f2be8a4703df9c6825f1be7a43a7db1e4285c9fd3064a7eb1f5296c9f936f2cf8b5714d0ac6925f2be8a4703da0e4185b9fc2f63a7db1e4285c9fd3062cf9b5814e0ad7936f2ce8b5714d0ac6f9c5814e0ad7a36f2cf8b5714e0ad6934b8ec2f5396da0e4175b8fc2f6397da0d8aff02fd4aff42fd0aff82fc02fccaf825e1be7a4603cf9c5824e1ad7936f2cb1f5296cafd3074b8ec2f5396da0e417e5286c9fd3074a8eb2f5296ca0d4175be8b4703df9c6925f1be7a4603df9c5825c9fc3064a7eb1e5286c9fd3074a8eb2714e0ad6935f2ce8b4703df9c6925f1bc2f6397da1e4285c9fc3064a7eb1e5280ad7936f2cf8b5714e0ad6935f2ce8b4fc2fc8af00b0c82ffcafcc2ff8afd02f396da0e4175b8fc2f63a7da0e4185c9f6ca0d4175b8fc2f6397da0e4185b9fc2603df9c5824e1bd7a36f2cf9b5814e0ad3074a8eb2f5396ca0d4175b8fc2f639f9c6925f1be7a4603df9c5824e1bd7a34a7eb1f5286c9fd3074a8eb2f5396da0935f2be8b4703df9c6825f1be7a4603da1e4285c9fd3064a7eb1f5286c9fd307bcaff4afd42ff0afd82fecafdc2fe8af2cf8b5714e0ad6935f2be8b4703df9c6185b8fc2f63a7db1e4285c9fd3064a7eb5814e0ad7936f2cf8b5714d0ad6935f814e0bd7a36f2cf8b5814e0ad6935f2cb2f5396da0e4175b8fc2f6397da0e4181be7a4603df9c5824e1ad7a36f2cf8b5286c9fd3074b8eb2f5396da0e4175b8fa4703df9c6825f1be7a4603df9c5824ee02fe4afe42fac2fe0afe82fdcafec2f9fd3064a7eb1f5296c9fd3074b8eb2f54d0ac6935f2be8a4703df9c6825f1be7f63a7db1e4285c9fd3064a7eb1f5296cd7936f2cf8b5714d0ac6925f2be8a470a36f2cf8b5714e0ad6935f2ce8b4703d6f2cf9b5814e0ad7936f2ce8b5714d0a3df9c5814e0ad7a36f2cf8b5714e0ad6074b8ec2f5396da0e4175b8fc2f6397dd8aff02fd4aff42fbc2ff82fc02fccafc6825e1be7a4603cf9c5824e1ad7936f7eb1f5296cafd3074b8ec2f5396da0e4b1e5286c9fd3074a8eb2f5296ca0d4172ce8b4703dfac6925f1be7a4603df9c5285c9fc3064a7eb1e5286c9fd3074a8eb5714e0ad6935f2ce8b4703df9c6925f8fc2f6397da1e4285c9fc3064a7eb1e54e0ad7936f2cf8b5714e0ad6935f2ce8fc2fc8af00b0c82ffcafcc2ff8afd02ff5396da0e4175b8fc2f63a7da0e4185c296ca0d4175b8fc2f6397da0e4185b9fa4603df9c5824e1bd7a36f2cf9b5814e9fd3074a8eb2f5396ca0d4175b8fc2f63df9c6925f1be7a4603df9c5824e1bd7064a7eb1f5286c9fd3074a8eb2f5396dd6935f2be8b4703df9c6825f1be7a4607da1e4285c9fd3064a7eb1f5286c9fd3bcaff4afd42ff0afd82fecafdc2fe8af6f2cf8b5714e0ad6935f2be8b4703df9e4185b8fc2f63a7db1e4285c9fd3064af9b5814e0ad7936f2cf8b5714d0ad693c5814e0bd7a36f2cf8b5814e0ad6935f8eb2f5396da0e4175b8fc2f6397da0e45f1be7a4603df9c5824e1ad7a36f2cf8f5286c9fd3074b8eb2f5396da0e4175be8a4703df9c6825f1be7a4603df9c582e02fe4afe42fac2fe0afe82fdcafec2f5c9fd3064a7eb1f5296c9fd3074b8eb2714d0ac6935f2be8a4703df9c6825f1bc2f63a7db1e4285c9fd3064a7eb1f5290ad7936f2cf8b5714d0ac6925f2be8a4d7a36f2cf8b5714e0ad6935f2ce8b4706da0e4175b8fc2f6397da0e4185c9fc3603cf9c5824e1ad7936f2cf8b5714e0ad3074b8ec2f5396da0e4175b8fc2f639d8aff02fd4aff42fbc2fd0aff82fccaff9c6825e1be7a4603cf9c5824e1ad7934a7eb1f5296cafd3074b8ec2f5396da0925f2be8a4703df9c6825e1be7a4603c5f2ce8b4703dfac6925f1be7a4603df9e4285c9fc3064a7eb1e5286c9fd3074af8b5714e0ad6935f2ce8b4703df9c6925b8fc2f6397da1e4285c9fc3064a7eb1814e0ad7936f2cf8b5714e0ad6935f2cfc2fc8af0030c82ffcafcc2ff8afd02fc2f5396da0e4175b8fc2f63a7da0e418f5296ca0d4175b8fc2f6397da0e4185be7a4603df9c5824e1bd7a36f2cf9b5816c9fd3074a8eb2f5396ca0d4175b8fc2703df9c6925f1be7a4603df9c5824e1bd3064a7eb1f5286c9fd3074a8eb2f5390ad6935f2be8b4703df9c6825f1be7a43a7da1e4285c9fd3064a7eb1f5286c9fbcaff4afd42ff0afd82fecafdc2fe8af936f2cf8b5714e0ad6935f2be8b4703da0e4185b8fc2f63a7db1e4285c9fd3063cf9b5814e0ad7936f2cf8b5714d0ad6f9c5814e0bd7a36f2cf8b5814e0ad6934b8eb2f5396da0e4175b8fc2f6397da0825f1be7a4603df9c5824e1ad7a36f2cb1f5286c9fd3074b8eb2f5396da0e4172be8a4703df9c6825f1be7a4603df9c5e02fe4afe42fac2fe0afe82fdcafec2f285c9fd3064a7eb1f5296c9fd3074b8eb5714d0ac6935f2be8a4703df9c6825f9fc2f63a7db1e4285c9fd3064a7eb1f54e0ad7936f2cf8b5714d0ac6925f2be80ad7a36f2cf8b5714e0ad6935f2ce8b4396da0e4175b8fc2f6397da0e4185c9fa4603cf9c5824e1ad7936f2cf8b5714e9fd3074b8ec2f5396da0e4175b8fc2f6d8aff02fd4aff42fbc2fd0aff82fccaf"
)  # nntrainer::quantizeAndRepackQ4_0X4x2 on _fixture(32, 256)


def _fixture(N: int, K: int) -> np.ndarray:
    """Deterministic FP32 [N, K] fixture, reproducible bit-for-bit in C++ via
    the same integer LCG (so the golden comparison is exact)."""
    idx = np.arange(N * K, dtype=np.uint64)
    state = (idx * np.uint64(1103515245) + np.uint64(12345)) & np.uint64(0xFFFFFFFF)
    u16 = ((state >> np.uint64(8)) & np.uint64(0xFFFF)).astype(np.float64)
    w = (u16 / 65535.0) * 2.0 - 1.0
    return w.astype(np.float32).reshape(N, K)


def _self_test() -> None:
    N, K = _GOLDEN_N, _GOLDEN_K
    w = _fixture(N, K)
    packed = quantize_and_repack_q4_0_x4x2(w)

    # (1) round-trip: x4x2 -> blocks -> x4x2 is byte-identical.
    blocks = quantize_q4_0(w)
    roundtrip = unpack_x4x2_to_blocks(packed, N, K)
    if roundtrip != blocks:
        raise RuntimeError(
            "q4_0_x4x2_quant: x4x2 round-trip does not reproduce the Q4_0 "
            "block stream. Refusing to write a corrupt weight binary."
        )

    # (2) cross-language: match the C++ golden exactly (once _GOLDEN_HEX set).
    if _GOLDEN_HEX:
        golden = bytes.fromhex(_GOLDEN_HEX)
        if packed != golden:
            raise RuntimeError(
                "q4_0_x4x2_quant: output does not match the C++ "
                "quantizeAndRepackQ4_0X4x2 golden. Layout/quant drift detected."
            )


_self_test()
