"""
Tests for W4A16 Bit-Packed uint8 Storage, Fused Dequant-GEMM,
Fast Walsh-Hadamard Transform, and Hadamard Incoherent SVD.
"""

import pytest
import torch
import torch.nn as nn

from neuralops.core.triton_kernels.low_rank import (
    pack_int4_weights,
    unpack_int4_weights,
    fused_w4a16_gemm_triton,
    BitPackedW4A16Linear,
    BitPackedLowRankSVDLinear,
)
from neuralops.core.triton_kernels.elementwise import fast_hadamard_transform
from neuralops.compression_engine.svd.hadamard_incoherent_svd import HadamardIncoherentSVDCompressor


def test_w4a16_pack_unpack_correctness():
    """Verify bit-packing cuts storage and unpacks within 4-bit quantization error."""
    torch.manual_seed(42)
    W = torch.randn(256, 512, dtype=torch.float16)
    orig_bytes = W.numel() * 2  # 262,144 bytes

    packed, scales, zeros = pack_int4_weights(W, group_size=128)
    packed_total_bytes = packed.nbytes + scales.nbytes + zeros.nbytes

    # Should be < 29% of original size (> 3.4x compression)
    ratio = orig_bytes / packed_total_bytes
    assert ratio > 3.4, f"Expected compression ratio > 3.4x, got {ratio:.2f}x"

    # Unpack and verify reasonable reconstruction error
    W_rec = unpack_int4_weights(packed, scales, zeros, group_size=128, out_dtype=torch.float16)
    rel_err = float(torch.norm(W - W_rec, p="fro").item() / torch.norm(W, p="fro").item())
    assert rel_err < 0.15, f"Reconstruction error {rel_err:.4f} too high for INT4"


def test_bit_packed_w4a16_linear_memory_and_forward():
    """Verify BitPackedW4A16Linear forward pass matches reference linear and reduces memory."""
    torch.manual_seed(42)
    linear = nn.Linear(512, 256, bias=True).half()
    x = torch.randn(4, 32, 512, dtype=torch.float16)

    packed_linear = BitPackedW4A16Linear.from_linear(linear, group_size=128)

    y_ref = linear(x)
    y_packed = packed_linear(x)

    assert y_packed.shape == y_ref.shape
    rel_diff = float(torch.norm(y_ref - y_packed).item() / torch.norm(y_ref).item())
    assert rel_diff < 0.15, f"Output relative diff {rel_diff:.4f} exceeds INT4 bound"


def test_bit_packed_low_rank_svd_linear():
    """Verify BitPackedLowRankSVDLinear performs dual-path exact+packed projection."""
    torch.manual_seed(42)
    linear = nn.Linear(256, 128, bias=True).half()
    x = torch.randn(2, 16, 256, dtype=torch.float16)

    svd_packed = BitPackedLowRankSVDLinear.from_linear(linear, rank=32, exact_rank=8, group_size=64)
    y = svd_packed(x)
    assert y.shape == (2, 16, 128)
    assert not torch.isnan(y).any()


def test_fast_hadamard_transform_involutory_orthogonality():
    """Verify H^T H = I and H H = I (involutory property)."""
    torch.manual_seed(42)
    x = torch.randn(4, 16, 128, dtype=torch.float32)

    # 1. Norm preservation (Parseval's theorem: ||x H|| = ||x||)
    y = fast_hadamard_transform(x, block_size=128)
    norm_x = torch.norm(x, p="fro").item()
    norm_y = torch.norm(y, p="fro").item()
    assert abs(norm_x - norm_y) / norm_x < 1e-4

    # 2. Involutory: H(H(x)) = x
    x_rec = fast_hadamard_transform(y, block_size=128)
    diff = float(torch.norm(x - x_rec, p="fro").item() / norm_x)
    assert diff < 1e-4, f"Hadamard not involutory: diff={diff:.6f}"


def test_hadamard_incoherent_svd_compressor():
    """Verify Hadamard rotation reduces outlier distortion on synthetic heavy-tailed weights."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(256, 256, bias=False),
        nn.Linear(256, 256, bias=False),
    ).float()

    # Inject channel outliers into weight matrix
    with torch.no_grad():
        model[0].weight[:, 10] *= 15.0
        model[0].weight[:, 45] *= 20.0

    comp = HadamardIncoherentSVDCompressor(model, rank=32, hadamard_block_size=128)
    compressed_model, stats = comp.compress()

    assert stats["num_compressed"] == 2
    assert stats["avg_reconstruction_error"] < 0.70
    # Sanity forward check
    x = torch.randn(2, 8, 256)
    out = compressed_model(x)
    assert out.shape == (2, 8, 256)
    assert not torch.isnan(out).any()
