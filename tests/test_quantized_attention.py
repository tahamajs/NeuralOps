"""Tests for FP8 and INT8 attention."""
import torch
from neuralops.core.quantized_attention import (
    fp8_scaled_attention,
    int8_scaled_attention,
    quantize_to_int8,
    dequantize_from_int8,
)


def test_quantize_int8():
    x = torch.randn(2, 4, 8)
    q_x, scale = quantize_to_int8(x)
    assert q_x.dtype == torch.int8
    deq = dequantize_from_int8(q_x, scale)
    assert torch.allclose(x, deq, atol=0.1)


def test_int8_scaled_attention():
    q = torch.randn(2, 2, 4, 8)
    k = torch.randn(2, 2, 4, 8)
    v = torch.randn(2, 2, 4, 8)
    out = int8_scaled_attention(q, k, v)
    assert out.shape == (2, 2, 4, 8)
    assert not torch.isnan(out).any()


def test_fp8_scaled_attention():
    q = torch.randn(2, 2, 4, 8)
    k = torch.randn(2, 2, 4, 8)
    v = torch.randn(2, 2, 4, 8)
    out = fp8_scaled_attention(q, k, v)
    assert out.shape == (2, 2, 4, 8)
    assert not torch.isnan(out).any()
