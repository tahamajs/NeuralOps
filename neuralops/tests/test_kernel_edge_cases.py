"""Edge case tests: seq_len=1, seq_len > window, empty batch, non-power-of-2 dims."""
import pytest
import torch
from neuralops.core.triton_kernels.sliding_window import sliding_window_attention_kernel
from neuralops.core.triton_kernels.flash_decoding import flash_decoding_attention
from neuralops.core.rope import fused_rope_embedding
from neuralops.core.triton_kernels.elementwise import fused_swiglu


def test_edge_case_seqlen_one():
    # Decoding token
    q = torch.randn(2, 4, 1, 32)
    k = torch.randn(2, 4, 1, 32)
    v = torch.randn(2, 4, 1, 32)
    out = sliding_window_attention_kernel(q, k, v, window_size=16)
    assert out.shape == (2, 4, 1, 32)
    assert not torch.isnan(out).any()


def test_edge_case_seqlen_greater_than_window():
    q = torch.randn(1, 2, 64, 16)
    k = torch.randn(1, 2, 64, 16)
    v = torch.randn(1, 2, 64, 16)
    out = sliding_window_attention_kernel(q, k, v, window_size=8)
    assert out.shape == (1, 2, 64, 16)


def test_edge_case_non_power_of_two():
    q = torch.randn(1, 3, 17, 30)
    k = torch.randn(1, 3, 17, 30)
    cos = torch.randn(1, 1, 17, 30)
    sin = torch.randn(1, 1, 17, 30)
    oq, ok = fused_rope_embedding(q, k, cos, sin)
    assert oq.shape == (1, 3, 17, 30)
    assert ok.shape == (1, 3, 17, 30)


def test_edge_case_flash_decoding_single_chunk():
    q = torch.randn(2, 2, 1, 16)
    k = torch.randn(2, 2, 5, 16)
    v = torch.randn(2, 2, 5, 16)
    out = flash_decoding_attention(q, k, v, chunk_size=32)
    assert out.shape == (2, 2, 1, 16)
