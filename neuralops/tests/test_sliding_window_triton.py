"""Tests for Sliding Window Triton kernel and fallback."""
import torch
from neuralops.core.triton_kernels.sliding_window import sliding_window_attention_kernel, sliding_window_pytorch


def test_sliding_window_kernel_matches_reference():
    torch.manual_seed(42)
    B, H, N, D = 2, 2, 64, 32
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    ref = sliding_window_pytorch(q, k, v, window_size=16)
    out = sliding_window_attention_kernel(q, k, v, window_size=16)

    assert torch.allclose(ref, out, atol=1e-5)
    assert out.shape == (B, H, N, D)
