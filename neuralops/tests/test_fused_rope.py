"""Tests for fused RoPE embedding."""
import torch
from neuralops.core.rope import fused_rope_embedding, apply_rotary_pos_emb


def test_fused_rope_embedding():
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 16, 64
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    cos = torch.randn(1, 1, N, D)
    sin = torch.randn(1, 1, N, D)

    q_ref, k_ref = apply_rotary_pos_emb(q, k, cos, sin)
    q_fused, k_fused = fused_rope_embedding(q, k, cos, sin)

    assert torch.allclose(q_ref, q_fused, atol=1e-5)
    assert torch.allclose(k_ref, k_fused, atol=1e-5)
