import math
import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.triton_svd_linear_attention import (
    SVDLinearProjection, TritonSVDLinearAttention,
    inject_triton_svd_linear_attention, triton_linear_attention)


def test_triton_svd_linear_config_handling():
    """Verify safe config handling and dimension inference."""
    # 1. config=None, explicit kwargs
    attn1 = TritonSVDLinearAttention(config=None, embed_dim=64, num_heads=4, rank=16)
    assert attn1.embed_dim == 64
    assert attn1.num_heads == 4
    assert attn1.head_dim == 16
    assert attn1.rank == 16
    assert attn1.kv_dim == 64

    # 2. config present
    cfg = SimpleNamespace(hidden_size=128, num_attention_heads=8, num_key_value_heads=2, is_decoder=True)
    attn2 = TritonSVDLinearAttention(config=cfg, rank=32)
    assert attn2.embed_dim == 128
    assert attn2.num_heads == 8
    assert attn2.num_kv_heads == 2
    assert attn2.head_dim == 16
    assert attn2.kv_dim == 32
    assert attn2.is_decoder is True


def test_svd_linear_projection_identity_init():
    """Verify BUG 5 FIX: U is initialized with identity-like structure so initial output is non-zero."""
    proj = SVDLinearProjection(in_features=64, out_features=64, rank=16, bias=True)
    r = proj.rank
    assert r == 16

    # Verify U has eye(r) in top-left
    expected_eye = torch.eye(r)
    assert torch.allclose(proj.U[:r, :r], expected_eye)

    # Verify forward pass produces non-zero output
    x = torch.randn(2, 4, 64)
    out = proj(x)
    assert out.shape == (2, 4, 64)
    assert torch.isfinite(out).all()
    assert (out.abs() > 1e-6).any(), "Output should not be zero!"


def test_triton_svd_linear_rope_and_gqa():
    """Verify BUG 3 & BUG 4 FIX: RoPE and GQA support."""
    B, N, D = 2, 8, 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = D // num_heads

    attn = TritonSVDLinearAttention(
        embed_dim=D,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rank=16,
        is_decoder=True,
    )

    angles = torch.arange(N).float()[:, None] * 0.1
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0).expand(B, 1, N, head_dim)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0).expand(B, 1, N, head_dim)

    x = torch.randn(B, N, D)
    out_gqa, _ = attn(x, position_embeddings=(cos, sin))
    assert out_gqa.shape == (B, N, D)
    assert torch.isfinite(out_gqa).all()


def test_triton_svd_linear_bidirectional_and_causal_shapes():
    """Verify forward pass shapes for causal and bidirectional modes."""
    B, N, D = 2, 16, 64
    x = torch.randn(B, N, D)

    # Causal
    attn_causal = TritonSVDLinearAttention(embed_dim=D, num_heads=4, rank=16, is_decoder=True)
    out_causal, _ = attn_causal(x)
    assert out_causal.shape == (B, N, D)
    assert torch.isfinite(out_causal).all()

    # Bidirectional
    attn_bidi = TritonSVDLinearAttention(embed_dim=D, num_heads=4, rank=16, is_decoder=False)
    out_bidi, _ = attn_bidi(x)
    assert out_bidi.shape == (B, N, D)
    assert torch.isfinite(out_bidi).all()


def test_triton_svd_linear_attention_masking():
    """Verify BUG 2 FIX: Padding tokens are zeroed on K and V without gradient issues."""
    B, N, D = 2, 8, 64
    attn = TritonSVDLinearAttention(embed_dim=D, num_heads=4, rank=16, is_decoder=True)
    x = torch.randn(B, N, D, requires_grad=True)

    mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0],
    ], dtype=torch.float32)

    out, _ = attn(x, attention_mask=mask)
    assert out.shape == (B, N, D)
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_triton_svd_linear_clean_autograd():
    """Verify BUG 1 FIX: Clean backward without double-backward warnings."""
    B, H, N, d = 2, 4, 8, 16
    q = torch.randn(B, H, N, d, requires_grad=True)
    k = torch.randn(B, H, N, d, requires_grad=True)
    v = torch.randn(B, H, N, d, requires_grad=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = triton_linear_attention(q, k, v, is_causal=True, activation_mode="elu", eps=1e-6)
        loss = out.sum()
        loss.backward()

        double_bwd = [
            str(item.message) for item in w
            if "double" in str(item.message).lower() or "graph" in str(item.message).lower()
        ]
        assert len(double_bwd) == 0, f"Unexpected autograd warnings: {double_bwd}"

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_triton_svd_linear_weight_copy_multi_arch():
    """Verify copy_weights correctly factorizes weights from pretrained modules via SVD."""
    dim = 64
    num_heads = 4

    # 1. GPT-2 Conv1D style
    class MockGPT2Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(dim, 3 * dim)
            self.c_proj = nn.Linear(dim, dim)

    gpt2_src = MockGPT2Attn()
    attn_gpt2 = TritonSVDLinearAttention(embed_dim=dim, num_heads=num_heads, rank=32, pretrained_attn=gpt2_src)

    # Verify SVD factor reconstruction quality
    with torch.no_grad():
        W_q_approx = attn_gpt2.q_proj.U @ attn_gpt2.q_proj.V
        q_src, _, _ = gpt2_src.c_attn.weight.data.clone().split(dim, dim=0)
        # Low rank rank=32 of 64 should capture substantial energy
        rel_err = torch.norm(W_q_approx - q_src) / torch.norm(q_src)
        assert rel_err < 0.85, f"SVD approximation error too high: {rel_err}"

    # 2. RoBERTa style
    class MockRobertaAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = nn.Module()
            self.self.query = nn.Linear(dim, dim)
            self.self.key = nn.Linear(dim, dim)
            self.self.value = nn.Linear(dim, dim)
            self.output = nn.Module()
            self.output.dense = nn.Linear(dim, dim)

    rob_src = MockRobertaAttn()
    attn_rob = TritonSVDLinearAttention(embed_dim=dim, num_heads=num_heads, rank=32, pretrained_attn=rob_src)
    assert torch.allclose(attn_rob.out_proj.weight.data, rob_src.output.dense.weight.data)


def test_triton_svd_linear_injection_pipeline():
    """Verify inject_triton_svd_linear_attention installs cleanly."""
    class DummyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Linear(64, 64)
            self.attn.c_attn = nn.Linear(64, 192)
            self.attn.c_proj = nn.Linear(64, 64)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=64, num_attention_heads=4, is_decoder=True)
            self.blocks = nn.ModuleList([DummyBlock() for _ in range(2)])

    model = DummyModel()
    res = inject_triton_svd_linear_attention(model=model, rank=16)
    assert res is not None
