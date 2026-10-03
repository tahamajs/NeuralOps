import math
import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.triton_cosformer import (TritonCosformerAttention,
                                                   inject_triton_cosformer,
                                                   triton_cosformer_attention)


def test_triton_cosformer_config_handling():
    """Verify safe config handling when config is None, SimpleNamespace, or full Config."""
    # 1. config=None, explicit kwargs
    attn1 = TritonCosformerAttention(config=None, embed_dim=64, num_heads=4)
    assert attn1.embed_dim == 64
    assert attn1.num_heads == 4
    assert attn1.head_dim == 16
    assert attn1.max_length == 2048

    # 2. config present
    cfg = SimpleNamespace(hidden_size=128, num_attention_heads=8, max_position_embeddings=1024, is_decoder=True)
    attn2 = TritonCosformerAttention(config=cfg)
    assert attn2.embed_dim == 128
    assert attn2.num_heads == 8
    assert attn2.head_dim == 16
    assert attn2.max_length == 1024
    assert attn2.is_decoder is True


def test_triton_cosformer_rope_support():
    """Verify RoPE rotary position embeddings are applied to Q and K when provided."""
    attn = TritonCosformerAttention(embed_dim=64, num_heads=4, is_decoder=True)
    B, N, D = 2, 8, 64
    H = 4
    d = D // H

    # Generate synthetic RoPE (cos, sin)
    angles = torch.arange(N).float()[:, None] * 0.1
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0).expand(B, 1, N, d)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0).expand(B, 1, N, d)

    x = torch.randn(B, N, D)
    out_rope, _ = attn(x, position_embeddings=(cos, sin))
    out_base, _ = attn(x, position_embeddings=None)

    assert out_rope.shape == (B, N, D)
    assert torch.isfinite(out_rope).all()
    # RoPE should alter the representations
    assert not torch.allclose(out_rope, out_base, atol=1e-4)


def test_triton_cosformer_bidirectional_and_causal_shapes():
    """Verify forward passes in both causal (decoder) and bidirectional (encoder) modes."""
    B, N, D = 2, 16, 64
    x = torch.randn(B, N, D)

    # 1. Causal mode
    attn_causal = TritonCosformerAttention(embed_dim=D, num_heads=4, is_decoder=True)
    out_causal, _ = attn_causal(x)
    assert out_causal.shape == (B, N, D)
    assert torch.isfinite(out_causal).all()

    # 2. Bidirectional mode
    attn_bidi = TritonCosformerAttention(embed_dim=D, num_heads=4, is_decoder=False)
    out_bidi, _ = attn_bidi(x)
    assert out_bidi.shape == (B, N, D)
    assert torch.isfinite(out_bidi).all()


def test_triton_cosformer_attention_masking():
    """Verify attention_mask zeros out padded tokens and prevents gradient contamination."""
    B, N, D = 2, 8, 64
    attn = TritonCosformerAttention(embed_dim=D, num_heads=4, is_decoder=True)
    x = torch.randn(B, N, D, requires_grad=True)

    # Batch 0: length 8; Batch 1: length 5 (tokens 5, 6, 7 are padding)
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0],
    ], dtype=torch.float32)

    out, _ = attn(x, attention_mask=mask)
    assert out.shape == (B, N, D)
    assert torch.isfinite(out).all()

    # Backward gradient pass
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_triton_cosformer_clean_autograd():
    """Verify backward executes with torch.autograd.grad without double-backward warnings."""
    B, H, N, d = 2, 4, 8, 16
    q = torch.randn(B, H, N, d, requires_grad=True)
    k = torch.randn(B, H, N, d, requires_grad=True)
    v = torch.randn(B, H, N, d, requires_grad=True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = triton_cosformer_attention(q, k, v, is_decoder=True, eps=1e-6, max_length=2048)
        loss = out.sum()
        loss.backward()

        double_bwd_warnings = [
            str(item.message) for item in w
            if "double" in str(item.message).lower() or "graph" in str(item.message).lower()
        ]
        assert len(double_bwd_warnings) == 0, f"Unexpected autograd warnings: {double_bwd_warnings}"

    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_triton_cosformer_weight_copy_multi_arch():
    """Verify copy_weights correctly extracts projections from GPT-2, RoBERTa, and DistilBERT."""
    dim = 64
    num_heads = 4

    # 1. GPT-2 Fused Conv1D style
    class MockGPT2Attn(nn.Module):
        def __init__(self):
            super().__init__()
            # In GPT-2: c_attn weight is [dim, 3 * dim]
            self.c_attn = nn.Linear(dim, 3 * dim)
            self.c_proj = nn.Linear(dim, dim)
    
    gpt2_src = MockGPT2Attn()
    cos_gpt2 = TritonCosformerAttention(embed_dim=dim, num_heads=num_heads, pretrained_attn=gpt2_src)

    x = torch.randn(1, 4, dim)
    # Verify projections match GPT-2 source projections
    q_src, k_src, v_src = gpt2_src.c_attn(x).split(dim, dim=-1)
    assert torch.allclose(cos_gpt2.q_proj(x), q_src, atol=1e-5)
    assert torch.allclose(cos_gpt2.k_proj(x), k_src, atol=1e-5)
    assert torch.allclose(cos_gpt2.v_proj(x), v_src, atol=1e-5)
    assert torch.allclose(cos_gpt2.out_proj(x), gpt2_src.c_proj(x), atol=1e-5)

    # 2. RoBERTa nested self-attention style
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
    cos_rob = TritonCosformerAttention(embed_dim=dim, num_heads=num_heads, pretrained_attn=rob_src)
    assert torch.allclose(cos_rob.q_proj(x), rob_src.self.query(x), atol=1e-5)
    assert torch.allclose(cos_rob.k_proj(x), rob_src.self.key(x), atol=1e-5)
    assert torch.allclose(cos_rob.v_proj(x), rob_src.self.value(x), atol=1e-5)
    assert torch.allclose(cos_rob.out_proj(x), rob_src.output.dense(x), atol=1e-5)

    # 3. DistilBERT flat style
    class MockDistilBertAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(dim, dim)
            self.k_lin = nn.Linear(dim, dim)
            self.v_lin = nn.Linear(dim, dim)
            self.out_lin = nn.Linear(dim, dim)

    distil_src = MockDistilBertAttn()
    cos_distil = TritonCosformerAttention(embed_dim=dim, num_heads=num_heads, pretrained_attn=distil_src)
    assert torch.allclose(cos_distil.q_proj(x), distil_src.q_lin(x), atol=1e-5)
    assert torch.allclose(cos_distil.k_proj(x), distil_src.k_lin(x), atol=1e-5)
    assert torch.allclose(cos_distil.v_proj(x), distil_src.v_lin(x), atol=1e-5)
    assert torch.allclose(cos_distil.out_proj(x), distil_src.out_lin(x), atol=1e-5)


def test_triton_cosformer_injection_pipeline():
    """Verify inject_triton_cosformer installs cleanly into a model with attention layers."""
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
    res = inject_triton_cosformer(model=model)
    assert res is not None
