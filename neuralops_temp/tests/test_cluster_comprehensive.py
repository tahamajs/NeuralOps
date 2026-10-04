from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.cluster_attention import (ClusterAttention,
                                                    LoRALinear, inject_cluster)


def test_cluster_lora_config_independence():
    # 1. config=None, comp_config present
    comp_cfg = SimpleNamespace(lora_rank=8, lora_alpha=16.0)
    p1 = ClusterAttention(config=None, comp_config=comp_cfg, d_model=64, num_heads=4)
    assert p1.lora_rank == 8
    assert p1.lora_alpha == 16.0

    # 2. config=None, comp_config=None, kwargs override
    p2 = ClusterAttention(config=None, comp_config=None, d_model=64, num_heads=4, lora_rank=4, lora_alpha=8.0)
    assert p2.lora_rank == 4
    assert p2.lora_alpha == 8.0

    # 3. config present
    cfg = SimpleNamespace(lora_rank=12, lora_alpha=24.0, num_attention_heads=4, hidden_size=64)
    p3 = ClusterAttention(config=cfg)
    assert p3.lora_rank == 12
    assert p3.lora_alpha == 24.0

def test_cluster_rotary_fallback():
    class MockRotaryEmb(nn.Module):
        def forward(self, x, seq_len=None):
            B, H, N, d = x.shape
            cos = torch.ones(B, 1, N, d)
            sin = torch.zeros(B, 1, N, d)
            return cos, sin

    class MockAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
            self.rotary_emb = MockRotaryEmb()

    old = MockAttn()
    p = ClusterAttention(d_model=64, num_heads=4, pretrained_attn=old)
    x = torch.randn(2, 8, 64)
    out, _ = p(x, position_embeddings=None)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()

def test_cluster_bidirectional_and_causal_shapes():
    x = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10)
    mask[:, 7:] = 0  # 3 padding tokens

    # Bidirectional (encoder) with clustering
    p_bidi = ClusterAttention(d_model=64, num_heads=4, is_causal=False, is_decoder=False, lora_rank=8, num_clusters=4)
    out_bidi, _ = p_bidi(x, attention_mask=mask)
    assert out_bidi.shape == (2, 10, 64)
    assert torch.isfinite(out_bidi).all()

    # Causal (decoder) with exact attention
    p_causal = ClusterAttention(d_model=64, num_heads=4, is_causal=True, is_decoder=True, lora_rank=8)
    out_causal, _ = p_causal(x, attention_mask=mask)
    assert out_causal.shape == (2, 10, 64)
    assert torch.isfinite(out_causal).all()

def test_cluster_kmeans_forward():
    p = ClusterAttention(d_model=64, num_heads=4, num_clusters=4, n_iters=3, window_size=16)
    x = torch.randn(2, 12, 64)
    out, attn = p(x, output_attentions=True)
    assert out.shape == (2, 12, 64)
    assert torch.isfinite(out).all()
    if attn is not None:
        assert attn.shape == (2, 4, 4, 12)  # [B, H, K, L]

def test_cluster_frobenius_norm_verification():
    class ZeroAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
            self.q_proj.weight.data.zero_()  # Zero norm

    old = ZeroAttn()
    with pytest.raises(RuntimeError, match="norm"):
        ClusterAttention(d_model=64, num_heads=4, pretrained_attn=old)

def test_cluster_fallback_on_copy_weights_failure():
    class BrokenLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
            self.attention.weight.data.zero_()  # Causes copy_weights to raise
    
    layer = BrokenLayer()
    res = inject_cluster(layer=layer)
    assert res is None  # Graceful fallback, no crash

def test_cluster_lora_adapters_freezing():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
    old = DummyAttn()
    p = ClusterAttention(d_model=64, num_heads=4, pretrained_attn=old, lora_rank=8)
    assert not p.q_proj.weight.requires_grad
    assert p.q_proj.lora_A.requires_grad
    assert p.q_proj.lora_B.requires_grad
    assert not p.out_proj.weight.requires_grad
    assert p.out_proj.lora_A.requires_grad
    assert p.out_proj.lora_B.requires_grad

def test_cluster_injection_all_4_architectures():
    class MockAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
            self.hidden_size = 64
            self.num_heads = 4

    # 1. DistilBERT clone
    class DistilBertClone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.attention = MockAttn()
            self.distilbert = nn.Module()
            self.distilbert.transformer = nn.Module()
            self.distilbert.transformer.layer = nn.ModuleList([layer])
    d_model = DistilBertClone()
    res_d = inject_cluster(model=d_model, lora_rank=8)
    assert res_d is not None
    assert isinstance(d_model.distilbert.transformer.layer[0].attention, ClusterAttention)

    # 2. RoBERTa clone
    class RobertaClone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.attention = MockAttn()
            self.roberta = nn.Module()
            self.roberta.encoder = nn.Module()
            self.roberta.encoder.layer = nn.ModuleList([layer])
    r_model = RobertaClone()
    res_r = inject_cluster(model=r_model, lora_rank=8)
    assert res_r is not None
    assert isinstance(r_model.roberta.encoder.layer[0].attention, ClusterAttention)

    # 3. GPT-2 clone
    class GPT2Clone(nn.Module):
        def __init__(self):
            super().__init__()
            block = nn.Module()
            block.attn = MockAttn()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([block])
    g_model = GPT2Clone()
    res_g = inject_cluster(model=g_model, lora_rank=8)
    assert res_g is not None
    assert isinstance(g_model.transformer.h[0].attn, ClusterAttention)

    # 4. Qwen3.5 clone
    class QwenClone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.self_attn = MockAttn()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([layer])
    q_model = QwenClone()
    res_q = inject_cluster(model=q_model, lora_rank=8)
    assert res_q is not None
    assert isinstance(q_model.model.layers[0].self_attn, ClusterAttention)


def test_cluster_config_none_no_attr_error():
    """config=None must not raise AttributeError when checking is_decoder in forward."""
    p = ClusterAttention(config=None, d_model=64, num_heads=4, lora_rank=0)
    x = torch.randn(2, 8, 64)
    out, _ = p(x)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


def test_cluster_rope_uses_query_not_key():
    """RoPE fallback must call rotary_emb with Q (queries), not K."""
    calls = []

    class TrackingRotary(nn.Module):
        def forward(self, x, seq_len=None):
            calls.append(x.shape)
            B, H, N, d = x.shape
            cos = torch.ones(B, H, N, d)
            sin = torch.zeros(B, H, N, d)
            return cos, sin

    class MockAttnWithRotary(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
            self.rotary_emb = TrackingRotary()
            self.hidden_size = 64
            self.num_heads = 4

    old = MockAttnWithRotary()
    p = ClusterAttention(d_model=64, num_heads=4, pretrained_attn=old, lora_rank=0)
    x = torch.randn(2, 8, 64)
    p(x, position_embeddings=None)
    # rotary_emb must have been called with Q: shape [2, 4, 8, 16] (seq_len=8, head_dim=16)
    assert len(calls) >= 1, "rotary_emb was never called"
    rotary_L = calls[0][2]   # sequence length dimension of the input passed to rotary_emb
    assert rotary_L == 8, f"rotary_emb called with seq_len={rotary_L}, expected 8 (Q length, not K length)"


def test_cluster_num_kv_heads_from_source_attn():
    """num_kv_heads should be picked up from source_attn when config=None."""
    class GQAAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 32)   # 2 kv_heads * 16
            self.v_proj = nn.Linear(64, 32)
            self.out_proj = nn.Linear(64, 64)
            self.hidden_size = 64
            self.num_heads = 4
            self.num_key_value_heads = 2

    old = GQAAttn()
    p = ClusterAttention(
        config=None, d_model=64, num_heads=4,
        pretrained_attn=old, lora_rank=0,
    )
    assert p.num_kv_heads == 2, f"Expected num_kv_heads=2, got {p.num_kv_heads}"
    assert p.kv_repeat == 2


def test_cluster_head_dim_fallback():
    class DummySourceAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 256
            self.hidden_size = 1024
            self.num_heads = 4
            self.q_proj = nn.Linear(1024, 1024)
            self.k_proj = nn.Linear(1024, 1024)
            self.v_proj = nn.Linear(1024, 1024)
            self.out_proj = nn.Linear(1024, 1024)

    src = DummySourceAttn()
    p = ClusterAttention(config=None, pretrained_attn=src)
    assert p.head_dim == 256
    assert p.all_head_size == 4 * 256


def test_cluster_padding_mask_query_zeroing():
    # 2D attention_mask with zeros at pad tokens
    p_dec = ClusterAttention(d_model=64, num_heads=4, is_decoder=True, is_causal=True)
    x = torch.randn(2, 6, 64)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
    out_dec, _ = p_dec(x, attention_mask=mask)
    # Positions 4, 5 in sample 0 and 3, 4, 5 in sample 1 must be exactly zero
    assert torch.all(out_dec[0, 4:] == 0.0)
    assert torch.all(out_dec[1, 3:] == 0.0)

    p_enc = ClusterAttention(d_model=64, num_heads=4, is_decoder=False, is_causal=False, num_clusters=2)
    out_enc, _ = p_enc(x, attention_mask=mask)
    assert torch.all(out_enc[0, 4:] == 0.0)
    assert torch.all(out_enc[1, 3:] == 0.0)


def test_cluster_third_rope_fallback_signature():
    # Mock rotary that only accepts rotary_mod(Q) without arguments
    class MinimalRotary(nn.Module):
        def forward(self, q):
            return torch.ones_like(q), torch.zeros_like(q)

    p = ClusterAttention(d_model=64, num_heads=4)
    p.rotary_emb = MinimalRotary()
    x = torch.randn(2, 6, 64)
    out, _ = p(x)
    assert out.shape == (2, 6, 64)
    assert torch.isfinite(out).all()
