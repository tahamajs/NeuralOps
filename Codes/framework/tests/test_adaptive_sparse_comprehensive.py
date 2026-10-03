from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from framework.attentions.adaptive_sparse_attention import (
    AdaptiveSparseAttention, LoRALinear, inject_adaptive_sparse)


def test_adaptive_sparse_lora_config_independence():
    # 1. config=None, comp_config present
    comp_cfg = SimpleNamespace(lora_rank=8, lora_alpha=16.0)
    p1 = AdaptiveSparseAttention(config=None, comp_config=comp_cfg, d_model=64, num_heads=4)
    assert p1.lora_rank == 8
    assert p1.lora_alpha == 16.0

    # 2. config=None, comp_config=None, kwargs override
    p2 = AdaptiveSparseAttention(config=None, comp_config=None, d_model=64, num_heads=4, lora_rank=4, lora_alpha=8.0)
    assert p2.lora_rank == 4
    assert p2.lora_alpha == 8.0

    # 3. config present
    cfg = SimpleNamespace(lora_rank=12, lora_alpha=24.0, num_attention_heads=4, hidden_size=64)
    p3 = AdaptiveSparseAttention(config=cfg)
    assert p3.lora_rank == 12
    assert p3.lora_alpha == 24.0

def test_adaptive_sparse_rotary_fallback():
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
    p = AdaptiveSparseAttention(d_model=64, num_heads=4, pretrained_attn=old)
    x = torch.randn(2, 8, 64)
    out, _ = p(x, position_embeddings=None)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()

def test_adaptive_sparse_bidirectional_and_causal_shapes():
    x = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10)
    mask[:, 7:] = 0  # 3 padding tokens

    # Bidirectional (encoder)
    p_bidi = AdaptiveSparseAttention(d_model=64, num_heads=4, is_causal=False, is_decoder=False, lora_rank=8)
    out_bidi, _ = p_bidi(x, attention_mask=mask)
    assert out_bidi.shape == (2, 10, 64)
    assert torch.isfinite(out_bidi).all()

    # Causal (decoder)
    p_causal = AdaptiveSparseAttention(d_model=64, num_heads=4, is_causal=True, is_decoder=True, lora_rank=8)
    out_causal, _ = p_causal(x, attention_mask=mask)
    assert out_causal.shape == (2, 10, 64)
    assert torch.isfinite(out_causal).all()

def test_adaptive_sparse_topk_masking():
    p = AdaptiveSparseAttention(d_model=64, num_heads=4, sparsity_ratio=0.5)
    p.eval()
    x = torch.randn(2, 12, 64)
    out, attn = p(x, output_attentions=True)
    assert out.shape == (2, 12, 64)
    assert torch.isfinite(out).all()
    if attn is not None:
        # Check that approximately 50% elements in each row are heavily suppressed
        suppressed_frac = (attn < 0.05).float().mean().item()
        assert suppressed_frac > 0.3
        # Check that attention distribution sums to 1 across keys
        assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5)

def test_adaptive_sparse_frobenius_norm_verification():
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
        AdaptiveSparseAttention(d_model=64, num_heads=4, pretrained_attn=old)

def test_adaptive_sparse_fallback_on_copy_weights_failure():
    class BrokenLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
            self.attention.weight.data.zero_()  # Causes copy_weights to raise
    
    layer = BrokenLayer()
    res = inject_adaptive_sparse(layer=layer)
    assert res is None  # Graceful fallback, no crash

def test_adaptive_sparse_lora_adapters_freezing():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
    old = DummyAttn()
    p = AdaptiveSparseAttention(d_model=64, num_heads=4, pretrained_attn=old, lora_rank=8)
    assert not p.q_proj.weight.requires_grad
    assert p.q_proj.lora_A.requires_grad
    assert p.q_proj.lora_B.requires_grad
    assert not p.out_proj.weight.requires_grad
    assert p.out_proj.lora_A.requires_grad
    assert p.out_proj.lora_B.requires_grad

def test_adaptive_sparse_injection_all_4_architectures():
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
    res_d = inject_adaptive_sparse(model=d_model, lora_rank=8)
    assert res_d is not None
    assert isinstance(d_model.distilbert.transformer.layer[0].attention, AdaptiveSparseAttention)

    # 2. RoBERTa clone
    class RobertaClone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.attention = nn.Module()
            layer.attention.self = MockAttn()
            self.roberta = nn.Module()
            self.roberta.encoder = nn.Module()
            self.roberta.encoder.layer = nn.ModuleList([layer])
    r_model = RobertaClone()
    res_r = inject_adaptive_sparse(model=r_model, lora_rank=8)
    assert res_r is not None
    assert isinstance(r_model.roberta.encoder.layer[0].attention.self, AdaptiveSparseAttention)

    # 3. GPT-2 clone
    class GPT2Clone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.attn = MockAttn()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([layer])
    g_model = GPT2Clone()
    res_g = inject_adaptive_sparse(model=g_model, lora_rank=8)
    assert res_g is not None
    assert isinstance(g_model.transformer.h[0].attn, AdaptiveSparseAttention)

    # 4. Qwen3.5 clone
    class QwenClone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.self_attn = MockAttn()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([layer])
    q_model = QwenClone()
    res_q = inject_adaptive_sparse(model=q_model, lora_rank=8)
    assert res_q is not None
    assert isinstance(q_model.model.layers[0].self_attn, AdaptiveSparseAttention)

def test_adaptive_sparse_differentiability():
    p = AdaptiveSparseAttention(d_model=64, num_heads=4, sparsity_ratio=0.5, lora_rank=8)
    x = torch.randn(2, 12, 64, requires_grad=True)
    out, _ = p(x)
    loss = out.sum() + p.get_sparsity_loss()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad != 0).any()
    assert p.sparsity_logits.grad is not None
    assert torch.isfinite(p.sparsity_logits.grad).all()

def test_adaptive_sparse_cross_attention_bypass():
    p = AdaptiveSparseAttention(d_model=64, num_heads=4, sparsity_ratio=0.8)
    p.eval()
    x = torch.randn(2, 8, 64)
    kv = torch.randn(2, 16, 64)
    out, attn = p(x, encoder_hidden_states=kv, output_attentions=True)
    assert out.shape == (2, 8, 64)
    assert attn is not None
    assert (attn > 0.0).all()

def test_adaptive_sparse_nested_encoder_outer_dense_wrap():
    class MockDense(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(64, 64))
            self.bias = nn.Parameter(torch.zeros(64))
            self.in_features = 64
            self.out_features = 64
        def forward(self, x):
            return x

    class MockSelf(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Module()
            self.attention.self = MockSelf()
            self.attention.output = nn.Module()
            self.attention.output.dense = MockDense()

    lyr = MockLayer()
    res = inject_adaptive_sparse(layer=lyr, lora_rank=8)
    assert res is not None
    assert isinstance(lyr.attention.output.dense, LoRALinear)
    assert lyr.attention.output.dense.lora_A is not None
    assert lyr.attention.output.dense.lora_A.requires_grad

def test_lora_linear_no_output_scale_double_application():
    layer = LoRALinear(64, 64, lora_rank=8)
    assert not hasattr(layer, "output_scale")
    x = torch.randn(2, 64)
    out = layer(x)
    assert out.shape == (2, 64)
