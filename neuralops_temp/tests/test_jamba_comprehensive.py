from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.jamba import (Jamba2Block, LoRALinear,
                                        apply_rotary_pos_emb, inject_jamba)


def test_jamba_lora_config_independence():
    # 1. config=None, comp_config present
    comp_cfg = SimpleNamespace(lora_rank=8, lora_alpha=16.0)
    p1 = Jamba2Block(config=None, comp_config=comp_cfg, d_model=64, n_heads=4)
    assert p1.lora_rank == 8
    assert p1.lora_alpha == 16.0

    # 2. config=None, comp_config=None, kwargs override
    p2 = Jamba2Block(config=None, comp_config=None, d_model=64, n_heads=4, lora_rank=4, lora_alpha=8.0)
    assert p2.lora_rank == 4
    assert p2.lora_alpha == 8.0

    # 3. config present
    cfg = SimpleNamespace(lora_rank=12, lora_alpha=24.0, num_attention_heads=4)
    p3 = Jamba2Block(config=cfg, d_model=64)
    assert p3.lora_rank == 12
    assert p3.lora_alpha == 24.0

def test_jamba_rotary_fallback():
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
    p = Jamba2Block(d_model=64, n_heads=4, pretrained_attn=old)
    x = torch.randn(2, 8, 64)
    out, _ = p(x, position_embeddings=None)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()

def test_jamba_bidirectional_and_causal_shapes():
    x = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10)
    mask[:, 7:] = 0  # 3 padding tokens

    # Bidirectional (encoder)
    p_bidi = Jamba2Block(d_model=64, n_heads=4, is_causal=False, is_decoder=False, lora_rank=8)
    out_bidi, _ = p_bidi(x, attention_mask=mask)
    assert out_bidi.shape == (2, 10, 64)
    assert torch.isfinite(out_bidi).all()

    # Causal (decoder)
    p_causal = Jamba2Block(d_model=64, n_heads=4, is_causal=True, is_decoder=True, lora_rank=8)
    out_causal, _ = p_causal(x, attention_mask=mask)
    assert out_causal.shape == (2, 10, 64)
    assert torch.isfinite(out_causal).all()

def test_jamba_dimensions_768_and_1024():
    for d in (64, 768, 1024):
        p = Jamba2Block(d_model=d, n_heads=8 if d >= 64 else 4, lora_rank=8)
        x = torch.randn(1, 4, d)
        out, _ = p(x)
        assert out.shape == (1, 4, d)
        assert torch.isfinite(out).all()

def test_jamba_frobenius_norm_verification():
    class ZeroAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
            self.q_proj.weight.data.zero_()  # Zero norm

    old = ZeroAttn()
    with pytest.raises(RuntimeError, match="norm too small"):
        Jamba2Block(d_model=64, n_heads=4, pretrained_attn=old)

def test_jamba_fallback_on_copy_weights_failure():
    class BrokenLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
            self.attention.weight.data.zero_()  # Causes copy_weights to raise
    
    layer = BrokenLayer()
    res = inject_jamba(layer=layer)
    assert res is None  # Graceful fallback, no crash

def test_jamba_lora_adapters_freezing():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
    old = DummyAttn()
    p = Jamba2Block(d_model=64, n_heads=4, pretrained_attn=old, lora_rank=8)
    assert not p.q_lin.weight.requires_grad
    assert p.q_lin.lora_A.requires_grad
    assert p.q_lin.lora_B.requires_grad
    assert not p.out_lin.weight.requires_grad
    assert p.out_lin.lora_A.requires_grad
    assert p.out_lin.lora_B.requires_grad

def test_jamba_injection_all_4_architectures():
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
    res_d = inject_jamba(model=d_model, lora_rank=8)
    assert res_d is not None
    assert isinstance(d_model.distilbert.transformer.layer[0].attention, Jamba2Block)

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
    res_r = inject_jamba(model=r_model, lora_rank=8)
    assert res_r is not None
    assert isinstance(r_model.roberta.encoder.layer[0].attention.self, Jamba2Block)

    # 3. GPT-2 clone
    class GPT2Clone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.attn = MockAttn()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([layer])
    g_model = GPT2Clone()
    res_g = inject_jamba(model=g_model, lora_rank=8)
    assert res_g is not None
    assert isinstance(g_model.transformer.h[0].attn, Jamba2Block)

    # 4. Qwen3.5 clone
    class QwenClone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.self_attn = MockAttn()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([layer])
    q_model = QwenClone()
    res_q = inject_jamba(model=q_model, lora_rank=8)
    assert res_q is not None
    assert isinstance(q_model.model.layers[0].self_attn, Jamba2Block)
