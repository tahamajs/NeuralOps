from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.performer_attention import (LoRALinear,
                                                      PerformerAttention,
                                                      apply_rotary_pos_emb,
                                                      inject_performer)


def test_performer_lora_config_independence():
    # 1. config=None, comp_config present
    comp_cfg = SimpleNamespace(lora_rank=8, lora_alpha=16.0, performer_rank=32)
    p1 = PerformerAttention(config=None, comp_config=comp_cfg, dim=64, num_heads=4)
    assert p1.lora_rank == 8
    assert p1.lora_alpha == 16.0
    assert p1.num_features == 32

    # 2. config=None, comp_config=None, kwargs override
    p2 = PerformerAttention(config=None, comp_config=None, dim=64, num_heads=4, lora_rank=4, lora_alpha=8.0, rank=16)
    assert p2.lora_rank == 4
    assert p2.lora_alpha == 8.0
    assert p2.num_features == 16

def test_performer_rotary_fallback():
    class MockRotaryEmb(nn.Module):
        def forward(self, x, pos_ids):
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
    p = PerformerAttention(dim=64, num_heads=4, pretrained_attn=old)
    x = torch.randn(2, 8, 64)
    out, _ = p(x, position_embeddings=None)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()

def test_performer_bidirectional_and_causal_shapes():
    x = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10)
    mask[:, 7:] = 0  # 3 padding tokens

    # Bidirectional
    p_bidi = PerformerAttention(dim=64, num_heads=4, is_decoder=False, lora_rank=8)
    out_bidi, _ = p_bidi(x, attention_mask=mask)
    assert out_bidi.shape == (2, 10, 64)
    assert torch.isfinite(out_bidi).all()

    # Causal
    p_causal = PerformerAttention(dim=64, num_heads=4, is_decoder=True, lora_rank=8)
    out_causal, _ = p_causal(x, attention_mask=mask)
    assert out_causal.shape == (2, 10, 64)
    assert torch.isfinite(out_causal).all()

def test_performer_frobenius_norm_verification():
    class ZeroAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
            self.q_proj.weight.data.zero_()  # Zero norm

    old = ZeroAttn()
    # Does not crash with RuntimeError, logs warning instead
    p = PerformerAttention(dim=64, num_heads=4, pretrained_attn=old)
    assert p is not None

def test_performer_fallback_on_copy_weights_failure():
    class BrokenLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
            self.attention.weight.data.zero_()  # Causes copy_weights to raise
    
    layer = BrokenLayer()
    res = inject_performer(layer=layer)
    assert res is None  # Graceful fallback, no crash

def test_performer_lora_adapters_freezing():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
    old = DummyAttn()
    p = PerformerAttention(dim=64, num_heads=4, pretrained_attn=old, lora_rank=8)
    assert not p.query.weight.requires_grad
    assert p.query.lora_A.requires_grad
    assert p.query.lora_B.requires_grad
    assert not p.out_proj.weight.requires_grad
    assert p.out_proj.lora_A.requires_grad
    assert p.out_proj.lora_B.requires_grad

def test_performer_injection_all_4_architectures():
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
    res_d = inject_performer(model=d_model, lora_rank=8)
    assert res_d is not None
    assert isinstance(d_model.distilbert.transformer.layer[0].attention, PerformerAttention)

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
    res_r = inject_performer(model=r_model, lora_rank=8)
    assert res_r is not None
    assert isinstance(r_model.roberta.encoder.layer[0].attention.self, PerformerAttention)

    # 3. GPT-2 clone
    class GPT2Clone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.attn = MockAttn()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([layer])
    g_model = GPT2Clone()
    res_g = inject_performer(model=g_model, lora_rank=8)
    assert res_g is not None
    assert isinstance(g_model.transformer.h[0].attn, PerformerAttention)

    # 4. Qwen3.5 clone
    class QwenClone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.self_attn = MockAttn()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([layer])
    q_model = QwenClone()
    res_q = inject_performer(model=q_model, lora_rank=8)
    assert res_q is not None
    assert isinstance(q_model.model.layers[0].self_attn, PerformerAttention)
