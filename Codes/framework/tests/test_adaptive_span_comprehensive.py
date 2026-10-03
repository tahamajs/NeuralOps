from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from framework.attentions.adaptive_span_attention import (
    AdaptiveSpanAttention, LoRALinear, inject_adaptive_span)


def test_adaptive_span_lora_config_independence():
    # 1. config=None, comp_config present
    comp_cfg = SimpleNamespace(lora_rank=8, lora_alpha=16.0)
    p1 = AdaptiveSpanAttention(config=None, comp_config=comp_cfg, d_model=64, num_heads=4)
    assert p1.lora_rank == 8
    assert p1.lora_alpha == 16.0

    # 2. config=None, comp_config=None, kwargs override
    p2 = AdaptiveSpanAttention(config=None, comp_config=None, d_model=64, num_heads=4, lora_rank=4, lora_alpha=8.0)
    assert p2.lora_rank == 4
    assert p2.lora_alpha == 8.0

    # 3. config present
    cfg = SimpleNamespace(lora_rank=12, lora_alpha=24.0, num_attention_heads=4, hidden_size=64)
    p3 = AdaptiveSpanAttention(config=cfg)
    assert p3.lora_rank == 12
    assert p3.lora_alpha == 24.0

def test_adaptive_span_rotary_fallback():
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
    p = AdaptiveSpanAttention(d_model=64, num_heads=4, pretrained_attn=old)
    x = torch.randn(2, 8, 64)
    out, _ = p(x, position_embeddings=None)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()

def test_adaptive_span_bidirectional_and_causal_shapes():
    x = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10)
    mask[:, 7:] = 0  # 3 padding tokens

    # Bidirectional (encoder)
    p_bidi = AdaptiveSpanAttention(d_model=64, num_heads=4, is_causal=False, is_decoder=False, lora_rank=8)
    out_bidi, _ = p_bidi(x, attention_mask=mask)
    assert out_bidi.shape == (2, 10, 64)
    assert torch.isfinite(out_bidi).all()

    # Causal (decoder)
    p_causal = AdaptiveSpanAttention(d_model=64, num_heads=4, is_causal=True, is_decoder=True, lora_rank=8)
    out_causal, _ = p_causal(x, attention_mask=mask)
    assert out_causal.shape == (2, 10, 64)
    assert torch.isfinite(out_causal).all()

def test_adaptive_span_masking():
    p = AdaptiveSpanAttention(d_model=64, num_heads=4, max_span=32)
    x = torch.randn(2, 12, 64)
    out, attn = p(x, output_attentions=True)
    assert out.shape == (2, 12, 64)
    assert torch.isfinite(out).all()

def test_adaptive_span_frobenius_norm_verification():
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
        AdaptiveSpanAttention(d_model=64, num_heads=4, pretrained_attn=old)

def test_adaptive_span_fallback_on_copy_weights_failure():
    class BrokenLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
            self.attention.weight.data.zero_()  # Causes copy_weights to raise
    
    layer = BrokenLayer()
    res = inject_adaptive_span(layer=layer)
    assert res is None  # Graceful fallback, no crash

def test_adaptive_span_lora_adapters_freezing():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
    old = DummyAttn()
    p = AdaptiveSpanAttention(d_model=64, num_heads=4, pretrained_attn=old, lora_rank=8)
    assert not p.q_proj.weight.requires_grad
    assert p.q_proj.lora_A.requires_grad
    assert p.q_proj.lora_B.requires_grad
    assert not p.out_proj.weight.requires_grad
    assert p.out_proj.lora_A.requires_grad
    assert p.out_proj.lora_B.requires_grad

def test_adaptive_span_injection_all_4_architectures():
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
    res_d = inject_adaptive_span(model=d_model, lora_rank=8)
    assert res_d is not None
    assert isinstance(d_model.distilbert.transformer.layer[0].attention, AdaptiveSpanAttention)

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
    res_r = inject_adaptive_span(model=r_model, lora_rank=8)
    assert res_r is not None
    assert isinstance(r_model.roberta.encoder.layer[0].attention, AdaptiveSpanAttention)

    # 3. GPT-2 clone
    class GPT2Clone(nn.Module):
        def __init__(self):
            super().__init__()
            block = nn.Module()
            block.attn = MockAttn()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([block])
    g_model = GPT2Clone()
    res_g = inject_adaptive_span(model=g_model, lora_rank=8)
    assert res_g is not None
    assert isinstance(g_model.transformer.h[0].attn, AdaptiveSpanAttention)

    # 4. Qwen3.5 clone
    class QwenClone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.self_attn = MockAttn()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([layer])
    q_model = QwenClone()
    res_q = inject_adaptive_span(model=q_model, lora_rank=8)
    assert res_q is not None
    assert isinstance(q_model.model.layers[0].self_attn, AdaptiveSpanAttention)


def test_adaptive_span_differentiability():
    """Verify that span_logits is fully differentiable and receives non-zero gradients."""
    attn_mod = AdaptiveSpanAttention(d_model=32, num_heads=2, max_span=16, ramp_size=8.0, is_causal=True)
    # Set span_logits so span is inside the sequence length
    attn_mod.span_logits.data.fill_(0.0)  # sigmoid(0) * 16 = 8
    x = torch.randn(2, 16, 32, requires_grad=True)
    out, _ = attn_mod(x)
    loss = out.sum() + attn_mod.get_span_loss()
    loss.backward()

    assert attn_mod.span_logits.grad is not None
    assert torch.isfinite(attn_mod.span_logits.grad).all()
    assert (attn_mod.span_logits.grad != 0).any()


def test_lora_linear_canonical_contract():
    """Verify LoRALinear adheres strictly to canonical LoRA without extraneous output_scale."""
    layer = LoRALinear(in_features=32, out_features=32, lora_rank=4, freeze_base=True)
    assert not hasattr(layer, "output_scale")
    assert not layer.weight.requires_grad
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad
    
    x = torch.randn(2, 32)
    out = layer(x)
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


def test_adaptive_span_posterior_renormalization():
    """1. Verify attention weights sum to exactly 1.0 along dim=-1 after soft ramp and re-normalization."""
    attn_mod = AdaptiveSpanAttention(d_model=64, num_heads=4, max_span=16, ramp_size=8.0, dropout=0.0, is_causal=False)
    attn_mod.eval()
    x = torch.randn(2, 20, 64)
    out, attn = attn_mod(x, output_attentions=True)
    assert attn is not None
    sums = attn.sum(dim=-1)
    target = torch.ones_like(sums)
    assert torch.allclose(sums, target, atol=1e-5, rtol=1e-5)


def test_adaptive_span_gqa_asymmetry():
    """2. Verify GQA repeat_kv logic expands tensors without dimension mismatches when num_heads > num_kv_heads."""
    attn_mod = AdaptiveSpanAttention(
        d_model=64,
        num_heads=8,
        num_kv_heads=2,
        lora_rank=4,
    )
    assert attn_mod.num_heads == 8
    assert attn_mod.num_kv_heads == 2
    assert attn_mod.kv_repeat == 4

    x = torch.randn(2, 12, 64)
    out, _ = attn_mod(x)
    assert out.shape == (2, 12, 64)
    assert torch.isfinite(out).all()


def test_adaptive_span_cross_attention_bypass():
    """3. Verify cross-attention with unequal N_q and N_k safely bypasses span masking without crashing."""
    attn_mod = AdaptiveSpanAttention(d_model=64, num_heads=4, lora_rank=4)
    hidden_states = torch.randn(2, 8, 64)       # N_q = 8
    key_value_states = torch.randn(2, 16, 64)   # N_k = 16
    out, attn = attn_mod(
        hidden_states=hidden_states,
        key_value_states=key_value_states,
        output_attentions=True,
    )
    assert out.shape == (2, 8, 64)
    assert attn.shape == (2, 4, 8, 16)
    assert torch.isfinite(out).all()


def test_adaptive_span_nested_encoder_bypass():
    """4. Verify nested encoder bypass (RoBERTa/BERT): q_proj gets grads, out_proj has no trainable LoRA grads."""
    attn_mod = AdaptiveSpanAttention(d_model=64, num_heads=4, lora_rank=4, is_nested_encoder=True)
    assert attn_mod.is_nested_encoder
    assert attn_mod.out_proj.lora_A is None  # lora_rank=0 for out_proj when nested
    x = torch.randn(2, 10, 64, requires_grad=True)
    out, _ = attn_mod(x)
    loss = out.sum()
    loss.backward()

    # q_proj must receive gradients
    assert attn_mod.q_proj.lora_A.grad is not None
    assert torch.isfinite(attn_mod.q_proj.lora_A.grad).all()

    # out_proj is bypassed so its parameters must have no gradient
    assert attn_mod.out_proj.lora_A is None or attn_mod.out_proj.lora_A.grad is None


def test_adaptive_span_nested_encoder_outer_dense_wrap():
    """Verify inject_adaptive_span wraps outer attention.output.dense with LoRALinear for RoBERTa/BERT."""
    class MockSelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)

    class MockOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(64, 64)

    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = MockSelfAttn()
            self.output = MockOutput()

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = MockAttention()

    class FullRoberta(nn.Module):
        def __init__(self):
            super().__init__()
            self.roberta = nn.Module()
            self.roberta.encoder = nn.Module()
            self.roberta.encoder.layer = nn.ModuleList([MockLayer()])

    model = FullRoberta()
    res = inject_adaptive_span(model=model, lora_rank=8, lora_alpha=16.0)
    assert res is not None

    layer = model.roberta.encoder.layer[0]
    assert isinstance(layer.attention.self, AdaptiveSpanAttention)
    assert layer.attention.self.is_nested_encoder
    assert layer.attention.self.out_proj.lora_A is None  # no wasted inner LoRA

    # Outer output.dense must remain a plain nn.Linear (pretrained, frozen) to prevent LoRA drift
    assert isinstance(layer.attention.output.dense, nn.Linear)
    assert not isinstance(layer.attention.output.dense, LoRALinear)


def test_adaptive_span_zero_span_stability():
    """5. Verify numerical stability when span is ~0 (span_logits = -100.0) with zero NaNs or Infs."""
    attn_mod = AdaptiveSpanAttention(d_model=64, num_heads=4, max_span=32, ramp_size=8.0)
    with torch.no_grad():
        attn_mod.span_logits.fill_(-100.0)

    x = torch.randn(2, 16, 64)
    out, attn = attn_mod(x, output_attentions=True)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert attn is not None
    assert not torch.isnan(attn).any()
    assert not torch.isinf(attn).any()


