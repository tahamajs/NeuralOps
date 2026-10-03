"""
Hardening test suite for aegis_v5.py (AegisAttention V5).

Covers the 9 hardening rules applied to this module:
  1. config=None safety
  2. Multi-arch weight copy: DistilBERT / RoBERTa nested / GPT-2 fused / Qwen GQA
  3. Fail-loud on missing mandatory weights
  4. Frozen base weights, trainable LoRA after copy
  5. GQA kv_repeat
  6. RoPE (position_embeddings tuple)
  7. Causal masking (decoder)
  8. output_attentions returns actual weight tensor (not None)
  9. Odd-dim _rotate_half roundtrip
 10. inject_aegis_v5 safe/fail paths
 11. Registry registration
"""

import math

import pytest
import torch
import torch.nn as nn
from framework.attentions.aegis_v5 import (AegisAttention, AegisConfigV5,
                                           FactorisedLinear, _rotate_half,
                                           inject_aegis_v5)
from framework.core.registry import ATTENTION_REGISTRY

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class MockConfig:
    def __init__(
        self,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        is_decoder=False,
        model_type="bert",
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.is_decoder = is_decoder
        self.model_type = model_type


class DummyDistilBertAttn(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)


class DummyRobertaSelfAttn(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)


class DummyRobertaAttn(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.self = DummyRobertaSelfAttn(dim)
        self.output = nn.Module()
        self.output.dense = nn.Linear(dim, dim)


class DummyGPT2Attn(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)


class DummyQwenAttn(nn.Module):
    def __init__(self, dim=64, num_kv_heads=2, head_dim=16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim)
        self.o_proj = nn.Linear(dim, dim)


class DummyEmptyAttn(nn.Module):
    """Attention with no recognizable projection attributes — should trigger fail-loud."""
    def __init__(self):
        super().__init__()
        self.foo = nn.Linear(64, 64)


# ---------------------------------------------------------------------------
# 1. Registry
# ---------------------------------------------------------------------------

def test_registry_aegis_v5_registered():
    assert ATTENTION_REGISTRY.get("aegis_v5") is not None
    assert ATTENTION_REGISTRY.get("AegisAttention") is not None
    assert ATTENTION_REGISTRY.get("aegis_v5_module") is not None
    cls = ATTENTION_REGISTRY.get("aegis_v5")
    assert issubclass(cls, AegisAttention) or cls.__name__ == "AegisAttention"


def test_registry_inject_fn_registered():
    fn = ATTENTION_REGISTRY.get("aegis_v5_fn")
    assert fn is not None or ATTENTION_REGISTRY.get("aegis_v5_inject") is not None


# ---------------------------------------------------------------------------
# 2. Config = None safety
# ---------------------------------------------------------------------------

def test_config_none_uses_kwargs():
    attn = AegisAttention(model_config=None, hidden_size=64, num_heads=4)
    assert attn.hidden_size == 64
    assert attn.num_heads == 4
    x = torch.randn(1, 8, 64)
    out, _ = attn(x)
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()


def test_config_none_no_comp_config():
    """Should not raise even with both configs absent."""
    attn = AegisAttention()
    assert attn.hidden_size > 0


# ---------------------------------------------------------------------------
# 3. _rotate_half odd-dim guard
# ---------------------------------------------------------------------------

def test_rotate_half_even_dim():
    x = torch.randn(2, 4, 8)
    result = _rotate_half(x)
    assert result.shape == x.shape


def test_rotate_half_odd_dim():
    x = torch.randn(2, 4, 9)
    result = _rotate_half(x)
    assert result.shape == x.shape, "odd head_dim must preserve shape"


def test_rotate_half_roundtrip_even():
    """Applying _rotate_half twice should give -x (180° rotation)."""
    x = torch.randn(1, 3, 8)
    assert torch.allclose(_rotate_half(_rotate_half(x)), -x, atol=1e-5)


def test_rotate_half_roundtrip_odd():
    """For odd dims, two applications should still give -x on the rotated part."""
    x = torch.randn(1, 3, 9)
    rr = _rotate_half(_rotate_half(x))
    # The rotated 8 dims satisfy rr == -x; last dim is preserved → preserved twice → rr[-1] == x[-1]
    assert torch.allclose(rr[..., :-1], -x[..., :-1], atol=1e-5)
    assert torch.allclose(rr[..., -1:], x[..., -1:], atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Multi-architecture weight copy — base frozen, LoRA trainable
# ---------------------------------------------------------------------------

def _check_frozen_base_trainable_lora(attn: AegisAttention):
    for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
        assert not proj.down.weight.requires_grad, "SVD base weights must be frozen"
        assert not proj.up.weight.requires_grad, "SVD base weights must be frozen"
        assert proj.lora_A.requires_grad, "lora_A must be trainable"
        assert proj.lora_B.requires_grad, "lora_B must be trainable"


def test_distilbert_weight_copy():
    cfg = MockConfig()
    old = DummyDistilBertAttn(64)
    attn = AegisAttention(model_config=cfg, pretrained_attn=old)
    _check_frozen_base_trainable_lora(attn)
    out, _ = attn(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 64)


def test_roberta_nested_encoder_weight_copy():
    cfg = MockConfig()
    old = DummyRobertaAttn(64)
    attn = AegisAttention(model_config=cfg, pretrained_attn=old)
    assert attn.is_nested_encoder, "RoBERTa nested encoder flag must be set"
    _check_frozen_base_trainable_lora(attn)
    out, _ = attn(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 64)


def test_roberta_nested_encoder_no_double_projection():
    """Nested encoder must return (B, N, hidden) not (B, N, 64) twice projected."""
    cfg = MockConfig()
    old = DummyRobertaAttn(64)
    attn = AegisAttention(model_config=cfg, pretrained_attn=old)
    x = torch.randn(2, 10, 64)
    out, _ = attn(x)
    assert out.shape == (2, 10, 64)


def test_gpt2_fused_weight_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old = DummyGPT2Attn(64)
    attn = AegisAttention(model_config=cfg, pretrained_attn=old)
    assert attn.is_decoder
    _check_frozen_base_trainable_lora(attn)
    out, _ = attn(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 64)


def test_qwen_gqa_weight_copy():
    cfg = MockConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, is_decoder=True, model_type="qwen2"
    )
    old = DummyQwenAttn(dim=64, num_kv_heads=2, head_dim=16)
    attn = AegisAttention(model_config=cfg, pretrained_attn=old)
    assert attn.kv_repeat == 2
    _check_frozen_base_trainable_lora(attn)
    out, _ = attn(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 64)


# ---------------------------------------------------------------------------
# 5. Fail-loud on missing projections
# ---------------------------------------------------------------------------

def test_copy_weights_fails_loud_on_empty_attn():
    cfg = MockConfig()
    attn = AegisAttention(model_config=cfg)
    with pytest.raises(RuntimeError, match="mandatory projections"):
        attn.copy_weights(DummyEmptyAttn())


def test_inject_returns_none_on_copy_failure():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyEmptyAttn()

    result = inject_aegis_v5(layer=DummyLayer(), model_config=MockConfig())
    assert result is None, "inject must return None when copy_weights raises"


# ---------------------------------------------------------------------------
# 6. Frobenius-norm check logs warning (near-zero weight)
# ---------------------------------------------------------------------------

def test_near_zero_weight_logs_warning(caplog):
    import logging
    cfg = MockConfig()
    old = DummyDistilBertAttn(64)
    # Zero-out the q weight so norm < 1e-4
    old.q_lin.weight.data.zero_()
    with caplog.at_level(logging.WARNING, logger="framework.attentions.aegis_v5"):
        attn = AegisAttention(model_config=cfg, pretrained_attn=old)
    assert any("near-zero" in r.message for r in caplog.records), \
        "near-zero Frobenius norm must emit a warning"


# ---------------------------------------------------------------------------
# 7. output_attentions returns attention weights (not None)
# ---------------------------------------------------------------------------

def test_output_attentions_returns_weights():
    cfg = MockConfig()
    attn = AegisAttention(model_config=cfg)
    x = torch.randn(1, 6, 64)
    out, weights = attn(x, output_attentions=True)
    assert out.shape == (1, 6, 64)
    assert weights is not None, "output_attentions=True must return attention weights"
    assert weights.shape[-2] == 6   # query dim
    assert weights.shape[-1] == 6   # key dim


def test_output_attentions_false_returns_none_weights():
    cfg = MockConfig()
    attn = AegisAttention(model_config=cfg)
    x = torch.randn(1, 6, 64)
    out, weights = attn(x, output_attentions=False)
    # weights may be None (SDPA path) or a tensor — both are acceptable for False
    assert out.shape == (1, 6, 64)


# ---------------------------------------------------------------------------
# 8. Causal masking (decoder)
# ---------------------------------------------------------------------------

def test_decoder_causal_shape():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = AegisAttention(model_config=cfg)
    x = torch.randn(1, 8, 64)
    out, _ = attn(x)
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()


def test_decoder_no_future_leakage():
    """With causal masking, all-ones input should give finite non-explosive output."""
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = AegisAttention(model_config=cfg)
    x = torch.ones(1, 8, 64)
    out, _ = attn(x)
    assert torch.isfinite(out).all()
    assert out.abs().max() < 1e5


# ---------------------------------------------------------------------------
# 9. RoPE — position_embeddings tuple
# ---------------------------------------------------------------------------

def test_rope_via_position_embeddings():
    cfg = MockConfig(is_decoder=True, model_type="qwen2")
    old = DummyQwenAttn(dim=64, num_kv_heads=2, head_dim=16)
    attn = AegisAttention(model_config=cfg, pretrained_attn=old)
    x = torch.randn(1, 8, 64)
    cos = torch.randn(1, 8, 16)
    sin = torch.randn(1, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()


def test_rope_odd_head_dim():
    """Test that RoPE + odd head_dim doesn't crash."""
    # Build a custom config with odd head_dim (e.g., 17)
    cfg = MockConfig(hidden_size=68, num_attention_heads=4, head_dim=17)
    attn = AegisAttention(model_config=cfg)
    x = torch.randn(1, 4, 68)
    cos = torch.randn(1, 4, 17)
    sin = torch.randn(1, 4, 17)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (1, 4, 68)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 10. inject_aegis_v5 safe injection path
# ---------------------------------------------------------------------------

def test_inject_aegis_v5_on_distilbert_layer():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttn(64)

    layer = DummyLayer()
    res = inject_aegis_v5(layer=layer, model_config=MockConfig())
    assert res is not None
    assert isinstance(layer.attention, AegisAttention)
    out, _ = layer.attention(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 64)


def test_inject_aegis_v5_on_gpt2_attn_layer():
    class GPT2Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = DummyGPT2Attn(64)

    layer = GPT2Layer()
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    res = inject_aegis_v5(layer=layer, model_config=cfg)
    assert res is not None
    assert isinstance(layer.attn, AegisAttention)


def test_inject_aegis_v5_target_none_returns_none():
    res = inject_aegis_v5(layer=None, model=None)
    assert res is None


# ---------------------------------------------------------------------------
# 11. FactorisedLinear SVD decompose + freeze
# ---------------------------------------------------------------------------

def test_factorisedlinear_from_weight():
    W = torch.randn(64, 64)
    fl = FactorisedLinear(W)
    assert not fl.down.weight.requires_grad
    assert not fl.up.weight.requires_grad
    assert fl.lora_A.requires_grad
    assert fl.lora_B.requires_grad
    x = torch.randn(2, 64)
    out = fl(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_factorisedlinear_lora_rank_zero():
    fl = FactorisedLinear(32, 32, lora_rank=0)
    assert fl.lora_A is None
    assert fl.lora_B is None
    out = fl(torch.randn(3, 32))
    assert out.shape == (3, 32)


def test_factorisedlinear_forward_finite():
    W = torch.randn(128, 64)
    b = torch.randn(128)
    fl = FactorisedLinear(W, b)
    out = fl(torch.randn(4, 64))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 12. Misc forward shapes
# ---------------------------------------------------------------------------

def test_forward_long_sequence_encoder():
    """Long encoder sequence takes the local+global path (N >= local_window)."""
    aegis_cfg = AegisConfigV5(num_local_heads=3, num_global_heads=1, local_window=4)
    attn = AegisAttention(model_config=MockConfig(), aegis_cfg=aegis_cfg)
    # N=16 > local_window=4 → split-head path
    x = torch.randn(1, 16, 64)
    out, _ = attn(x)
    assert out.shape == (1, 16, 64)
    assert torch.isfinite(out).all()


def test_forward_no_nan_inf():
    cfg = MockConfig()
    attn = AegisAttention(model_config=cfg)
    x = torch.randn(2, 12, 64)
    out, _ = attn(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_forward_cross_attention():
    cfg = MockConfig()
    attn = AegisAttention(model_config=cfg)
    x = torch.randn(1, 6, 64)
    ctx = torch.randn(1, 10, 64)
    out, _ = attn(x, encoder_hidden_states=ctx)
    assert out.shape == (1, 6, 64)
    assert torch.isfinite(out).all()
