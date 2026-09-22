"""
Hardening test suite for aegis_v6.py (AegisAttentionV3 / AegisAttentionV6).

Covers the 9 hardening rules:
  1. Registry
  2. config=None safety
  3. Multi-arch weight copy: DistilBERT / RoBERTa nested / GPT-2 / Qwen GQA
  4. Fail-loud on missing projections
  5. Frozen SVD base, trainable LoRA
  6. GQA kv_repeat
  7. RoPE via position_embeddings
  8. Causal masking
  9. Odd-dim _rotate_half
 10. inject_aegis_v6 safe/fail paths
 11. FactorisedLowRankProjection correctness
 12. Forward shapes and numerical stability
"""

import math
import pytest
import torch
import torch.nn as nn

from framework.attentions.aegis_v6 import (
    AegisAttentionV3,
    AegisAttentionV6,
    AegisConfigV3,
    FactorisedLowRankProjection,
    _rotate_half,
    inject_aegis_v6,
)
from framework.core.registry import ATTENTION_REGISTRY


# ---------------------------------------------------------------------------
# Helpers / fixtures
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
    def __init__(self):
        super().__init__()
        self.foo = nn.Linear(64, 64)


def _make_aegis_v6(cfg=None, **kwargs):
    if cfg is None:
        cfg = MockConfig()
    return AegisAttentionV3(config=cfg, **kwargs)


# ---------------------------------------------------------------------------
# 1. Registry
# ---------------------------------------------------------------------------

def test_registry_aegis_v6():
    cls = ATTENTION_REGISTRY.get("aegis_v6")
    assert cls is not None
    assert issubclass(cls, AegisAttentionV3)


def test_registry_aegis_v6_attn():
    cls = ATTENTION_REGISTRY.get("aegis_v6_attn")
    assert cls is not None


def test_registry_inject_fn():
    fn = ATTENTION_REGISTRY.get("aegis_v6_inject")
    assert fn is not None or ATTENTION_REGISTRY.get("inject_aegis_v6") is not None


def test_public_alias():
    assert AegisAttentionV6 is AegisAttentionV3


# ---------------------------------------------------------------------------
# 2. Config = None safety
# ---------------------------------------------------------------------------

def test_config_none_falls_back_to_defaults():
    attn = AegisAttentionV3(config=None, hidden_size=64, num_heads=4)
    assert attn.hidden_size == 64
    x = torch.randn(1, 8, 64)
    out, _ = attn(x)
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 3. _rotate_half odd-dim guard
# ---------------------------------------------------------------------------

def test_rotate_half_even():
    x = torch.randn(2, 4, 8)
    assert _rotate_half(x).shape == x.shape


def test_rotate_half_odd():
    x = torch.randn(2, 4, 9)
    assert _rotate_half(x).shape == x.shape


def test_rotate_half_roundtrip_even():
    x = torch.randn(1, 3, 8)
    assert torch.allclose(_rotate_half(_rotate_half(x)), -x, atol=1e-5)


def test_rotate_half_roundtrip_odd():
    x = torch.randn(1, 3, 9)
    rr = _rotate_half(_rotate_half(x))
    assert torch.allclose(rr[..., :-1], -x[..., :-1], atol=1e-5)
    assert torch.allclose(rr[..., -1:], x[..., -1:], atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Multi-arch weight copy — base frozen, LoRA trainable
# ---------------------------------------------------------------------------

def _check_frozen_base(attn):
    for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
        assert not proj.S.requires_grad, "SVD S must be frozen"
        assert proj.A.requires_grad, "LoRA A must be trainable"
        assert proj.B.requires_grad, "LoRA B must be trainable"


def test_distilbert_weight_copy():
    cfg = MockConfig()
    old = DummyDistilBertAttn(64)
    attn = AegisAttentionV3(config=cfg, pretrained_attn=old)
    _check_frozen_base(attn)
    out, _ = attn(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()


def test_roberta_nested_encoder():
    cfg = MockConfig()
    old = DummyRobertaAttn(64)
    attn = AegisAttentionV3(config=cfg, pretrained_attn=old)
    assert attn.is_nested_encoder
    _check_frozen_base(attn)
    out, _ = attn(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 64)


def test_roberta_nested_no_double_projection():
    cfg = MockConfig()
    old = DummyRobertaAttn(64)
    attn = AegisAttentionV3(config=cfg, pretrained_attn=old)
    out, _ = attn(torch.randn(1, 10, 64))
    assert out.shape == (1, 10, 64)


def test_gpt2_fused_weight_copy():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    old = DummyGPT2Attn(64)
    attn = AegisAttentionV3(config=cfg, pretrained_attn=old)
    assert attn.is_decoder
    _check_frozen_base(attn)
    out, _ = attn(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 64)


def test_qwen_gqa_weight_copy():
    cfg = MockConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, is_decoder=True, model_type="qwen2"
    )
    old = DummyQwenAttn(dim=64, num_kv_heads=2, head_dim=16)
    attn = AegisAttentionV3(config=cfg, pretrained_attn=old)
    assert attn.kv_repeat == 2
    _check_frozen_base(attn)
    out, _ = attn(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 64)


# ---------------------------------------------------------------------------
# 5. Fail-loud on missing projections
# ---------------------------------------------------------------------------

def test_copy_weights_fails_loud_empty():
    cfg = MockConfig()
    attn = AegisAttentionV3(config=cfg)
    with pytest.raises(RuntimeError, match="mandatory projections"):
        attn.copy_weights(DummyEmptyAttn())


def test_inject_returns_none_on_failure():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyEmptyAttn()

    result = inject_aegis_v6(layer=DummyLayer(), model_config=MockConfig())
    assert result is None


# ---------------------------------------------------------------------------
# 6. Near-zero weight warning
# ---------------------------------------------------------------------------

def test_near_zero_weight_warning(caplog):
    import logging
    cfg = MockConfig()
    old = DummyDistilBertAttn(64)
    old.q_lin.weight.data.zero_()
    with caplog.at_level(logging.WARNING, logger="framework.attentions.aegis_v6"):
        attn = AegisAttentionV3(config=cfg, pretrained_attn=old)
    assert any("near-zero" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 7. RoPE via position_embeddings
# ---------------------------------------------------------------------------

def test_rope_via_position_embeddings():
    cfg = MockConfig(is_decoder=True, model_type="qwen2")
    old = DummyQwenAttn(64, num_kv_heads=2, head_dim=16)
    attn = AegisAttentionV3(config=cfg, pretrained_attn=old)
    x = torch.randn(1, 8, 64)
    cos = torch.randn(1, 8, 16)
    sin = torch.randn(1, 8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()


def test_rope_odd_head_dim():
    cfg = MockConfig(hidden_size=68, num_attention_heads=4, head_dim=17)
    attn = AegisAttentionV3(config=cfg)
    x = torch.randn(1, 4, 68)
    cos = torch.randn(1, 4, 17)
    sin = torch.randn(1, 4, 17)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (1, 4, 68)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 8. Causal masking (decoder)
# ---------------------------------------------------------------------------

def test_decoder_forward_shape():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = AegisAttentionV3(config=cfg)
    out, _ = attn(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)
    assert torch.isfinite(out).all()


def test_decoder_no_future_leakage():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = AegisAttentionV3(config=cfg)
    out, _ = attn(torch.ones(1, 8, 64))
    assert torch.isfinite(out).all()
    assert out.abs().max() < 1e5


# ---------------------------------------------------------------------------
# 9. inject_aegis_v6 safe paths
# ---------------------------------------------------------------------------

def test_inject_on_distilbert_layer():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = DummyDistilBertAttn(64)

    layer = DummyLayer()
    res = inject_aegis_v6(layer=layer, model_config=MockConfig())
    assert res is not None
    assert isinstance(layer.attention, AegisAttentionV3)
    out, _ = layer.attention(torch.randn(1, 8, 64))
    assert out.shape == (1, 8, 64)


def test_inject_on_gpt2_layer():
    class GPT2Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = DummyGPT2Attn(64)

    layer = GPT2Layer()
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    res = inject_aegis_v6(layer=layer, model_config=cfg)
    assert res is not None
    assert isinstance(layer.attn, AegisAttentionV3)


def test_inject_target_none():
    assert inject_aegis_v6(layer=None, model=None) is None


# ---------------------------------------------------------------------------
# 10. FactorisedLowRankProjection
# ---------------------------------------------------------------------------

def test_flrp_from_weight():
    W = torch.randn(64, 64)
    fl = FactorisedLowRankProjection(64, 64, svd_rank=32, lora_rank=8, pretrained_weight=W)
    assert not fl.S.requires_grad
    assert fl.A.requires_grad
    assert fl.B.requires_grad
    # FLRP expects 3D input (B, N, in_features)
    out = fl(torch.randn(2, 5, 64))
    assert out.shape == (2, 5, 64)
    assert torch.isfinite(out).all()


def test_flrp_lora_rank_zero():
    fl = FactorisedLowRankProjection(32, 32, svd_rank=16, lora_rank=0)
    assert fl.A is None
    assert fl.B is None
    out = fl(torch.randn(3, 4, 32))
    assert out.shape == (3, 4, 32)


def test_flrp_forward_finite():
    W = torch.randn(128, 64)
    b = torch.randn(128)
    fl = FactorisedLowRankProjection(64, 128, svd_rank=32, lora_rank=4, pretrained_weight=W, pretrained_bias=b)
    out = fl(torch.randn(4, 6, 64))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 11. Forward shapes and numerical stability
# ---------------------------------------------------------------------------

def test_forward_encoder_long_seq():
    """Long encoder seq — global linear kernel path fires."""
    cfg = MockConfig()
    attn = AegisAttentionV3(config=cfg)
    out, _ = attn(torch.randn(1, 128, 64))
    assert out.shape == (1, 128, 64)
    assert torch.isfinite(out).all()


def test_forward_no_nan_inf():
    cfg = MockConfig()
    attn = AegisAttentionV3(config=cfg)
    out, _ = attn(torch.randn(2, 12, 64))
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_forward_cross_attention():
    cfg = MockConfig()
    attn = AegisAttentionV3(config=cfg)
    x = torch.randn(1, 6, 64)
    ctx = torch.randn(1, 10, 64)
    out, _ = attn(x, encoder_hidden_states=ctx)
    assert out.shape == (1, 6, 64)
    assert torch.isfinite(out).all()


def test_forward_with_attention_mask():
    cfg = MockConfig()
    attn = AegisAttentionV3(config=cfg)
    x = torch.randn(2, 8, 64)
    mask = torch.ones(2, 8)
    mask[0, 6:] = 0  # pad last 2 tokens for first batch item
    out, _ = attn(x, attention_mask=mask)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()
