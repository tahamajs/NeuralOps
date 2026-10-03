"""Tests for CHELA-SLAY Attention Module."""

from .testing_utils import MockConfig, DummyDistilBertAttention
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.chela_slay import (ChelaSlayAttention, LoRALinear,
                                             _rotate_half, inject_chela_slay)
from neuralops.core.registry import ATTENTION_REGISTRY

def test_registry():
    assert "chela_slay" in ATTENTION_REGISTRY.list_available()
    assert "chela_slay_attention" in ATTENTION_REGISTRY.list_available()
    assert ATTENTION_REGISTRY.get("chela_slay") is ChelaSlayAttention

def test_odd_dim_rotate_half():
    x = torch.randn(2, 4, 15)
    rotated = _rotate_half(x)
    assert rotated.shape == x.shape
    assert torch.equal(rotated[..., -1], x[..., -1])

def test_trainable_yat_proj_and_lora_gate():
    cfg = MockConfig()
    attn = ChelaSlayAttention(config=cfg, feature_dim=32)
    assert isinstance(attn.yat_proj, nn.Parameter)
    assert attn.yat_proj.requires_grad
    assert isinstance(attn.lora_gate, nn.Linear)
    # gate alias property
    assert attn.gate is attn.lora_gate
    assert attn.random_proj is attn.yat_proj

def test_magnitude_preserving_spherical_norm():
    cfg = MockConfig()
    attn = ChelaSlayAttention(config=cfg, feature_dim=32)
    x = torch.randn(2, 4, 8, 16) * 10.0
    norm_x = attn._spherical_normalize(x)
    assert norm_x.shape == x.shape
    # check that it scales with log1p(norm)
    norms = torch.norm(norm_x, p=2, dim=-1)
    orig_norms = torch.norm(x, p=2, dim=-1)
    expected_norms = torch.log1p(orig_norms)
    assert torch.allclose(norms, expected_norms, atol=1e-4)

def test_chunked_causal_cumsum_equivalence():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = ChelaSlayAttention(config=cfg, feature_dim=32)
    attn.eval()
    # Sequence length 128 > chunk_size 64
    x = torch.randn(2, 128, 64)
    with torch.no_grad():
        out, _ = attn(x)
    assert out.shape == (2, 128, 64)

def test_bidirectional_linear_attention():
    cfg = MockConfig(is_decoder=False, model_type="bert")
    attn = ChelaSlayAttention(config=cfg, feature_dim=32)
    x = torch.randn(2, 32, 64)
    out, _ = attn(x)
    assert out.shape == (2, 32, 64)

def test_gqa_expansion():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    attn = ChelaSlayAttention(config=cfg, feature_dim=32)
    assert attn.kv_repeat == 2
    x = torch.randn(2, 16, 64)
    out, _ = attn(x)
    assert out.shape == (2, 16, 64)

def test_fail_loud_missing_weights():
    cfg = MockConfig()
    class IncompleteAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(64, 64)
            # k_lin and v_lin are intentionally missing

    incomplete = IncompleteAttention()
    with pytest.raises(RuntimeError):
        ChelaSlayAttention(config=cfg, pretrained_attn=incomplete, feature_dim=32)

def test_nested_encoder_weight_copy():
    class DummySelf(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)

    class DummyNested(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = DummySelf()
            self.output = nn.Module()
            self.output.dense = nn.Linear(64, 64)

    nested = DummyNested()
    cfg = MockConfig()
    attn = ChelaSlayAttention(config=cfg, pretrained_attn=nested, feature_dim=32)
    assert attn.is_nested_encoder
    x = torch.randn(2, 16, 64)
    out, _ = attn(x)
    assert out.shape == (2, 16, 64)

def test_inject_chela_slay():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Module()
            self.attention.q_lin = nn.Linear(64, 64)
            self.attention.k_lin = nn.Linear(64, 64)
            self.attention.v_lin = nn.Linear(64, 64)
            self.attention.out_lin = nn.Linear(64, 64)

    layer = DummyLayer()
    res = inject_chela_slay(layer=layer, model_config=MockConfig(), feature_dim=32)
    assert res is not None
    assert isinstance(layer.attention, ChelaSlayAttention)
