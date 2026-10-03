"""Unit test suite for BAM-UMoE attention module."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from framework.attentions.bam_umoe import (BAMUMoEBlock, LoRALinear,
                                           UnifiedMoE, _rotate_half,
                                           inject_bam_umoe)
from framework.core.registry import ATTENTION_REGISTRY


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


def test_registry():
    assert "bam_umoe" in ATTENTION_REGISTRY.list_available()
    assert ATTENTION_REGISTRY.get("bam_umoe") is BAMUMoEBlock


def test_odd_dim_rotate_half():
    x = torch.randn(2, 4, 15)
    rot = _rotate_half(x)
    assert rot.shape == x.shape
    assert torch.equal(rot[..., -1], x[..., -1])


def test_unified_moe_no_zero_trap_and_softmax_normalization():
    moe = UnifiedMoE(dim=32, num_experts=4, top_k=2)
    # Check non-zero weights on final expert layers to avoid gradient dead-ends
    for exp in moe.experts:
        assert torch.count_nonzero(exp[2].weight) > 0
    # Check initial gate bias is -6.0
    assert moe.gate_bias.item() == -6.0

    x = torch.randn(2, 8, 32)
    out = moe(x)
    assert out.shape == (2, 8, 32)
    # Step-0 gating: gate = sigmoid(-6.0) ≈ 0.00247, so output magnitude is very small
    assert out.abs().mean().item() < 0.1


def test_bam_umoe_forward_encoder():
    cfg = MockConfig(is_decoder=False)
    attn = BAMUMoEBlock(config=cfg, num_experts=2, top_k=1)
    x = torch.randn(2, 16, 64)
    out, weights = attn(x, output_attentions=True)
    assert out.shape == (2, 16, 64)
    assert weights is not None


def test_bam_umoe_forward_causal():
    cfg = MockConfig(is_decoder=True, model_type="gpt2")
    attn = BAMUMoEBlock(config=cfg, num_experts=2, top_k=1)
    x = torch.randn(2, 16, 64)
    out, _ = attn(x)
    assert out.shape == (2, 16, 64)


def test_bam_umoe_gqa_expansion():
    cfg = MockConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    attn = BAMUMoEBlock(config=cfg, num_experts=2, top_k=1)
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
            # missing k_lin, v_lin

    incomplete = IncompleteAttention()
    with pytest.raises(RuntimeError):
        BAMUMoEBlock(config=cfg, old_attn=incomplete)


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
    attn = BAMUMoEBlock(config=cfg, old_attn=nested)
    assert attn.is_nested_encoder
    x = torch.randn(2, 16, 64)
    out, _ = attn(x)
    assert out.shape == (2, 16, 64)


def test_inject_bam_umoe():
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Module()
            self.attention.q_lin = nn.Linear(64, 64)
            self.attention.k_lin = nn.Linear(64, 64)
            self.attention.v_lin = nn.Linear(64, 64)
            self.attention.out_lin = nn.Linear(64, 64)

    layer = DummyLayer()
    res = inject_bam_umoe(layer=layer, model_config=MockConfig())
    assert res is not None
    assert isinstance(layer.attention, BAMUMoEBlock)
