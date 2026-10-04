"""Comprehensive test suite for EAT (Efficient Attention Transformer)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.eat import (EATBlock, LoRALinear, _rotate_half,
                                      inject_eat)
from neuralops.core.registry import ATTENTION_REGISTRY


def test_registry():
    assert "eat" in ATTENTION_REGISTRY.list_available()
    assert "eat_attention" in ATTENTION_REGISTRY.list_available()
    assert ATTENTION_REGISTRY.get("eat") is EATBlock


def test_lora_linear():
    layer = LoRALinear(in_features=64, out_features=128, bias=True, lora_rank=8, lora_alpha=16.0)
    x = torch.randn(2, 10, 64)
    out = layer(x)
    assert out.shape == (2, 10, 128)

    # Base weight frozen check
    layer.freeze_base_weights()
    assert not layer.weight.requires_grad
    assert not layer.bias.requires_grad
    assert layer.lora_A.requires_grad
    assert layer.lora_B.requires_grad


def test_eat_forward_bidirectional():
    config = SimpleNamespace(hidden_size=64, num_attention_heads=4, is_decoder=False)
    eat = EATBlock(config=config, d_model=64, n_heads=4)
    x = torch.randn(2, 16, 64)
    mask = torch.ones(2, 16)
    out, attn = eat(x, attention_mask=mask)
    assert out.shape == (2, 16, 64)
    assert attn is None


def test_eat_forward_causal():
    config = SimpleNamespace(hidden_size=64, num_attention_heads=4, is_decoder=True)
    eat = EATBlock(config=config, d_model=64, n_heads=4)
    x = torch.randn(2, 16, 64)
    out, attn = eat(x, output_attentions=True)
    assert out.shape == (2, 16, 64)


def test_eat_with_token_pruning():
    config = SimpleNamespace(hidden_size=64, num_attention_heads=4, keep_ratio=0.5, enable_pruning=True)
    eat = EATBlock(config=config, d_model=64, n_heads=4, keep_ratio=0.5)
    assert eat.prune_layer is not None
    assert eat.prune_layer.keep_ratio == 0.5

    x = torch.randn(2, 16, 64)
    eat.eval()
    out, _ = eat(x)
    assert out.shape == (2, 16, 64)


def test_rotate_half_odd_dimension():
    x = torch.randn(2, 4, 15)
    rot = _rotate_half(x)
    assert rot.shape == x.shape
    # even part (0..13) is rotated, last element (14) is passed through untouched
    assert torch.allclose(rot[..., 14], x[..., 14])


def test_copy_weights_distilbert():
    class MockDistilBertAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(64, 64)
            self.k_lin = nn.Linear(64, 64)
            self.v_lin = nn.Linear(64, 64)
            self.out_lin = nn.Linear(64, 64)

    mock = MockDistilBertAttn()
    config = SimpleNamespace(hidden_size=64, num_attention_heads=4)
    eat = EATBlock(config=config, d_model=64, n_heads=4)
    eat.copy_weights(mock)
    assert torch.allclose(eat.q_lin.weight, mock.q_lin.weight)
    assert torch.allclose(eat.out_lin.weight, mock.out_lin.weight)


def test_copy_weights_gpt2():
    class MockGPT2Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(64, 192)
            self.c_proj = nn.Linear(64, 64)

    mock = MockGPT2Attn()
    config = SimpleNamespace(hidden_size=64, num_attention_heads=4, is_decoder=True)
    eat = EATBlock(config=config, d_model=64, n_heads=4)
    eat.copy_weights(mock)
    assert eat.q_lin.weight.shape == (64, 64)
    assert eat.out_lin.weight.shape == (64, 64)


def test_copy_weights_missing_raises():
    class IncompleteAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(64, 64)

    mock = IncompleteAttn()
    config = SimpleNamespace(hidden_size=64, num_attention_heads=4)
    eat = EATBlock(config=config, d_model=64, n_heads=4)
    with pytest.raises(RuntimeError):
        eat.copy_weights(mock)


def test_inject_eat_single_layer():
    class MockAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(64, 64)
            self.k_lin = nn.Linear(64, 64)
            self.v_lin = nn.Linear(64, 64)
            self.out_lin = nn.Linear(64, 64)

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = MockAttn()

    layer = MockLayer()
    config = SimpleNamespace(hidden_size=64, num_attention_heads=4)
    ret = inject_eat(layer=layer, model_config=config, layer_idx=0, total_layers=6)
    assert isinstance(ret, EATBlock)
    assert layer.attention is ret


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
