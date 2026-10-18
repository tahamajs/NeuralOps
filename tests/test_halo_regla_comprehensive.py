import pytest
import torch
import torch.nn as nn
from neuralops.attentions.halo_regla import (HALOReGLAAttention,
                                             inject_halo_regla)


def test_halo_regla_standalone():
    attn = HALOReGLAAttention(hidden_size=64, num_heads=4, lora_rank=8)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)
    loss = out.sum()
    loss.backward()
    assert attn.q_proj.lora_A.grad is not None
    assert attn.q_proj.weight.grad is None

def test_halo_regla_copy_weights_and_freezing():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
    old = DummyAttn()
    attn = HALOReGLAAttention(hidden_size=64, num_heads=4, pretrained_attn=old, lora_rank=8)
    assert not attn.q_proj.weight.requires_grad
    assert attn.q_proj.lora_A.requires_grad
    assert attn.q_proj.lora_B.requires_grad

def test_halo_regla_whole_model_injection():
    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
            self.attention.hidden_size = 64
            self.attention.num_heads = 4
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([MockLayer() for _ in range(2)])
    model = MockModel()
    res = inject_halo_regla(model=model, lora_rank=8)
    assert res is not None
    assert isinstance(model.transformer.h[0].attention, HALOReGLAAttention)
    x = torch.randn(2, 8, 64)
    out, _ = model.transformer.h[0].attention(x)
    assert out.shape == (2, 8, 64)

def test_halo_regla_odd_dim_rope():
    attn = HALOReGLAAttention(hidden_size=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    cos = torch.randn(8, 15)
    sin = torch.randn(8, 15)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)

def test_nested_encoder_bypass():
    attn = HALOReGLAAttention(hidden_size=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    enc = torch.randn(2, 8, 64)
    out, _ = attn(x, encoder_hidden_states=enc)
    assert torch.equal(out, x)
