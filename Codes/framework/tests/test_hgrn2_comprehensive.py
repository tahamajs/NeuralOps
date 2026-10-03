import pytest
import torch
import torch.nn as nn
from framework.attentions.hgrn2_gated_deltanet import (
    HGRN2GatedDeltaNet, inject_hgrn2_gated_deltanet)
from framework.attentions.hgrn2_layer import HGRN2Layer, inject_hgrn2_layer


def test_hgrn2_layer_standalone():
    layer = HGRN2Layer(d_model=64, is_decoder=False, lora_rank=8)
    x = torch.randn(2, 8, 64)
    out, _ = layer(x)
    assert out.shape == (2, 8, 64)
    loss = out.sum()
    loss.backward()
    assert layer.v_lora_A.grad is not None

def test_hgrn2_gated_deltanet_standalone():
    net = HGRN2GatedDeltaNet(d_model=64, num_heads=4, is_decoder=False, lora_rank=8)
    x = torch.randn(2, 8, 64)
    out, _ = net(x)
    assert out.shape == (2, 8, 64)
    loss = out.sum()
    loss.backward()
    assert net.out_proj.lora_A.grad is not None

def test_hgrn2_copy_weights_and_freezing():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
    old = DummyAttn()
    net = HGRN2GatedDeltaNet(d_model=64, num_heads=4, old_attn=old, lora_rank=8)
    assert not net.out_proj.weight.requires_grad
    assert net.out_proj.lora_A.requires_grad
    assert net.out_proj.lora_B.requires_grad

def test_hgrn2_whole_model_injection():
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
    res = inject_hgrn2_gated_deltanet(model=model, lora_rank=8)
    assert res is not None
    assert isinstance(model.transformer.h[0].attention, HGRN2GatedDeltaNet)
    x = torch.randn(2, 8, 64)
    out, _ = model.transformer.h[0].attention(x)
    assert out.shape == (2, 8, 64)

def test_nested_encoder_bypass():
    net = HGRN2GatedDeltaNet(d_model=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    enc = torch.randn(2, 8, 64)
    out, _ = net(x, encoder_hidden_states=enc)
    assert torch.equal(out, x)
