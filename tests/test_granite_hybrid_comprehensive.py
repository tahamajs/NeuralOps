import pytest
import torch
import torch.nn as nn
from neuralops.attentions.granite_hybrid import (GraniteHybridBlock,
                                                 inject_granite_hybrid)


def test_granite_hybrid_standalone():
    block = GraniteHybridBlock(dim=64, num_heads=4, lora_rank=8, ratio=1)
    x = torch.randn(2, 8, 64)
    # Step 1: Attention branch (counter=1, 1 % 2 = 1)
    out1, _ = block(x)
    assert out1.shape == (2, 8, 64)
    # Step 2: Mamba branch (counter=2, 2 % 2 = 0)
    out2, _ = block(x)
    assert out2.shape == (2, 8, 64)
    loss = (out1 + out2).sum()
    loss.backward()
    assert block.q_proj.lora_A.grad is not None
    assert block.q_proj.weight.grad is None

def test_granite_hybrid_copy_weights_and_freezing():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
    old = DummyAttn()
    block = GraniteHybridBlock(dim=64, num_heads=4, pretrained_attn=old, lora_rank=8)
    assert not block.q_proj.weight.requires_grad
    assert block.q_proj.lora_A.requires_grad
    assert block.q_proj.lora_B.requires_grad

def test_granite_hybrid_whole_model_injection():
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
    res = inject_granite_hybrid(model=model, lora_rank=8)
    assert res is not None
    assert isinstance(model.transformer.h[0].attention, GraniteHybridBlock)
    x = torch.randn(2, 8, 64)
    out, _ = model.transformer.h[0].attention(x)
    assert out.shape == (2, 8, 64)

def test_nested_encoder_bypass():
    block = GraniteHybridBlock(dim=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    enc = torch.randn(2, 8, 64)
    out, _ = block(x, encoder_hidden_states=enc)
    assert torch.equal(out, x)


def test_granite_hybrid_gqa_and_rope():
    class DummyQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 32)
            self.v_proj = nn.Linear(64, 32)
            self.out_proj = nn.Linear(64, 64)
            self.num_attention_heads = 4
            self.num_key_value_heads = 2
            self.head_dim = 16

    old = DummyQwen()
    block = GraniteHybridBlock(dim=64, old_attn=old, ratio=0)
    assert block.num_kv_heads == 2
    assert block.kv_repeat == 2
    x = torch.randn(2, 8, 64)
    cos = torch.randn(2, 1, 8, 16)
    sin = torch.randn(2, 1, 8, 16)
    out, _ = block(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)


def test_granite_hybrid_3d_mask():
    block = GraniteHybridBlock(dim=64, num_heads=4, ratio=0)
    x = torch.randn(2, 8, 64)
    mask = torch.zeros(2, 8, 8)
    out, _ = block(x, attention_mask=mask)
    assert out.shape == (2, 8, 64)

