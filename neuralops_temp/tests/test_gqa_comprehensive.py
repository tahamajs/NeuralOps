import pytest
import torch
import torch.nn as nn
from neuralops.attentions.gqa_attention import (GroupedQueryAttention,
                                                LoRALinear,
                                                apply_rotary_pos_emb,
                                                inject_gqa)


def test_gqa_standalone():
    block = GroupedQueryAttention(dim=64, num_heads=4, num_kv_heads=2, lora_rank=8)
    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert out.shape == (2, 8, 64)
    loss = out.sum()
    loss.backward()
    assert block.query.lora_A.grad is not None
    assert block.query.weight.grad is None

def test_gqa_copy_weights_and_freezing():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 32)
            self.v_proj = nn.Linear(64, 32)
            self.out_proj = nn.Linear(64, 64)
    old = DummyAttn()
    block = GroupedQueryAttention(dim=64, num_heads=4, num_kv_heads=2, pretrained_attn=old, lora_rank=8)
    assert not block.query.weight.requires_grad
    assert block.query.lora_A.requires_grad
    assert block.query.lora_B.requires_grad

def test_gqa_odd_dim_rope():
    # Odd head dimension safety check
    q = torch.randn(2, 4, 8, 17)
    k = torch.randn(2, 2, 8, 17)
    cos = torch.randn(8, 17)
    sin = torch.randn(8, 17)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape

def test_gqa_whole_model_injection():
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
    res = inject_gqa(model=model, lora_rank=8)
    assert res is not None
    assert isinstance(model.transformer.h[0].attention, GroupedQueryAttention)
    x = torch.randn(2, 8, 64)
    out, _ = model.transformer.h[0].attention(x)
    assert out.shape == (2, 8, 64)

def test_nested_encoder_bypass():
    block = GroupedQueryAttention(dim=64, num_heads=4, num_kv_heads=2)
    x = torch.randn(2, 8, 64)
    enc = torch.randn(2, 8, 64)
    out, _ = block(x, encoder_hidden_states=enc)
    assert torch.equal(out, x)
