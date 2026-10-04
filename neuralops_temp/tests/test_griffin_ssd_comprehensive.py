import pytest
import torch
import torch.nn as nn
from neuralops.attentions.griffin_ssd import (GriffinSSDBlock,
                                              inject_griffin_ssd)


def test_griffin_ssd_standalone():
    block = GriffinSSDBlock(d_model=64, num_heads=4, lora_rank=8, is_decoder=False)
    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert out.shape == (2, 8, 64)
    loss = out.sum()
    loss.backward()
    assert block.W_q.lora_A.grad is not None
    assert block.W_q.weight.grad is None

def test_griffin_ssd_causal_mode():
    block = GriffinSSDBlock(d_model=64, num_heads=4, lora_rank=8, is_decoder=True)
    x = torch.randn(1, 8, 64)
    out, _ = block(x)
    assert out.shape == (1, 8, 64)

def test_griffin_ssd_copy_weights_and_freezing():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
    old = DummyAttn()
    block = GriffinSSDBlock(d_model=64, num_heads=4, old_attn=old, lora_rank=8)
    assert not block.W_q.weight.requires_grad
    assert block.W_q.lora_A.requires_grad
    assert block.W_q.lora_B.requires_grad

def test_griffin_ssd_whole_model_injection():
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
    res = inject_griffin_ssd(model=model, lora_rank=8)
    assert res is not None
    assert isinstance(model.transformer.h[0].attention, GriffinSSDBlock)
    x = torch.randn(2, 8, 64)
    out, _ = model.transformer.h[0].attention(x)
    assert out.shape == (2, 8, 64)

def test_griffin_ssd_odd_dim_rope():
    block = GriffinSSDBlock(d_model=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    cos = torch.randn(8, 15)
    sin = torch.randn(8, 15)
    out, _ = block(x, position_embeddings=(cos, sin))
    assert out.shape == (2, 8, 64)

def test_nested_encoder_bypass():
    block = GriffinSSDBlock(d_model=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    enc = torch.randn(2, 8, 64)
    out, _ = block(x, encoder_hidden_states=enc)
    assert torch.equal(out, x)


def test_griffin_ssd_comp_config_and_gqa():
    class DummyOldAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 16
            self.k_proj = nn.Linear(64, 32)  # 32 // 16 = 2 kv heads
            self.q_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 32)
            self.out_proj = nn.Linear(64, 64)

    old = DummyOldAttn()
    comp_config = {"lora_rank": 4, "window_size": 32}
    block = GriffinSSDBlock(d_model=64, num_heads=4, old_attn=old, comp_config=comp_config)
    assert block.num_heads == 4
    assert block.num_kv_heads == 2
    assert block.head_dim == 16
    assert block.lora_rank == 4
    assert block.window_size == 32

    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert out.shape == (2, 8, 64)


def test_griffin_ssd_rotary_emb_probing():
    class MockRotaryEmb(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, q, seq_len=None):
            if seq_len is None:
                seq_len = q.shape[-2]
            cos = torch.ones(seq_len, self.dim)
            sin = torch.zeros(seq_len, self.dim)
            return cos, sin

    class DummyAttnWithRoPE(nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = MockRotaryEmb(16)
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)

    old = DummyAttnWithRoPE()
    block = GriffinSSDBlock(d_model=64, num_heads=4, old_attn=old)
    assert block.rotary_emb is not None

    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert out.shape == (2, 8, 64)

