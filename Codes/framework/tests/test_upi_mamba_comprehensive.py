import pytest
import torch
import torch.nn as nn
import types

from framework.attentions.upi_stuffed_mamba import (
    UPIMambaBlock,
    LoRALinear,
    inject_upi_mamba,
)


def make_config(hidden_size=64, is_decoder=False, model_type="bert"):
    return types.SimpleNamespace(
        hidden_size=hidden_size,
        is_decoder=is_decoder,
        model_type=model_type,
        dropout=0.0,
        lora_rank=16,
    )


class DummyAttn(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.hidden_size = d


def test_upi_mamba_forward_and_shapes():
    cfg = make_config(hidden_size=64)
    attn = UPIMambaBlock(cfg, pretrained_attn=DummyAttn(64), d_state=16, d_conv=4)
    attn.eval()

    for seq_len in [4, 16, 25]:
        B, L, D = 2, seq_len, 64
        x = torch.randn(B, L, D)
        out, _ = attn(x)
        assert out.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out.shape}"
        assert not torch.isnan(out).any()


def test_upi_mamba_causal_mode():
    cfg = make_config(hidden_size=64, is_decoder=True, model_type="gpt2")
    attn = UPIMambaBlock(cfg, pretrained_attn=DummyAttn(64), is_decoder=True)
    attn.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    out, _ = attn(x)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()


def test_upi_mamba_masks():
    cfg = make_config(hidden_size=64)
    attn = UPIMambaBlock(cfg, pretrained_attn=DummyAttn(64))
    attn.eval()

    B, L, D = 2, 12, 64
    x = torch.randn(B, L, D)

    # 2D boolean/binary mask
    mask_2d = torch.ones(B, L)
    mask_2d[:, -3:] = 0
    out_2d, _ = attn(x, attention_mask=mask_2d)
    assert out_2d.shape == (B, L, D)
    assert not torch.isnan(out_2d).any()

    # 4D attention mask
    mask_4d = torch.zeros(B, 1, L, L)
    mask_4d[:, :, :, -3:] = -10000.0
    out_4d, _ = attn(x, attention_mask=mask_4d)
    assert out_4d.shape == (B, L, D)
    assert not torch.isnan(out_4d).any()


def test_upi_mamba_lora_and_gradients():
    cfg = make_config(hidden_size=64)
    attn = UPIMambaBlock(cfg, pretrained_attn=DummyAttn(64), lora_rank=8)
    attn.train()

    # Check base weights frozen and lora weights trainable
    assert not attn.in_proj.weight.requires_grad
    assert attn.in_proj.lora_A.requires_grad
    assert attn.in_proj.lora_B.requires_grad
    assert not attn.out_proj.weight.requires_grad
    assert attn.out_proj.lora_A.requires_grad
    assert attn.out_proj.lora_B.requires_grad

    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = attn(x)
    loss = out.sum()
    loss.backward()

    assert attn.in_proj.lora_A.grad is not None
    assert attn.in_proj.lora_B.grad is not None
    assert attn.out_proj.lora_A.grad is not None
    assert attn.out_proj.lora_B.grad is not None
    assert attn.in_proj.weight.grad is None
    assert attn.out_proj.weight.grad is None


def test_upi_mamba_inject():
    class DummyLayer(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.attention = DummyAttn(d)

    layer = DummyLayer(64)
    cfg = make_config(hidden_size=64)
    injected = inject_upi_mamba(layer, model_config=cfg, old_attn=layer.attention)
    assert isinstance(injected, UPIMambaBlock)
    assert layer.attention is injected

    # Test None graceful handling
    res = inject_upi_mamba(layer, model_config=cfg, old_attn=None)
    assert res is None
