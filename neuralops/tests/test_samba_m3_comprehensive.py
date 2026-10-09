import types

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.samba_m3 import (LoRALinear, SAMBAM3Block,
                                           _rotate_half, apply_rotary_pos_emb,
                                           inject_samba_m3)


def make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=False, model_type="bert"):
    return types.SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=hidden_size // num_heads,
        is_decoder=is_decoder,
        model_type=model_type,
        dropout=0.0,
        lora_rank=16,
    )


class DummyAttn(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.hidden_size = d
        self.num_heads = h


def test_samba_m3_forward_and_shapes():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = SAMBAM3Block(cfg, d_model=64, n_heads=4, pretrained_attn=DummyAttn(64, 4))
    attn.eval()

    for seq_len in [2, 8, 16, 25]:
        B, L, D = 2, seq_len, 64
        x = torch.randn(B, L, D)
        out, _ = attn(x)
        assert out.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out.shape}"
        assert not torch.isnan(out).any()


def test_samba_m3_causal_mode():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    attn = SAMBAM3Block(cfg, d_model=64, n_heads=4, is_decoder=True, pretrained_attn=DummyAttn(64, 4))
    attn.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    out, _ = attn(x)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()


def test_samba_m3_rotate_half_and_odd_dim():
    for d in [16, 17, 31, 32]:
        t = torch.randn(2, 4, 8, d)
        rotated = _rotate_half(t)
        assert rotated.shape == t.shape, f"Odd dim {d} failed in _rotate_half"

    head_dim_odd = 15
    q = torch.randn(2, 4, 8, head_dim_odd)
    k = torch.randn(2, 2, 8, head_dim_odd)
    cos = torch.randn(8, head_dim_odd)
    sin = torch.randn(8, head_dim_odd)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.isnan(q_rot).any()


def test_samba_m3_sliding_window_and_masks():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = SAMBAM3Block(cfg, d_model=64, n_heads=4, window=4, pretrained_attn=DummyAttn(64, 4))
    attn.eval()

    B, L, D = 2, 12, 64
    x = torch.randn(B, L, D)

    # 2D mask
    mask_2d = torch.ones(B, L)
    mask_2d[:, -3:] = 0
    out_2d, _ = attn(x, attention_mask=mask_2d)
    assert out_2d.shape == (B, L, D)
    assert not torch.isnan(out_2d).any()


def test_samba_m3_lora_and_gradients():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = SAMBAM3Block(cfg, d_model=64, n_heads=4, pretrained_attn=DummyAttn(64, 4), lora_rank=8)
    attn.train()

    # Base frozen, lora trainable
    assert not attn.q_lin.weight.requires_grad
    assert attn.q_lin.lora_A.requires_grad
    assert attn.q_lin.lora_B.requires_grad

    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = attn(x)
    loss = out.sum()
    loss.backward()

    assert attn.q_lin.lora_A.grad is not None
    assert attn.q_lin.lora_B.grad is not None
    assert attn.q_lin.weight.grad is None


def test_samba_m3_inject():
    class DummyLayer(nn.Module):
        def __init__(self, d=64, h=4):
            super().__init__()
            self.attention = DummyAttn(d, h)

    layer = DummyLayer(64, 4)
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    injected = inject_samba_m3(layer, model_config=cfg, old_attn=layer.attention)
    assert isinstance(injected, SAMBAM3Block)
    assert layer.attention is injected

    empty_layer = nn.Module()
    assert inject_samba_m3(empty_layer, model_config=cfg, old_attn=None) is None


# ===========================================================================
# Targeted Tests for SAMBA-M3 Bug Fixes
# ===========================================================================

def test_samba_m3_decoder_return_tuple_layout():
    """Verify decoder return layout for output_attentions True and False."""
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    attn = SAMBAM3Block(cfg, d_model=64, n_heads=4, is_decoder=True)

    x = torch.randn(2, 4, 64)
    # 1. output_attentions=False -> (out, present_key_value)
    res = attn(x, output_attentions=False)
    assert len(res) == 2, f"Expected 2-tuple, got {len(res)}"
    out, present = res
    assert out.shape == (2, 4, 64)
    assert isinstance(present, tuple)
    assert present[0].shape == (2, 2, 4, 16)

    # 2. output_attentions=True -> (out, present_key_value, None)
    res_attn = attn(x, output_attentions=True)
    assert len(res_attn) == 3, f"Expected 3-tuple, got {len(res_attn)}"
    out_a, present_a, attn_w = res_attn
    assert out_a.shape == (2, 4, 64)
    assert attn_w is None


def test_samba_m3_past_state_propagation_to_mamba():
    """Verify past_state is passed to mamba and included in present_key_value."""
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    attn = SAMBAM3Block(cfg, d_model=64, n_heads=4, is_decoder=True)

    # Mock mamba returning (out, present_mamba)
    mock_state = torch.randn(2, 16)
    def mock_mamba_fwd(h, **kwargs):
        return h, mock_state
    attn.mamba.forward = mock_mamba_fwd

    x = torch.randn(2, 4, 64)
    out, present = attn(x, past_state=mock_state)
    assert len(present) == 3, "present_key_value should be (k, v, present_mamba)"
    k, v, pres_m = present
    assert pres_m is mock_state


def test_samba_m3_3d_4d_mask_chunking_past_len():
    """Verify 3D and 4D attention mask query slicing with past_len > 0."""
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    attn = SAMBAM3Block(cfg, d_model=64, n_heads=4, is_decoder=True)

    past_k = torch.randn(2, 2, 8, 16)
    past_v = torch.randn(2, 2, 8, 16)
    past_kv = (past_k, past_v)

    x_new = torch.randn(2, 2, 64)  # L=2, past_len=8 -> total keys=10

    # 3D mask of shape (B, 10, 10)
    mask_3d = torch.zeros(2, 10, 10)
    out_3d, _ = attn(x_new, attention_mask=mask_3d, past_key_value=past_kv)
    assert out_3d.shape == (2, 2, 64)
    assert torch.isfinite(out_3d).all()

    # 4D mask of shape (B, 1, 10, 10)
    mask_4d = torch.zeros(2, 1, 10, 10)
    out_4d, _ = attn(x_new, attention_mask=mask_4d, past_key_value=past_kv)
    assert out_4d.shape == (2, 2, 64)
    assert torch.isfinite(out_4d).all()


def test_samba_m3_spectral_mamba_2d_and_3d():
    """Verify SpectralMamba handles both 2D and 3D shapes without broadcasting issues."""
    from neuralops.attentions.samba_m3 import SpectralMamba
    sm = SpectralMamba(d_model=64)
    
    # 2D input (B, D)
    x_2d = torch.randn(2, 64)
    out_2d, _ = sm(x_2d)
    assert out_2d.shape == (2, 64)

    # 3D input (B, L, D)
    x_3d = torch.randn(2, 8, 64)
    out_3d, _ = sm(x_3d)
    assert out_3d.shape == (2, 8, 64)

