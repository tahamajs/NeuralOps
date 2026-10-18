import types

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.samba_moe import (MinimalMoEFallback, SambaMoEBlock,
                                            inject_samba_moe)


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
        num_experts=4,
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


def test_samba_moe_forward_and_shapes():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    block = SambaMoEBlock(cfg, d_model=64, n_heads=4, n_experts=4, pretrained_attn=DummyAttn(64, 4))
    block.eval()

    for seq_len in [2, 8, 16, 25]:
        B, L, D = 2, seq_len, 64
        x = torch.randn(B, L, D)
        out, _ = block(x)
        assert out.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out.shape}"
        assert not torch.isnan(out).any()
        assert isinstance(block.aux_loss, torch.Tensor)


def test_samba_moe_causal_mode():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    block = SambaMoEBlock(cfg, d_model=64, n_heads=4, is_decoder=True, pretrained_attn=DummyAttn(64, 4))
    block.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    out, _ = block(x)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()


def test_samba_moe_backward_and_gradients():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    block = SambaMoEBlock(cfg, d_model=64, n_heads=4, n_experts=4, pretrained_attn=DummyAttn(64, 4), lora_rank=8)
    block.train()

    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = block(x)
    loss = out.sum() + block.aux_loss
    loss.backward()

    assert block.swa_mamba.q_lin.lora_A.grad is not None
    assert block.swa_mamba.q_lin.weight.grad is None


def test_samba_moe_inject():
    class DummyLayer(nn.Module):
        def __init__(self, d=64, h=4):
            super().__init__()
            self.attention = DummyAttn(d, h)

    layer = DummyLayer(64, 4)
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    injected = inject_samba_moe(layer, model_config=cfg, old_attn=layer.attention)
    assert isinstance(injected, SambaMoEBlock)
    assert layer.attention is injected

    empty_layer = nn.Module()
    assert inject_samba_moe(empty_layer, model_config=cfg, old_attn=None) is None


# ===========================================================================
# Targeted Tests for Samba-MoE Bug Fixes
# ===========================================================================

def test_samba_moe_decoder_return_signature_and_kv():
    """Verify return signature and past_state propagation for decoders and encoders."""
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    block = SambaMoEBlock(cfg, d_model=64, n_heads=4, is_decoder=True)

    x = torch.randn(2, 4, 64)
    # Decoder output_attentions=False -> (out, past_state)
    res = block(x, output_attentions=False)
    assert len(res) == 2, f"Expected 2-tuple, got {len(res)}"
    out, past_state = res
    assert out.shape == (2, 4, 64)
    assert past_state is not None

    # Decoder output_attentions=True -> (out, past_state, None)
    res_attn = block(x, output_attentions=True)
    assert len(res_attn) == 3, f"Expected 3-tuple, got {len(res_attn)}"
    out_a, past_a, attn_w = res_attn
    assert out_a.shape == (2, 4, 64)
    assert attn_w is None

    # Encoder -> (out, None)
    cfg_enc = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=False, model_type="bert")
    block_enc = SambaMoEBlock(cfg_enc, d_model=64, n_heads=4, is_decoder=False)
    res_enc = block_enc(x)
    assert len(res_enc) == 2
    out_e, state_e = res_enc
    assert state_e is None


def test_samba_moe_minimal_fallback_aux_loss():
    """Verify functional load-balancing auxiliary loss in MinimalMoEFallback."""
    from neuralops.attentions.ssm.samba_moe import MinimalMoEFallback
    moe = MinimalMoEFallback(d_model=64, n_experts=4)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out, weights = moe(x)
    assert out.shape == (2, 8, 64)
    assert weights.shape == (2, 8, 4)
    assert moe.aux_loss.item() > 0.0, "aux_loss should be positive functional loss"
    moe.aux_loss.backward()
    assert moe.router.weight.grad is not None


def test_samba_moe_norm_dtype_consistency():
    """Verify LayerNorm dtype matches target_dtype."""
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=False)
    block = SambaMoEBlock(cfg, d_model=64, n_heads=4)
    x = torch.randn(2, 4, 64, dtype=torch.float32)
    out, _ = block(x)
    assert out.dtype == torch.float32

