import types

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.token_compressed_attention import (
    LoRALinear, TokenCompressedAttention, _rotate_half, apply_rotary_pos_emb,
    inject_token_compressed_attn)


def make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=False, model_type="bert", lora_rank=16, lora_alpha=32.0):
    return types.SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=hidden_size // num_heads,
        is_decoder=is_decoder,
        model_type=model_type,
        dropout=0.0,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
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


class DummyDistilBertAttn(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.q_lin = nn.Linear(d, d)
        self.k_lin = nn.Linear(d, d)
        self.v_lin = nn.Linear(d, d)
        self.out_lin = nn.Linear(d, d)
        self.hidden_size = d
        self.num_heads = h


class DummyRobertaSelf(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)


class DummyRobertaOutput(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.dense = nn.Linear(d, d)


class DummyRobertaLayer(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.self = DummyRobertaSelf(d, h)
        self.output = DummyRobertaOutput(d)


class DummyGPT2Attn(nn.Module):
    def __init__(self, d=64, h=4):
        super().__init__()
        self.c_attn = nn.Linear(d, 3 * d)
        self.c_proj = nn.Linear(d, d)
        self.split_size = d
        self.num_heads = h
        self.is_decoder = True


class DummyRotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, seq_len=None):
        L = seq_len if seq_len is not None else x.shape[-2]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(L, device=x.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def test_token_compressed_forward_and_shapes():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = TokenCompressedAttention(cfg, pretrained_attn=DummyAttn(64, 4), compression_ratio=0.5)
    attn.eval()

    for seq_len in [2, 7, 16, 32]:
        B, L, D = 2, seq_len, 64
        x = torch.randn(B, L, D)
        out, _ = attn(x)
        assert out.shape == (B, L, D), f"Failed for seq_len {seq_len}: expected {(B, L, D)}, got {out.shape}"
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


def test_token_compressed_causal_mode():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2, is_decoder=True, model_type="gpt2")
    attn = TokenCompressedAttention(cfg, pretrained_attn=DummyAttn(64, 4), is_decoder=True, compression_ratio=0.5)
    attn.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    out, _ = attn(x)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_token_compressed_masks():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = TokenCompressedAttention(cfg, pretrained_attn=DummyAttn(64, 4), compression_ratio=0.5)
    attn.eval()

    B, L, D = 2, 12, 64
    x = torch.randn(B, L, D)

    # 2D mask
    mask_2d = torch.ones(B, L)
    mask_2d[:, -3:] = 0
    out_2d, _ = attn(x, attention_mask=mask_2d)
    assert out_2d.shape == (B, L, D)
    assert not torch.isnan(out_2d).any()

    # 4D mask
    mask_4d = torch.zeros(B, 1, L, L)
    mask_4d[:, :, :, -3:] = -10000.0
    out_4d, _ = attn(x, attention_mask=mask_4d)
    assert out_4d.shape == (B, L, D)
    assert not torch.isnan(out_4d).any()


def test_token_compressed_lora_and_gradients():
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    attn = TokenCompressedAttention(cfg, pretrained_attn=DummyAttn(64, 4), lora_rank=8)
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


def test_token_compressed_config_none_safety():
    # Must not raise AttributeError when config=None
    attn = TokenCompressedAttention(config=None, hidden_size=64, num_heads=4, lora_rank=8, lora_alpha=16.0)
    assert attn.dim == 64
    assert attn.num_heads == 4
    assert attn.lora_rank == 8
    assert attn.lora_alpha == 16.0

    comp_cfg = types.SimpleNamespace(lora_rank=12, lora_alpha=24.0, compression_ratio=0.6)
    attn2 = TokenCompressedAttention(config=None, comp_config=comp_cfg, hidden_size=64, num_heads=4)
    assert attn2.lora_rank == 12
    assert attn2.lora_alpha == 24.0
    assert attn2.compression_ratio == 0.6


def test_token_compressed_distilbert_copy():
    src = DummyDistilBertAttn(64, 4)
    attn = TokenCompressedAttention(config=None, pretrained_attn=src, hidden_size=64, num_heads=4)
    assert not attn.is_nested_encoder
    assert torch.allclose(attn.q_lin.weight, src.q_lin.weight)
    assert torch.allclose(attn.out_lin.weight, src.out_lin.weight)


def test_token_compressed_roberta_nested_copy():
    layer = DummyRobertaLayer(64, 4)
    attn = TokenCompressedAttention(config=None, pretrained_attn=layer, hidden_size=64, num_heads=4)
    assert attn.is_nested_encoder
    assert torch.allclose(attn.q_lin.weight, layer.self.query.weight)
    assert torch.allclose(attn.k_lin.weight, layer.self.key.weight)

    # In forward, nested encoder bypasses out_lin
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)


def test_token_compressed_gpt2_conv1d_and_lora_wrapper():
    # Conv1D layout test
    gpt2_attn = DummyGPT2Attn(64, 4)
    # Simulate Conv1D weight [in_features, out_features]
    gpt2_attn.c_attn.weight.data = torch.randn(64, 192)
    gpt2_attn.c_proj.weight.data = torch.randn(64, 64)

    # Wrap c_attn in a dummy LoRALinear having base_layer
    class LoRAWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base_layer = base
            self.lora_A = nn.Parameter(torch.randn(4, 64))
            self.lora_B = nn.Parameter(torch.randn(192, 4))
            self.lora_scaling = 2.0

    wrapped_attn = types.SimpleNamespace(
        c_attn=LoRAWrapper(gpt2_attn.c_attn),
        c_proj=gpt2_attn.c_proj,
        num_heads=4,
        is_decoder=True
    )

    attn = TokenCompressedAttention(config=None, pretrained_attn=wrapped_attn, hidden_size=64, num_heads=4, lora_rank=4)
    assert attn.is_decoder
    # Verify proper transposed slice
    expected_q = gpt2_attn.c_attn.weight.data.t()[:64, :]
    assert torch.allclose(attn.q_lin.weight, expected_q)


def test_token_compressed_rotary_fallback():
    src = DummyAttn(64, 4)
    src.rotary_emb = DummyRotaryEmbedding(16)

    attn = TokenCompressedAttention(config=None, pretrained_attn=src, hidden_size=64, num_heads=4)
    assert attn.rotary_emb is not None

    # Forward without position_embeddings triggers self.rotary_emb
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)
    assert not torch.isnan(out).any()


def test_token_compressed_frobenius_check():
    # If source weight is zero, RuntimeError must be raised
    src = DummyAttn(64, 4)
    src.q_proj.weight.data.zero_()

    with pytest.raises(RuntimeError, match="norm is near zero"):
        TokenCompressedAttention(config=None, pretrained_attn=src, hidden_size=64, num_heads=4)


def test_token_compressed_inject_safe():
    class DummyLayer(nn.Module):
        def __init__(self, d=64, h=4):
            super().__init__()
            self.attention = DummyAttn(d, h)

    layer = DummyLayer(64, 4)
    cfg = make_config(hidden_size=64, num_heads=4, num_kv_heads=2)
    comp_cfg = types.SimpleNamespace(lora_rank=8, lora_alpha=16.0, compression_ratio=0.5)

    injected = inject_token_compressed_attn(layer, model_config=cfg, comp_config=comp_cfg, old_attn=layer.attention)
    assert isinstance(injected, TokenCompressedAttention)
    assert layer.attention is injected
    assert injected.lora_rank == 8
    assert injected.lora_alpha == 16.0

    # Test failure: zeroed weight returns None without replacing layer.attention
    bad_layer = DummyLayer(64, 4)
    bad_layer.attention.q_proj.weight.data.zero_()
    orig_attn = bad_layer.attention

    result = inject_token_compressed_attn(bad_layer, model_config=cfg, comp_config=comp_cfg, old_attn=bad_layer.attention)
    assert result is None
    assert bad_layer.attention is orig_attn

