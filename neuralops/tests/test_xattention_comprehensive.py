import types

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.xattention import (LoRALinear, XAttention,
                                             _rotate_half,
                                             apply_rotary_pos_emb,
                                             inject_xattention)


def make_config(hidden_size=64, num_heads=4, is_decoder=False, model_type="bert", block_size=4, sparsity=0.5):
    return types.SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=hidden_size // num_heads,
        is_decoder=is_decoder,
        model_type=model_type,
        dropout=0.0,
        lora_rank=16,
        xattn_block_size=block_size,
        xattn_sparsity=sparsity,
    )


class DummyAttn(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.hidden_size = d
        self.num_heads = 4


def test_block_sparse_routing():
    # Sequence length 16, block size 4 -> 4 blocks.
    # sparsity = 0.5 -> select top 2 blocks per query block.
    cfg = make_config(hidden_size=64, num_heads=4, is_decoder=False, model_type="bert", block_size=4, sparsity=0.5)
    attn = XAttention(cfg, pretrained_attn=DummyAttn(64), block_size=4, sparsity=0.5)
    attn.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)

    out, attn_weights = attn(x, output_attentions=True)
    assert out.shape == (B, L, D)
    assert attn_weights is not None
    assert attn_weights.shape == (B, 4, L, L)
    assert not torch.isnan(out).any()

    # Verify that in each query row, some blocks are masked out (near zero)
    # Since 2 out of 4 blocks are selected, exactly 8 out of 16 tokens in each row should be pruned
    # At least some positions should have attention weights exactly equal to 0.0
    zero_weights = (attn_weights < 1e-6)
    assert zero_weights.any(), "Block sparsity failed: no weights were masked to zero"


def test_causal_shape_and_causality():
    cfg = make_config(hidden_size=64, num_heads=4, is_decoder=True, model_type="gpt2", block_size=4, sparsity=0.5)
    attn = XAttention(cfg, pretrained_attn=DummyAttn(64), block_size=4, sparsity=0.5)
    attn.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)

    # 1. Shape check
    out, _ = attn(x)
    assert out.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs"

    # 2. Strict Causality check:
    # Modifying future tokens (index > l_probe) must not affect output at position l_probe
    l_probe = 5
    x_mod = x.clone()
    x_mod[:, l_probe + 1 :, :] = torch.randn_like(x_mod[:, l_probe + 1 :, :]) * 5.0 + 10.0

    out_orig, _ = attn(x)
    out_mod, _ = attn(x_mod)

    diff = torch.max(torch.abs(out_orig[:, : l_probe + 1, :] - out_mod[:, : l_probe + 1, :]))
    assert diff.item() < 1e-5, f"Causality violated! Max diff at probe pos: {diff.item()}"


def test_bidirectional_shape_and_centering():
    cfg = make_config(hidden_size=64, num_heads=4, is_decoder=False, model_type="bert", block_size=4, sparsity=0.5)
    attn = XAttention(cfg, pretrained_attn=DummyAttn(64), block_size=4, sparsity=0.5)
    attn.eval()

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L)
    mask[:, -4:] = 0  # Mask out last four tokens

    out, _ = attn(x, attention_mask=mask)
    assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Bidirectional output contains NaNs"


def test_rope_and_even_odd_dim_safety():
    for d in [16, 17, 63, 64]:
        t = torch.randn(2, 4, 8, d)
        rotated = _rotate_half(t)
        assert rotated.shape == t.shape, f"Odd dim {d} failed shape preservation in _rotate_half"

    head_dim_odd = 15
    q = torch.randn(2, 8, 4, head_dim_odd)
    k = torch.randn(2, 8, 4, head_dim_odd)
    cos = torch.randn(8, head_dim_odd)
    sin = torch.randn(8, head_dim_odd)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.isnan(q_rot).any()


def test_lora_adapters_and_freezing():
    cfg = make_config(hidden_size=64, num_heads=4)
    dummy = DummyAttn(64)
    dummy.q_proj.lora_A = nn.Parameter(torch.randn(16, 64))
    dummy.q_proj.lora_B = nn.Parameter(torch.randn(64, 16))
    dummy.q_proj.lora_scaling = 2.0

    attn = XAttention(cfg, pretrained_attn=dummy)

    assert attn.query.weight.requires_grad is False
    assert attn.query.lora_A.requires_grad is True
    assert attn.query.lora_B.requires_grad is True
    assert attn.query.lora_scaling == 2.0
    assert torch.allclose(attn.query.lora_A, dummy.q_proj.lora_A)


def test_head_mask_broadcasting():
    cfg = make_config(hidden_size=64, num_heads=4)
    attn = XAttention(cfg, pretrained_attn=DummyAttn(64))
    attn.eval()
    B, L, D, H = 2, 8, 64, 4
    x = torch.randn(B, L, D)

    # 1D: [H]
    hm1 = torch.ones(H)
    out1, _ = attn(x, head_mask=hm1)
    assert out1.shape == (B, L, D)

    # 2D: [B, H]
    hm2 = torch.ones(B, H)
    out2, _ = attn(x, head_mask=hm2)
    assert out2.shape == (B, L, D)

    # 3D: [B, H, L]
    hm3 = torch.ones(B, H, L)
    out3, _ = attn(x, head_mask=hm3)
    assert out3.shape == (B, L, D)

    # 4D: [B, H, L, L]
    hm4 = torch.ones(B, H, L, L)
    out4, _ = attn(x, head_mask=hm4)
    assert out4.shape == (B, L, D)


def test_inject_xattention():
    class DummyLayer(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.attention = DummyAttn(d)

    layer = DummyLayer(64)
    res = inject_xattention(layer=layer, old_attn=layer.attention)
    assert res is not None
    assert isinstance(layer.attention, XAttention)

    # When old_attn is None, should return None gracefully
    none_res = inject_xattention(layer=layer, old_attn=None)
    assert none_res is None


def test_forward_never_returns_none():
    cfg = make_config(hidden_size=64, num_heads=4)
    attn = XAttention(cfg, pretrained_attn=DummyAttn(64))
    attn.eval()
    
    # Intentionally broken rotary embedding that raises an exception
    class BrokenRotary(nn.Module):
        def forward(self, *args, **kwargs):
            raise RuntimeError("Forced RoPE failure for resilience testing")

    attn.rotary_emb = BrokenRotary()
    x = torch.randn(2, 16, 64)
    out, attn_weights = attn(x)
    assert out is not None, "forward() must never return None even if RoPE fails"
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()


def test_state_dict_keys_clean():
    cfg = make_config(hidden_size=64, num_heads=4)
    attn = XAttention(cfg, pretrained_attn=DummyAttn(64))
    sd = attn.state_dict()
    # Aliases W_q, W_k, W_v, out should be properties and NOT present in state_dict
    for key in sd.keys():
        assert not key.startswith("W_q."), f"Duplicate key {key} found in state_dict"
        assert not key.startswith("W_k."), f"Duplicate key {key} found in state_dict"
        assert not key.startswith("W_v."), f"Duplicate key {key} found in state_dict"
        assert not key.startswith("out."), f"Duplicate key {key} found in state_dict"
    assert "query.weight" in sd
    assert "key.weight" in sd
    assert "value.weight" in sd
    assert "out_proj.weight" in sd
    assert "output_scale" in sd


def test_position_embeddings_shorter_than_q():
    cfg = make_config(hidden_size=64, num_heads=4)
    attn = XAttention(cfg, pretrained_attn=DummyAttn(64))
    attn.eval()
    B, L, D = 2, 32, 64
    x = torch.randn(B, L, D)
    # Cos and sin shorter than sequence length (8 < 32)
    cos = torch.randn(8, 16)
    sin = torch.randn(8, 16)
    out, _ = attn(x, position_embeddings=(cos, sin))
    assert out is not None
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()


def test_padding_masked_block_routing():
    cfg = make_config(hidden_size=64, num_heads=4, block_size=4, sparsity=0.5)
    attn = XAttention(cfg, pretrained_attn=DummyAttn(64), block_size=4, sparsity=0.5)
    attn.eval()
    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    # Mask out last 6 tokens
    mask = torch.ones(B, L)
    mask[:, 10:] = 0
    out, _ = attn(x, attention_mask=mask)
    assert out is not None
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()
    # Masked positions should be zeroed
    assert (out[:, 10:, :] == 0.0).all()


def test_qwen_and_distilbert_architectures():
    # 1. DistilBERT architecture simulation
    class DummyDistilBertAttn(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.q_lin = nn.Linear(d, d)
            self.k_lin = nn.Linear(d, d)
            self.v_lin = nn.Linear(d, d)
            self.out_lin = nn.Linear(d, d)
            self.n_heads = 4
            self.dim = d

    class DummyDistilBertBlock(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.attention = DummyDistilBertAttn(d)

    dbert_block = DummyDistilBertBlock(64)
    injected_dbert = inject_xattention(layer=dbert_block, old_attn=dbert_block.attention)
    assert injected_dbert is not None
    assert isinstance(dbert_block.attention, XAttention)
    x_db = torch.randn(2, 12, 64)
    out_db, _ = dbert_block.attention(x_db)
    assert out_db.shape == (2, 12, 64)

    # 2. Qwen3.5 architecture simulation with GQA and RoPE
    class DummyQwenAttn(nn.Module):
        def __init__(self, d=64, num_heads=4, num_kv_heads=2):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d // 2)
            self.v_proj = nn.Linear(d, d // 2)
            self.o_proj = nn.Linear(d, d)
            self.num_heads = num_heads
            self.num_key_value_heads = num_kv_heads
            self.hidden_size = d
            self.head_dim = d // num_heads

    qwen_cfg = types.SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        is_decoder=True,
        model_type="qwen3_5",
        dropout=0.0,
        lora_rank=16,
    )
    qwen_attn_dummy = DummyQwenAttn(64, num_heads=4, num_kv_heads=2)
    qwen_xattn = XAttention(qwen_cfg, pretrained_attn=qwen_attn_dummy, block_size=4, sparsity=0.5)
    assert qwen_xattn.num_kv_heads == 2
    assert qwen_xattn.kv_repeat == 2

    x_qw = torch.randn(2, 16, 64)
    cos = torch.randn(16, 16)
    sin = torch.randn(16, 16)
    out_qw, _ = qwen_xattn(x_qw, position_embeddings=(cos, sin))
    assert out_qw.shape == (2, 16, 64)
    assert not torch.isnan(out_qw).any()


