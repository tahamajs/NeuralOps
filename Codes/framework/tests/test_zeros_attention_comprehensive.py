import pytest
import torch
import torch.nn as nn
import types

from framework.attentions.zeros_attention import (
    ZeroSAttention,
    inject_zeros_attention,
    _rotate_half,
    apply_rotary_pos_emb,
    LoRALinear,
)


def make_config(hidden_size=64, num_heads=4, is_decoder=False, model_type="bert"):
    return types.SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=hidden_size // num_heads,
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
        self.num_heads = 4


def test_causal_shape_and_causality():
    cfg = make_config(hidden_size=64, num_heads=4, is_decoder=True, model_type="gpt2")
    attn = ZeroSAttention(cfg, pretrained_attn=DummyAttn(64))
    attn.eval()

    B, L, D = 2, 8, 64
    x = torch.randn(B, L, D)

    # 1. Shape check
    out, _ = attn(x)
    assert out.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs"

    # 2. Strict Causality check:
    # Modifying future tokens (e.g. at index > l) must not affect output at position l
    l_probe = 3
    x_mod = x.clone()
    # Change all tokens strictly after l_probe
    x_mod[:, l_probe + 1 :, :] = torch.randn_like(x_mod[:, l_probe + 1 :, :]) * 5.0 + 10.0

    out_orig, _ = attn(x)
    out_mod, _ = attn(x_mod)

    # Positions 0 to l_probe must be identical within numerical precision
    diff = torch.max(torch.abs(out_orig[:, : l_probe + 1, :] - out_mod[:, : l_probe + 1, :]))
    assert diff.item() < 1e-5, f"Causality violated! Max diff at probe pos: {diff.item()}"


def test_bidirectional_shape_and_centering():
    cfg = make_config(hidden_size=64, num_heads=4, is_decoder=False, model_type="bert")
    attn = ZeroSAttention(cfg, pretrained_attn=DummyAttn(64))
    attn.eval()

    B, L, D = 2, 8, 64
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L)
    mask[:, -2:] = 0  # Mask out last two tokens

    out, _ = attn(x, attention_mask=mask)
    assert out.shape == (B, L, D), f"Expected {(B, L, D)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Bidirectional output contains NaNs"


def test_rope_and_even_odd_dim_safety():
    # Test _rotate_half on both even and odd dimensions
    for d in [16, 17, 63, 64]:
        t = torch.randn(2, 4, 8, d)
        rotated = _rotate_half(t)
        assert rotated.shape == t.shape, f"Odd dim {d} failed shape preservation in _rotate_half"

    # Test apply_rotary_pos_emb with odd dimension
    head_dim_odd = 15
    q = torch.randn(2, 8, 4, head_dim_odd)
    k = torch.randn(2, 8, 4, head_dim_odd)
    cos = torch.randn(8, head_dim_odd)
    sin = torch.randn(8, head_dim_odd)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.isnan(q_rot).any()


def test_inject_with_none_old_attn_returns_none():
    layer = nn.Module()
    layer.attention = DummyAttn(64)
    res = inject_zeros_attention(layer=layer, old_attn=None)
    assert res is None, "inject_zeros_attention must return None when old_attn is None"


def test_lora_adapter_preservation():
    # Construct source module with LoRA adapters
    src = DummyAttn(64)
    # Wrap q_proj with mock LoRA
    src.q_proj.lora_A = nn.Parameter(torch.randn(16, 64) * 0.5)
    src.q_proj.lora_B = nn.Parameter(torch.randn(64, 16) * 0.5)

    cfg = make_config(hidden_size=64, num_heads=4)
    zeros = ZeroSAttention(cfg, pretrained_attn=src)

    # Check that lora weights were copied and remain trainable
    assert zeros.W_q.lora_A is not None
    assert zeros.W_q.lora_B is not None
    assert torch.allclose(zeros.W_q.lora_A, src.q_proj.lora_A)
    assert torch.allclose(zeros.W_q.lora_B, src.q_proj.lora_B)
    assert zeros.W_q.lora_A.requires_grad is True
    assert zeros.W_q.lora_B.requires_grad is True
    assert zeros.W_q.weight.requires_grad is False


def test_head_mask_broadcasting():
    cfg = make_config(hidden_size=64, num_heads=4, is_decoder=False)
    attn = ZeroSAttention(cfg, pretrained_attn=DummyAttn(64))
    attn.eval()

    B, L, D, H = 2, 8, 64, 4
    x = torch.randn(B, L, D)

    # 1D mask [H]
    m1 = torch.ones(H)
    out1, _ = attn(x, head_mask=m1)
    assert out1.shape == (B, L, D)

    # 2D mask [B, H]
    m2 = torch.ones(B, H)
    out2, _ = attn(x, head_mask=m2)
    assert out2.shape == (B, L, D)

    # 3D mask [B, H, L]
    m3 = torch.ones(B, H, L)
    out3, _ = attn(x, head_mask=m3)
    assert out3.shape == (B, L, D)

    # 4D mask [B, H, L, L]
    m4 = torch.ones(B, H, L, L)
    out4, _ = attn(x, head_mask=m4)
    assert out4.shape == (B, L, D)


def test_four_architectures_forward():
    try:
        from test_four_architectures import get_all_four_models
        models = get_all_four_models(d=64, h=4)
    except ImportError:
        pytest.skip("test_four_architectures fixture not available in environment")
        return
    x = torch.randn(2, 8, 64)

    for arch_name, model in models.items():
        # Inject ZeroSAttention into the model self-attention
        old_attn = (
            getattr(model, "attention", None)
            or getattr(model, "attn", None)
            or getattr(model, "self_attn", None)
        )
        assert old_attn is not None, f"Could not find attention module in {arch_name}"

        injected = inject_zeros_attention(
            layer=model,
            model_config=model.config,
            old_attn=old_attn,
        )
        assert injected is not None, f"Injection failed for {arch_name}"

        # Forward check
        out = model(x)
        assert out.shape == (2, 8, 2), f"{arch_name} output shape mismatch: {out.shape}"
        assert not torch.isnan(out).any(), f"{arch_name} output contains NaNs"
        assert not torch.isinf(out).any(), f"{arch_name} output contains Infs"

        # Backward check
        loss = out.sum()
        loss.backward()
        grad_params = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_params > 0, f"{arch_name} computed zero gradients"
