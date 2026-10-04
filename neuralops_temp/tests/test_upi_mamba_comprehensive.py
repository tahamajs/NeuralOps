import types

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.upi_stuffed_mamba import (LoRALinear, UPIMambaBlock,
                                                    inject_upi_mamba)


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


def test_upi_mamba_copy_weights_all_4_architectures():
    d = 64
    cfg = make_config(hidden_size=d)

    # 1. DistilBERT
    class DummyDistilBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(d, d)
            self.k_lin = nn.Linear(d, d)
            self.v_lin = nn.Linear(d, d)
            self.out_lin = nn.Linear(d, d)

    m_distil = UPIMambaBlock(cfg, pretrained_attn=DummyDistilBert())
    x = torch.randn(2, 8, d)
    out1, _ = m_distil(x)
    assert out1.shape == (2, 8, d)

    # 2. RoBERTa nested
    class DummyRoberta(nn.Module):
        def __init__(self):
            super().__init__()
            self.self = nn.Module()
            self.self.query = nn.Linear(d, d)
            self.self.key = nn.Linear(d, d)
            self.self.value = nn.Linear(d, d)
            self.output = nn.Module()
            self.output.dense = nn.Linear(d, d)

    m_roberta = UPIMambaBlock(cfg, pretrained_attn=DummyRoberta())
    out2, _ = m_roberta(x)
    assert out2.shape == (2, 8, d)

    # 3. GPT-2 Conv1D
    class Conv1D(nn.Module):
        def __init__(self, nf, nx):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(nx, nf))
            self.bias = nn.Parameter(torch.zeros(nf))

    class DummyGPT2(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = Conv1D(3 * d, d)
            self.c_proj = Conv1D(d, d)

    m_gpt2 = UPIMambaBlock(cfg, pretrained_attn=DummyGPT2(), is_decoder=True)
    out3, _ = m_gpt2(x)
    assert out3.shape == (2, 8, d)

    # 4. Qwen GQA
    class DummyQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(d, d // 2)
            self.v_proj = nn.Linear(d, d // 2)
            self.o_proj = nn.Linear(d, d)

    m_qwen = UPIMambaBlock(cfg, pretrained_attn=DummyQwen())
    out4, _ = m_qwen(x)
    assert out4.shape == (2, 8, d)


def test_upi_mamba_is_selective():
    """Verify the SSM is truly selective: B_t, C_t depend on input, and a spike evolves state."""
    d = 64
    cfg = make_config(hidden_size=d, is_decoder=True, model_type="gpt2")
    m = UPIMambaBlock(cfg, pretrained_attn=DummyAttn(d), is_decoder=True)
    m.eval()

    L = 16
    x1 = torch.randn(1, L, d)
    x2 = x1.clone()
    # Inject large perturbation only at position 5
    x2[0, 5, :] += 10.0

    with torch.no_grad():
        out1, _ = m(x1)
        out2, _ = m(x2)

    # State evolution: outputs prior to position 5 should be identical (causal)
    diff_before = torch.norm(out1[0, :5] - out2[0, :5]).item()
    assert diff_before < 1e-4, f"Causality violated before spike: diff={diff_before}"

    # Output at and after position 5 must differ due to state evolution
    diff_at_and_after = torch.norm(out1[0, 5:] - out2[0, 5:]).item()
    assert diff_at_and_after > 1.0, f"Expected state to evolve after spike, got diff={diff_at_and_after}"

    # Verify B_t and C_t projections are input-dependent
    x_proj_out1 = m.x_proj(x1)
    _, B_t1, C_t1 = torch.split(x_proj_out1, [m.inner_dim, m.d_state, m.d_state], dim=-1)
    x_proj_out2 = m.x_proj(x2)
    _, B_t2, C_t2 = torch.split(x_proj_out2, [m.inner_dim, m.d_state, m.d_state], dim=-1)

    assert not torch.allclose(B_t1[0, 5], B_t2[0, 5]), "B_t must be input-dependent"
    assert not torch.allclose(C_t1[0, 5], C_t2[0, 5]), "C_t must be input-dependent"


