import pytest
import torch
import torch.nn as nn
from neuralops.attentions.svd_linear_attention import (
    SVDLinear, SVDLinearAttention, apply_rotary_pos_emb,
    inject_svd_linear_attention)


class DummyConfig:
    def __init__(self, hidden_size=64, num_heads=4, head_dim=16, is_decoder=False):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_heads
        self.is_decoder = is_decoder
        self.dropout = 0.0
        self.attention_probs_dropout_prob = 0.0


def test_svd_linear_attention_forward():
    cfg = DummyConfig(hidden_size=64, num_heads=4, head_dim=16)
    attn = SVDLinearAttention(config=cfg, rank=16, lora_rank=8)
    x = torch.randn(2, 8, 64)
    out, attn_weights = attn(x)
    assert out.shape == (2, 8, 64), f"Expected (2, 8, 64), got {out.shape}"
    assert attn_weights is None


def test_svd_linear_attention_copy_weights_and_lora():
    cfg = DummyConfig(hidden_size=64, num_heads=4, head_dim=16)
    
    # Mock source layer like RoBERTa SelfAttention
    class MockInnerAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(64, 64)
            self.key = nn.Linear(64, 64)
            self.value = nn.Linear(64, 64)
            self.dense = nn.Linear(64, 64)

    mock_src = MockInnerAttn()
    attn = SVDLinearAttention(config=cfg, old_attn=mock_src, rank=16, freeze_svd=True, lora_rank=8)

    # Check SVD weights are frozen and LoRA is trainable
    assert attn.q_proj.U.requires_grad is False
    assert attn.q_proj.V.requires_grad is False
    assert attn.q_proj.lora_A.requires_grad is True
    assert attn.q_proj.lora_B.requires_grad is True

    x = torch.randn(2, 8, 64, requires_grad=True)
    out, _ = attn(x)
    loss = out.sum()
    loss.backward()

    assert attn.q_proj.lora_A.grad is not None
    assert attn.q_proj.lora_B.grad is not None


def test_svd_linear_attention_gpt2_conv1d():
    # Mock Conv1D from GPT-2
    class MockConv1D(nn.Module):
        def __init__(self, nf, nx):
            super().__init__()
            self.nf = nf
            self.weight = nn.Parameter(torch.randn(nx, nf))
            self.bias = nn.Parameter(torch.zeros(nf))

        def forward(self, x):
            return torch.matmul(x, self.weight) + self.bias

    class MockGPT2Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = MockConv1D(192, 64)  # 3 * 64
            self.c_proj = MockConv1D(64, 64)

    mock_gpt2 = MockGPT2Attn()
    cfg = DummyConfig(hidden_size=64, num_heads=4, head_dim=16, is_decoder=True)
    attn = SVDLinearAttention(config=cfg, old_attn=mock_gpt2, rank=16)

    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)


def test_odd_dim_rope_safety():
    # Odd dimension head_dim = 17
    B, H, N, D = 2, 4, 8, 17
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    cos = torch.randn(N, D)
    sin = torch.randn(N, D)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == (B, H, N, D)
    assert k_rot.shape == (B, H, N, D)
    assert not torch.isnan(q_rot).any()
    assert not torch.isnan(k_rot).any()


def test_dual_whole_model_injection():
    class DummyTransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

    class DummyDistilBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = DummyConfig(hidden_size=64, num_heads=4, head_dim=16)
            self.distilbert = nn.Module()
            self.distilbert.transformer = nn.Module()
            self.distilbert.transformer.layer = nn.ModuleList([DummyTransformerLayer() for _ in range(3)])

    model = DummyDistilBert()
    injected_model = inject_svd_linear_attention(model=model, rank=16)
    assert injected_model is not None

    for lyr in model.distilbert.transformer.layer:
        assert isinstance(lyr.attention, SVDLinearAttention)
        x = torch.randn(2, 8, 64)
        out, _ = lyr.attention(x)
        assert out.shape == (2, 8, 64)
