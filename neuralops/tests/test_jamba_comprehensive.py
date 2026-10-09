from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.jamba import (Jamba2Block, LoRALinear,
                                        apply_rotary_pos_emb, inject_jamba)


def test_jamba_lora_config_independence():
    # 1. config=None, comp_config present
    comp_cfg = SimpleNamespace(lora_rank=8, lora_alpha=16.0)
    p1 = Jamba2Block(config=None, comp_config=comp_cfg, d_model=64, n_heads=4)
    assert p1.lora_rank == 8
    assert p1.lora_alpha == 16.0

    # 2. config=None, comp_config=None, kwargs override
    p2 = Jamba2Block(config=None, comp_config=None, d_model=64, n_heads=4, lora_rank=4, lora_alpha=8.0)
    assert p2.lora_rank == 4
    assert p2.lora_alpha == 8.0

    # 3. config present
    cfg = SimpleNamespace(lora_rank=12, lora_alpha=24.0, num_attention_heads=4)
    p3 = Jamba2Block(config=cfg, d_model=64)
    assert p3.lora_rank == 12
    assert p3.lora_alpha == 24.0

def test_jamba_rotary_fallback():
    class MockRotaryEmb(nn.Module):
        def forward(self, x, seq_len=None):
            B, H, N, d = x.shape
            cos = torch.ones(B, 1, N, d)
            sin = torch.zeros(B, 1, N, d)
            return cos, sin

    class MockAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
            self.rotary_emb = MockRotaryEmb()

    old = MockAttn()
    p = Jamba2Block(d_model=64, n_heads=4, pretrained_attn=old)
    x = torch.randn(2, 8, 64)
    out, _ = p(x, position_embeddings=None)
    assert out.shape == (2, 8, 64)
    assert torch.isfinite(out).all()

def test_jamba_bidirectional_and_causal_shapes():
    x = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10)
    mask[:, 7:] = 0  # 3 padding tokens

    # Bidirectional (encoder)
    p_bidi = Jamba2Block(d_model=64, n_heads=4, is_causal=False, is_decoder=False, lora_rank=8)
    out_bidi, _ = p_bidi(x, attention_mask=mask)
    assert out_bidi.shape == (2, 10, 64)
    assert torch.isfinite(out_bidi).all()

    # Causal (decoder)
    p_causal = Jamba2Block(d_model=64, n_heads=4, is_causal=True, is_decoder=True, lora_rank=8)
    out_causal, _ = p_causal(x, attention_mask=mask)
    assert out_causal.shape == (2, 10, 64)
    assert torch.isfinite(out_causal).all()

def test_jamba_dimensions_768_and_1024():
    for d in (64, 768, 1024):
        p = Jamba2Block(d_model=d, n_heads=8 if d >= 64 else 4, lora_rank=8)
        x = torch.randn(1, 4, d)
        out, _ = p(x)
        assert out.shape == (1, 4, d)
        assert torch.isfinite(out).all()

def test_jamba_frobenius_norm_verification():
    class ZeroAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
            self.q_proj.weight.data.zero_()  # Zero norm

    old = ZeroAttn()
    with pytest.warns(UserWarning, match="norm too small"):
        block = Jamba2Block(d_model=64, n_heads=4, pretrained_attn=old)
    assert block is not None

def test_jamba_fallback_on_copy_weights_failure():
    class BrokenLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Module()
            self.attention.hidden_size = -1  # Invalid dimension causes Jamba2Block init to fail
    
    layer = BrokenLayer()
    res = inject_jamba(layer=layer)
    assert res is None  # Graceful fallback, no crash

def test_jamba_lora_adapters_freezing():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
    old = DummyAttn()
    p = Jamba2Block(d_model=64, n_heads=4, pretrained_attn=old, lora_rank=8)
    assert not p.q_lin.weight.requires_grad
    assert p.q_lin.lora_A.requires_grad
    assert p.q_lin.lora_B.requires_grad
    assert not p.out_lin.weight.requires_grad
    assert p.out_lin.lora_A.requires_grad
    assert p.out_lin.lora_B.requires_grad

def test_jamba_injection_all_4_architectures():
    class MockAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
            self.hidden_size = 64
            self.num_heads = 4

    # 1. DistilBERT clone
    class DistilBertClone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.attention = MockAttn()
            self.distilbert = nn.Module()
            self.distilbert.transformer = nn.Module()
            self.distilbert.transformer.layer = nn.ModuleList([layer])
    d_model = DistilBertClone()
    res_d = inject_jamba(model=d_model, lora_rank=8)
    assert res_d is not None
    assert isinstance(d_model.distilbert.transformer.layer[0].attention, Jamba2Block)

    # 2. RoBERTa clone
    class RobertaClone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.attention = nn.Module()
            layer.attention.self = MockAttn()
            self.roberta = nn.Module()
            self.roberta.encoder = nn.Module()
            self.roberta.encoder.layer = nn.ModuleList([layer])
    r_model = RobertaClone()
    res_r = inject_jamba(model=r_model, lora_rank=8)
    assert res_r is not None
    assert isinstance(r_model.roberta.encoder.layer[0].attention.self, Jamba2Block)

    # 3. GPT-2 clone
    class GPT2Clone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.attn = MockAttn()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([layer])
    g_model = GPT2Clone()
    res_g = inject_jamba(model=g_model, lora_rank=8)
    assert res_g is not None
    assert isinstance(g_model.transformer.h[0].attn, Jamba2Block)

    # 4. Qwen3.5 clone
    class QwenClone(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            layer.self_attn = MockAttn()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([layer])
    q_model = QwenClone()
    res_q = inject_jamba(model=q_model, lora_rank=8)
    assert res_q is not None
    assert isinstance(q_model.model.layers[0].self_attn, Jamba2Block)


# ===========================================================================
# Targeted Tests for Jamba Refactor Fixes
# ===========================================================================

def test_jamba_checkpointing_env_flag(monkeypatch):
    """Verify that JAMBA_CHECKPOINT environment variable controls checkpointing."""
    import os
    monkeypatch.delenv("JAMBA_CHECKPOINT", raising=False)
    block = Jamba2Block(d_model=64, n_heads=4, is_decoder=True)
    block.train()
    x = torch.randn(2, 4, 64)
    # Direct fast execution when disabled
    out, present = block(x)
    assert out.shape == (2, 4, 64)
    assert present is not None

    # Checkpoint execution when enabled
    monkeypatch.setenv("JAMBA_CHECKPOINT", "1")
    out_cp, present_cp = block(x)
    assert out_cp.shape == (2, 4, 64)
    assert present_cp is not None


def test_jamba_decoder_tuple_returns_gpt2():
    """Verify strictly aligned return tuples for GPT-2 format."""
    block = Jamba2Block(d_model=64, n_heads=4, is_decoder=True)
    block._return_format = "gpt2"
    x = torch.randn(2, 4, 64)

    # 1. output_attentions=False -> (out, present) 2-tuple
    res = block(x, output_attentions=False)
    assert len(res) == 2
    out, present = res
    assert out.shape == (2, 4, 64)
    assert present is not None

    # 2. output_attentions=True -> (out, present, attn_weights) 3-tuple
    res_attn = block(x, output_attentions=True)
    assert len(res_attn) == 3
    out_a, present_a, attn_w = res_attn
    assert out_a.shape == (2, 4, 64)
    assert present_a is not None
    assert attn_w is not None


def test_jamba_encoder_tuple_returns_bert():
    """Verify strictly aligned return tuples for BERT / encoder format."""
    block = Jamba2Block(d_model=64, n_heads=4, is_decoder=False)
    block._return_format = "bert"
    x = torch.randn(2, 4, 64)

    # 1. output_attentions=False -> (out,) 1-tuple
    res = block(x, output_attentions=False)
    assert len(res) == 1
    assert res[0].shape == (2, 4, 64)

    # 2. output_attentions=True -> (out, attn_weights) 2-tuple
    res_attn = block(x, output_attentions=True)
    assert len(res_attn) == 2
    out_a, attn_w = res_attn
    assert out_a.shape == (2, 4, 64)
    assert attn_w is not None


def test_jamba_sdpa_path_priority():
    """Verify SDPA path executes cleanly without crash for decoder with and without 2D mask."""
    block = Jamba2Block(d_model=64, n_heads=4, is_decoder=True)
    x = torch.randn(2, 8, 64)

    # Case 1: attention_mask is None
    out1, _ = block(x, attention_mask=None)
    assert out1.shape == (2, 8, 64)
    assert torch.isfinite(out1).all()

    # Case 2: attention_mask is 2D full valid (all 1s)
    mask_full = torch.ones(2, 8, dtype=torch.long)
    out2, _ = block(x, attention_mask=mask_full)
    assert out2.shape == (2, 8, 64)
    assert torch.isfinite(out2).all()

    # Case 3: attention_mask is 2D with padding
    mask_pad = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)
    out3, _ = block(x, attention_mask=mask_pad)
    assert out3.shape == (2, 8, 64)
    assert torch.isfinite(out3).all()


def test_jamba_aux_loss_training_accumulation():
    """Verify aux_loss accumulates during training and not during eval."""
    block = Jamba2Block(d_model=64, n_heads=4, n_experts=4)
    x = torch.randn(2, 4, 64)

    # Eval mode: aux_loss should remain 0
    block.eval()
    block.reset_aux_loss()
    block(x)
    assert block.aux_loss == 0.0

    # Train mode: aux_loss should be accumulated
    block.train()
    block.reset_aux_loss()
    block(x)
    assert block._aux_loss_count == 1
    assert block.aux_loss >= 0.0


def test_jamba_scale_parameter_dtype_device():
    """Verify lora_mamba_scale and lora_moe_scale match source_attn parameters."""
    class MockLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(64, 64, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(64, dtype=torch.float32))

    class MockAttnWithDtype(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = MockLinear()
            self.k_proj = MockLinear()
            self.v_proj = MockLinear()
            self.out_proj = MockLinear()

    old = MockAttnWithDtype()
    block = Jamba2Block(d_model=64, n_heads=4, pretrained_attn=old)
    assert block.lora_mamba_scale.dtype == old.q_proj.weight.dtype
    assert block.lora_moe_scale.dtype == old.q_proj.weight.dtype

