"""Comprehensive unit tests for native LoRA module."""

import os
import tempfile
import torch
import torch.nn as nn
import pytest

from framework.utils.lora import (
    LoRALinear,
    apply_lora,
    get_lora_trainable_parameters,
    save_lora_weights,
    load_lora_weights,
)


class TestLoRALinear:
    """Test suite for the standalone LoRALinear layer."""

    def test_initialization_and_forward_shape(self):
        base = nn.Linear(64, 128, bias=True)
        lora = LoRALinear(base, r=8, alpha=16.0, dropout=0.0)
        x = torch.randn(4, 16, 64)
        out = lora(x)
        assert out.shape == (4, 16, 128)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_zero_initialization_identity(self):
        """At step 0, LoRA output must be exactly identical to base linear output."""
        base = nn.Linear(64, 128, bias=True)
        lora = LoRALinear(base, r=8, alpha=16.0, dropout=0.0)
        x = torch.randn(4, 16, 64)
        base_out = base(x)
        lora_out = lora(x)
        assert torch.allclose(base_out, lora_out, atol=1e-6)

    def test_parameter_freezing(self):
        """Base weights must be frozen, only lora_A and lora_B trainable."""
        base = nn.Linear(64, 128, bias=True)
        lora = LoRALinear(base, r=8, alpha=16.0)
        assert not lora.base_layer.weight.requires_grad
        if lora.base_layer.bias is not None:
            assert not lora.base_layer.bias.requires_grad
        assert lora.lora_A.weight.requires_grad
        assert lora.lora_B.weight.requires_grad

    def test_gradient_flow_and_backprop(self):
        """Gradients must flow into lora_A and lora_B, but not base weights."""
        base = nn.Linear(32, 64, bias=True)
        lora = LoRALinear(base, r=4, alpha=8.0)
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = lora(x)
        loss = out.sum()
        loss.backward()

        assert lora.base_layer.weight.grad is None
        assert lora.lora_A.weight.grad is not None
        assert lora.lora_B.weight.grad is not None
        assert x.grad is not None

    def test_merge_and_unmerge(self):
        """Verify weights merge into base_layer and unmerge correctly."""
        base = nn.Linear(32, 32, bias=False)
        lora = LoRALinear(base, r=4, alpha=8.0, dropout=0.0)
        lora.eval()
        # Assign non-zero to lora_B to produce a delta
        lora.lora_B.weight.data.fill_(0.1)

        x = torch.randn(2, 32)
        out_unmerged = lora(x)

        lora.merge()
        assert lora.merged
        out_merged = lora(x)
        assert torch.allclose(out_unmerged, out_merged, atol=1e-5)

        lora.unmerge()
        assert not lora.merged
        out_reverted = lora(x)
        assert torch.allclose(out_unmerged, out_reverted, atol=1e-5)


class TestApplyLoRAOnModels:
    """Test suite for apply_lora on multi-layer transformer structures."""

    def test_apply_lora_to_synthetic_qwen_block(self):
        class MockQwenAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(896, 896, bias=False)
                self.k_proj = nn.Linear(896, 128, bias=False)
                self.v_proj = nn.Linear(896, 128, bias=False)
                self.o_proj = nn.Linear(896, 896, bias=False)

        class MockQwenMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(896, 4864, bias=False)
                self.up_proj = nn.Linear(896, 4864, bias=False)
                self.down_proj = nn.Linear(4864, 896, bias=False)

        class MockQwenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MockQwenAttention()
                self.mlp = MockQwenMLP()
                self.score = nn.Linear(896, 2, bias=False)

        model = MockQwenModel()
        model, count = apply_lora(model, r=16, alpha=32.0)

        # 4 attn + 3 mlp = 7 linear layers adapted
        assert count == 7
        assert isinstance(model.attn.q_proj, LoRALinear)
        assert isinstance(model.attn.k_proj, LoRALinear)
        assert isinstance(model.mlp.gate_proj, LoRALinear)
        assert not isinstance(model.score, LoRALinear)  # Score head must not be wrapped

        # Verify trainable parameters
        trainable = get_lora_trainable_parameters(model)
        assert len(trainable) > 0
        total_trainable = sum(p.numel() for p in trainable)
        total_base = sum(p.numel() for p in model.parameters())
        assert total_trainable < total_base  # Substantial parameter efficiency

    def test_apply_lora_to_distilbert_style(self):
        class MockDistilBertBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_lin = nn.Linear(768, 768)
                self.k_lin = nn.Linear(768, 768)
                self.v_lin = nn.Linear(768, 768)
                self.out_lin = nn.Linear(768, 768)
                self.classifier = nn.Linear(768, 2)

        model = MockDistilBertBlock()
        model, count = apply_lora(model, r=8, alpha=16.0)
        assert count == 4
        assert isinstance(model.q_lin, LoRALinear)
        assert not isinstance(model.classifier, LoRALinear)

    def test_save_and_load_checkpoint(self):
        base = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        model, count = apply_lora(base, r=4, alpha=8.0)
        tmp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()
        try:
            save_lora_weights(model, tmp_path, metadata={"steps": 50})
            assert os.path.exists(tmp_path)
            res = load_lora_weights(model, tmp_path)
            assert res["metadata"]["steps"] == 50
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
