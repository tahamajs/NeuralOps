#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for LoRA, DoRA, and QLoRA fine-tuning methods."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.trainer.methods.dora import DoRALinear, apply_dora, merge_dora_into_linear
from neuralops.trainer.methods.lora import DisLoRALinear, LoRALinear, apply_lora
from neuralops.trainer.methods.qlora import QLoRALinear, QuantizedLinear, apply_qlora
from neuralops.trainer.methods.registry import get_method


class TestLoRAVariants:
    """Verifies LoRA adapter architectures and gradient flow."""

    def test_lora_linear_identity_init(self):
        torch.manual_seed(42)
        base = nn.Linear(32, 16, bias=True)
        lora = LoRALinear(base, r=4, alpha=8.0, dropout=0.0)

        x = torch.randn(2, 32)
        base_out = base(x)
        lora_out = lora(x)

        # LoRA_B is zero-initialized, so initial forward matches base linear
        assert torch.allclose(base_out, lora_out, atol=1e-5)

    def test_lora_gradient_isolation(self):
        base = nn.Linear(32, 16)
        lora = LoRALinear(base, r=4, alpha=8.0)
        x = torch.randn(2, 32)
        out = lora(x)
        loss = out.sum()
        loss.backward()

        # Base linear parameters must stay frozen
        assert not base.weight.requires_grad
        assert lora.lora_A.weight.grad is not None
        assert lora.lora_B.weight.grad is not None


class TestDoRA:
    """Verifies Weight-Decomposed Low-Rank Adaptation (DoRA)."""

    def test_dora_initialization_and_forward(self):
        torch.manual_seed(42)
        base = nn.Linear(32, 16, bias=True)
        dora = DoRALinear(base, r=4, alpha=8.0)

        # Magnitude parameter m initialized to Frobenius norm of base weight rows
        expected_m = base.weight.norm(p=2, dim=1)
        assert torch.allclose(dora.m, expected_m, atol=1e-5)

        x = torch.randn(2, 32)
        base_out = base(x)
        dora_out = dora(x)
        assert torch.allclose(base_out, dora_out, atol=1e-4)

    def test_dora_weight_merge(self):
        torch.manual_seed(42)
        base = nn.Linear(32, 16, bias=False)
        dora = DoRALinear(base, r=4, alpha=8.0)

        # Perturb B to simulate training update
        dora.lora_B.weight.data.normal_(0.0, 0.1)

        x = torch.randn(2, 32)
        pre_merge_out = dora(x)

        merged_linear = merge_dora_into_linear(dora)
        post_merge_out = merged_linear(x)

        assert torch.allclose(pre_merge_out, post_merge_out, atol=1e-5)
        assert dora.r == 0


class TestQLoRA:
    """Verifies Quantized Linear and QLoRA adapter functionality."""

    def test_quantized_linear_roundtrip(self):
        torch.manual_seed(42)
        base = nn.Linear(32, 16, bias=True)
        qlin = QuantizedLinear.from_linear(base, bits=4, block_size=16)

        x = torch.randn(2, 32)
        base_out = base(x)
        q_out = qlin(x)

        # Quantization should closely approximate floating-point output
        cos_sim = nn.functional.cosine_similarity(base_out.view(-1), q_out.view(-1), dim=0)
        assert cos_sim.item() > 0.95

    def test_qlora_linear_forward_and_backward(self):
        torch.manual_seed(42)
        base = nn.Linear(32, 16)
        qlora = QLoRALinear.from_linear(base, rank=4, alpha=8.0, bits=4)

        x = torch.randn(2, 32)
        out = qlora(x)
        assert out.shape == (2, 16)

        loss = out.sum()
        loss.backward()
        assert qlora.lora_A.weight.grad is not None
        assert qlora.lora_B.weight.grad is not None


class TestAdapterMethodApplication:
    """Verifies model-level injection for lora, dora, and qlora."""

    def test_apply_adapters_on_dummy_transformer(self):
        class DummyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(32, 32)
                self.k_proj = nn.Linear(32, 32)
                self.v_proj = nn.Linear(32, 32)
                self.out_proj = nn.Linear(32, 32)

        from types import SimpleNamespace

        # Test apply_lora
        m1 = DummyBlock()
        cfg_lora = SimpleNamespace(lora_r=4, lora_alpha=8.0, lora_dropout=0.0)
        apply_lora(m1, cfg_lora)
        assert isinstance(m1.q_proj, LoRALinear)

        # Test apply_dora
        m2 = DummyBlock()
        cfg_dora = SimpleNamespace(lora_r=4, lora_alpha=8.0, lora_dropout=0.0)
        apply_dora(m2, cfg_dora)
        assert isinstance(m2.q_proj, DoRALinear)

        # Test apply_qlora
        m3 = DummyBlock()
        cfg_qlora = SimpleNamespace(lora_r=4, lora_alpha=8.0, lora_dropout=0.0, bits=4)
        apply_qlora(m3, cfg_qlora)
        assert isinstance(m3.q_proj, QLoRALinear)
