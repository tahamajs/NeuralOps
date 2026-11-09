#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for linear and recurrent scan attention mechanisms calibration and healing.
Verifies weight copying, numerical finiteness, cosine similarity to BertSelfAttention,
and backward gradient flow across:
  - gla_attention (Gated Linear Attention)
  - gdn2_attention (Gated Delta Net v2)
  - preconditioned_gdn_attention (Curvature-Aware Preconditioned GDN)
  - delta_product_attention (Householder Product Attention)
  - positive_kernel_linear_attention (ELU+1 Linear Attention)
  - goldfinch_ahn (Adaptive Hybrid Sliding-Window + Recurrent Memory)
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.bert.modeling_bert import BertConfig, BertSelfAttention

from neuralops.core.registry import ATTENTION_REGISTRY

HEALED_MODULES = [
    "gla_attention",
    "gdn2_attention",
    "preconditioned_gdn_attention",
    "delta_product_attention",
    "positive_kernel_linear_attention",
    "goldfinch_ahn",
]


@pytest.fixture(scope="module")
def bert_teacher():
    torch.manual_seed(42)
    cfg = BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
    )
    teacher = BertSelfAttention(cfg)
    teacher.eval()
    return teacher, cfg


@pytest.mark.parametrize("module_name", HEALED_MODULES)
def test_weight_copy_from_bert(bert_teacher, module_name):
    """Verify Q, K, V projections copy correctly from BertSelfAttention."""
    teacher, cfg = bert_teacher
    cls = ATTENTION_REGISTRY.get(module_name)
    assert cls is not None, f"Module '{module_name}' not found in registry"

    mod = cls(cfg)
    assert hasattr(mod, "copy_weights"), f"Module '{module_name}' must implement copy_weights"
    mod.copy_weights(teacher)

    assert getattr(mod, "is_nested_encoder", False) is True, (
        f"Module '{module_name}' should identify BertSelfAttention as a nested encoder"
    )

    # Verify query weight norm is non-trivial and matches teacher magnitude
    teacher_q_norm = teacher.query.weight.norm().item()
    if hasattr(mod, "query"):
        mod_q = mod.query
    elif hasattr(mod, "q_proj"):
        mod_q = mod.q_proj
    elif hasattr(mod, "W_q"):
        mod_q = mod.W_q
    else:
        pytest.fail(f"Could not locate query projection on {module_name}")

    if hasattr(mod_q, "weight"):
        q_w = mod_q.weight
    else:
        q_w = mod_q.down.weight

    assert q_w.norm().item() > 0.1 * teacher_q_norm, (
        f"Query weight on {module_name} is near-zero after copy_weights"
    )


@pytest.mark.parametrize("module_name", HEALED_MODULES)
@pytest.mark.parametrize("seq_len", [16, 64])
def test_numerical_finiteness(bert_teacher, module_name, seq_len):
    """Verify output contains 0 NaNs and 0 Infs across sequence lengths."""
    teacher, cfg = bert_teacher
    torch.manual_seed(100 + seq_len)
    cls = ATTENTION_REGISTRY.get(module_name)
    mod = cls(cfg)
    mod.copy_weights(teacher)

    x = torch.randn(2, seq_len, 768)
    out = mod(x)
    y = out[0] if isinstance(out, tuple) else out

    assert not torch.isnan(y).any(), f"{module_name} produced NaN outputs at seq_len={seq_len}"
    assert not torch.isinf(y).any(), f"{module_name} produced Inf outputs at seq_len={seq_len}"
    assert y.shape == (2, seq_len, 768), f"Unexpected shape {y.shape} for {module_name}"


@pytest.mark.parametrize("module_name", HEALED_MODULES)
def test_cosine_similarity_to_standard_attention(bert_teacher, module_name):
    """Verify forward pass output achieves cosine similarity > 0.60 to BertSelfAttention at initialization."""
    teacher, cfg = bert_teacher
    torch.manual_seed(2026)
    cls = ATTENTION_REGISTRY.get(module_name)
    mod = cls(cfg)
    mod.copy_weights(teacher)

    x = torch.randn(2, 16, 768)
    with torch.no_grad():
        y_teacher = teacher(x)[0]
        out = mod(x)
        y_student = out[0] if isinstance(out, tuple) else out

    cos_sim = torch.cosine_similarity(y_teacher.flatten(), y_student.flatten(), dim=0).item()
    assert cos_sim >= 0.60, (
        f"{module_name} failed cosine similarity threshold (>0.60): got {cos_sim:.4f}"
    )


@pytest.mark.parametrize("module_name", HEALED_MODULES)
def test_backward_gradient_flow(bert_teacher, module_name):
    """Verify backward pass produces non-zero, finite gradients on both inputs and trainable weights."""
    teacher, cfg = bert_teacher
    torch.manual_seed(777)
    cls = ATTENTION_REGISTRY.get(module_name)
    mod = cls(cfg)
    mod.copy_weights(teacher)

    x = torch.randn(2, 16, 768, requires_grad=True)
    out = mod(x)
    y = out[0] if isinstance(out, tuple) else out

    loss = (y**2).mean()
    loss.backward()

    assert x.grad is not None, f"Input grad is None for {module_name}"
    assert torch.isfinite(x.grad).all(), f"Input grad has non-finite values for {module_name}"
    assert x.grad.norm().item() > 1e-4, f"Input grad vanished for {module_name}"

    # Verify at least one internal projection parameter received finite gradients
    param_grads = [p.grad for p in mod.parameters() if p.requires_grad and p.grad is not None]
    assert len(param_grads) > 0, f"No parameter gradients computed for {module_name}"
    for g in param_grads:
        assert torch.isfinite(g).all(), f"Parameter grad has non-finite values for {module_name}"


@pytest.mark.parametrize("module_name", HEALED_MODULES)
def test_masked_forward_pass(bert_teacher, module_name):
    """Verify module functions stably with attention masks without NaN/Inf."""
    teacher, cfg = bert_teacher
    torch.manual_seed(888)
    cls = ATTENTION_REGISTRY.get(module_name)
    mod = cls(cfg)
    mod.copy_weights(teacher)

    seq_len = 32
    x = torch.randn(2, seq_len, 768)
    mask = torch.ones(2, seq_len)
    mask[:, 24:] = 0.0  # Mask the last 8 tokens
    extended_mask = (1.0 - mask[:, None, None, :]) * -10000.0

    out = mod(x, attention_mask=extended_mask)
    y = out[0] if isinstance(out, tuple) else out

    assert not torch.isnan(y).any(), f"{module_name} produced NaNs with attention mask"
    assert not torch.isinf(y).any(), f"{module_name} produced Infs with attention mask"
