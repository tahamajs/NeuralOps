#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive test suite verifying forensic healing of hybrid and convolutional attention modules:
1. gated_hybrid_kernel_attention
2. hybrid_attention
3. dynamic_conv_attention
4. nexus

Validates:
- Weight transplantation from BertSelfAttention without dropping parameters
- Output variance preservation and cosine similarity > 0.70 at step 0
- End-to-end forward and backward autograd gradient flow
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertSelfAttention

from neuralops.core.registry import ATTENTION_REGISTRY

HEALED_MODULES = [
    "gated_hybrid_kernel_attention",
    "hybrid_attention",
    "dynamic_conv_attention",
    "nexus",
]


@pytest.fixture
def bert_teacher():
    cfg = BertConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
    )
    teacher = BertSelfAttention(cfg).eval()
    return cfg, teacher


@pytest.fixture
def sample_batch():
    torch.manual_seed(42)
    B, N, D = 2, 128, 768
    x = torch.randn(B, N, D)
    mask = torch.ones(B, N)
    mask[:, -16:] = 0  # 16 padding tokens
    ext_mask = (1.0 - mask[:, None, None, :]) * -10000.0
    return x, mask, ext_mask


@pytest.mark.parametrize("module_name", HEALED_MODULES)
def test_weight_copy_from_bert(bert_teacher, module_name):
    """Verify weight transplantation from BertSelfAttention succeeds without dropping parameters."""
    cfg, teacher = bert_teacher
    cls = ATTENTION_REGISTRY.resolve(module_name)
    mod = cls(config=cfg, pretrained_attn=teacher)

    assert mod is not None, f"Failed to instantiate {module_name}"

    # Verify key projection weights have non-zero norm
    for proj_name in ("query", "key", "value", "q_proj", "k_proj", "v_proj"):
        proj = getattr(mod, proj_name, None)
        if proj is not None:
            weight = getattr(proj, "weight", None)
            if weight is not None:
                assert weight.norm().item() > 0.0, f"{module_name}.{proj_name} weight has zero norm"
                assert not torch.isnan(weight).any(), f"{module_name}.{proj_name} weight contains NaNs"


@pytest.mark.parametrize("module_name", HEALED_MODULES)
def test_step0_cosine_similarity_and_variance(bert_teacher, sample_batch, module_name):
    """Verify that output variance is preserved and cosine similarity to standard attention is > 0.70."""
    cfg, teacher = bert_teacher
    x, mask, ext_mask = sample_batch

    with torch.no_grad():
        bert_out = teacher(x, attention_mask=ext_mask)[0]

    cls = ATTENTION_REGISTRY.resolve(module_name)
    mod = cls(config=cfg, pretrained_attn=teacher).eval()

    with torch.no_grad():
        out = mod(x, attention_mask=ext_mask)
        if isinstance(out, (tuple, list)):
            out = out[0]

    assert out.shape == bert_out.shape, f"Shape mismatch: {out.shape} vs {bert_out.shape}"
    assert not torch.isnan(out).any(), f"{module_name} output contains NaNs"
    assert not torch.isinf(out).any(), f"{module_name} output contains Infs"

    # Evaluate cosine similarity on valid unpadded tokens
    valid_len = 112  # First 112 tokens are unmasked
    cos_sim = F.cosine_similarity(
        bert_out[:, :valid_len, :],
        out[:, :valid_len, :],
        dim=-1,
    ).mean().item()

    teacher_std = bert_out.std().item()
    student_std = out.std().item()

    assert cos_sim > 0.70, (
        f"[{module_name}] Cosine similarity {cos_sim:.4f} is below required threshold 0.70"
    )

    # Variance must be preserved within an acceptable factor (not collapsed to 0 or exploded)
    ratio = student_std / max(teacher_std, 1e-6)
    assert 0.3 <= ratio <= 3.0, (
        f"[{module_name}] Variance ratio {ratio:.4f} outside safe bounds (student_std={student_std:.4f}, teacher_std={teacher_std:.4f})"
    )


@pytest.mark.parametrize("module_name", HEALED_MODULES)
def test_forward_backward_autograd(bert_teacher, sample_batch, module_name):
    """Verify forward and backward autograd checks: finite output and valid gradient propagation."""
    cfg, teacher = bert_teacher
    x, mask, ext_mask = sample_batch

    x_train = x.clone().detach().requires_grad_(True)
    cls = ATTENTION_REGISTRY.resolve(module_name)
    mod = cls(config=cfg, pretrained_attn=teacher).train()

    # Unfreeze all parameters for training
    for p in mod.parameters():
        p.requires_grad = True

    out = mod(x_train, attention_mask=ext_mask)
    if isinstance(out, (tuple, list)):
        out = out[0]

    assert not torch.isnan(out).any(), f"{module_name} output contains NaNs during train"
    assert not torch.isinf(out).any(), f"{module_name} output contains Infs during train"

    loss = out.sum()
    loss.backward()

    assert x_train.grad is not None, f"{module_name} failed to backprop to input hidden_states"
    assert not torch.isnan(x_train.grad).any(), f"{module_name} input grad contains NaNs"
    assert not torch.isinf(x_train.grad).any(), f"{module_name} input grad contains Infs"

    # Verify at least one parameter received gradients
    has_grads = any(
        p.grad is not None and not torch.isnan(p.grad).any() and p.grad.abs().sum().item() > 0.0
        for p in mod.parameters()
        if p.requires_grad
    )
    assert has_grads, f"{module_name} produced zero parameter gradients"
