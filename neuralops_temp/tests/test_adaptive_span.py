#!/usr/bin/env python3
"""
Unit tests for AdaptiveSpanAttention module.
Verifies all 8 core properties:
1. Module instantiates with correct span initialization (span_logits = 4.0, sigmoid ~ 0.982).
2. Span regularization loss is non-negative and strictly positive for initialized spans.
3. Multi-module span loss aggregation: get_total_span_loss computes correct average across layers.
4. Forward pass produces expected shape [B, N, D] and finite outputs.
5. Softmax NaN safety: completely masked rows produce 0 attention without NaN or inf.
6. Gradient flow: span_logits receives non-zero gradients when span regularization loss is backpropagated.
7. Registry registration: 'adaptive_span' and 'adaptive_span_attention' are registered.
8. Supports causal, bidirectional, and GQA flags appropriately.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("."))

from neuralops.attentions.adaptive_span_attention import (
    AdaptiveSpanAttention, get_total_span_loss, inject_adaptive_span)
from neuralops.core.registry import ATTENTION_REGISTRY


def test_1_instantiation_and_span_init():
    """Verify AdaptiveSpanAttention initializes with span_logits=4.0 and current_span near max_span."""
    attn = AdaptiveSpanAttention(hidden_size=256, num_heads=4, max_span=64)
    assert hasattr(attn, "span_logits"), "Missing span_logits parameter"
    assert attn.span_logits.shape == (4,), f"Expected shape (4,), got {attn.span_logits.shape}"
    # sigmoid(4.0) ~ 0.9820
    expected_ratio = torch.sigmoid(torch.tensor(4.0)).item()
    expected_span = expected_ratio * 64
    assert torch.allclose(attn.current_span, torch.full((4,), expected_span), atol=1e-3), (
        f"Expected span ~ {expected_span}, got {attn.current_span}"
    )
    print("[PASS] Test 1: Instantiation and span initialization verified")


def test_2_span_regularization_loss_positive():
    """Verify get_span_regularization_loss returns a positive scalar loss."""
    attn = AdaptiveSpanAttention(hidden_size=256, num_heads=4, max_span=64)
    loss = attn.get_span_regularization_loss(lambda_span=0.01)
    assert isinstance(loss, torch.Tensor), "Expected torch.Tensor loss"
    assert loss.dim() == 0, f"Expected scalar loss, got dim {loss.dim()}"
    assert loss.item() > 0.0, f"Expected strictly positive loss, got {loss.item()}"
    # At logits=4.0, normalized span ~ 0.982, lambda=0.01 -> ~ 0.00982
    expected = 0.01 * torch.sigmoid(torch.tensor(4.0)).item()
    assert abs(loss.item() - expected) < 1e-4, f"Expected {expected:.5f}, got {loss.item():.5f}"
    print(f"[PASS] Test 2: Span regularization loss is strictly positive ({loss.item():.5f})")


def test_3_multi_module_span_loss_aggregation():
    """Verify get_total_span_loss aggregates across multiple layers in a model."""
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer0 = AdaptiveSpanAttention(hidden_size=128, num_heads=2, max_span=32)
            self.layer1 = AdaptiveSpanAttention(hidden_size=128, num_heads=2, max_span=32)
            # Set layer0 logits to 0.0 (sigmoid = 0.5) and layer1 to 4.0 (~0.982)
            self.layer0.span_logits.data.fill_(0.0)
            self.layer1.span_logits.data.fill_(4.0)

    model = ToyModel()
    total_loss = get_total_span_loss(model, lambda_span=0.02)
    # Expected: 0.02 * (0.5 + sigmoid(4.0)) / 2 = 0.01 * (0.5 + 0.98201) = 0.01482
    exp_0 = 0.02 * 0.5
    exp_1 = 0.02 * torch.sigmoid(torch.tensor(4.0)).item()
    expected = (exp_0 + exp_1) / 2.0
    assert abs(total_loss.item() - expected) < 1e-4, f"Expected {expected:.5f}, got {total_loss.item():.5f}"
    print(f"[PASS] Test 3: Multi-module span loss aggregation verified ({total_loss.item():.5f})")


def test_4_forward_pass_output_shape():
    """Verify forward pass returns expected output shape [B, N, D] with finite values."""
    attn = AdaptiveSpanAttention(hidden_size=128, num_heads=4, max_span=32)
    B, N, D = 2, 16, 128
    x = torch.randn(B, N, D)
    out, attn_weights = attn(x)
    assert out.shape == (B, N, D), f"Expected shape {(B, N, D)}, got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"
    print("[PASS] Test 4: Forward pass output shape and finite values verified")


def test_5_softmax_nan_safety_all_masked():
    """Verify fully masked rows do not result in NaNs in attention or output."""
    attn = AdaptiveSpanAttention(hidden_size=128, num_heads=4, max_span=32)
    B, N, D = 2, 8, 128
    x = torch.randn(B, N, D)
    # Mask out the last token completely for all queries
    mask = torch.zeros(B, 1, N, N)
    mask[:, :, :, -1] = -10000.0  # Entire key position -1 is masked out
    # Also mask an entire row (e.g. query row 0 has all keys masked)
    mask[0, 0, 0, :] = -10000.0

    out, attn_weights = attn(x, attention_mask=mask, output_attentions=True)
    assert torch.isfinite(out).all(), "Output has NaNs when rows/keys are fully masked"
    if attn_weights is not None:
        assert torch.isfinite(attn_weights).all(), "Attn weights have NaNs when fully masked"
        # The fully masked row should have zero attention weights
        assert (attn_weights[0, :, 0, :] == 0.0).all(), "Fully masked query row should have 0 attention"
    print("[PASS] Test 5: Softmax NaN safety on all-masked rows verified")


def test_6_gradient_flow_to_span_logits():
    """Verify span_logits receives non-zero gradients when span regularization loss is backpropagated."""
    attn = AdaptiveSpanAttention(hidden_size=128, num_heads=4, max_span=32)
    x = torch.randn(2, 16, 128)
    out, _ = attn(x)
    task_loss = out.sum()
    span_loss = attn.get_span_regularization_loss(lambda_span=0.05)
    total_loss = task_loss + span_loss
    total_loss.backward()

    assert attn.span_logits.grad is not None, "span_logits received no gradient"
    assert (attn.span_logits.grad != 0.0).all(), f"span_logits grad is zero: {attn.span_logits.grad}"
    # Span regularization penalizes large spans, so its gradient contribution should be positive
    assert (attn.span_logits.grad > 0.0).any(), "Gradient should push span_logits to shrink"
    print(f"[PASS] Test 6: Gradient flow to span_logits verified (grads: {attn.span_logits.grad.tolist()})")


def test_7_registry_keys():
    """Verify 'adaptive_span' and 'adaptive_span_attention' are present in ATTENTION_REGISTRY."""
    assert "adaptive_span" in ATTENTION_REGISTRY, "'adaptive_span' not found in ATTENTION_REGISTRY"
    assert "adaptive_span_attention" in ATTENTION_REGISTRY, "'adaptive_span_attention' not found in ATTENTION_REGISTRY"
    reg_cls = ATTENTION_REGISTRY.get("adaptive_span")
    assert reg_cls is AdaptiveSpanAttention, f"Expected AdaptiveSpanAttention, got {reg_cls}"
    print("[PASS] Test 7: ATTENTION_REGISTRY registrations verified")


def test_8_attention_flags():
    """Verify supports_causal, supports_bidirectional, supports_gqa, and is_custom_attention flags."""
    assert AdaptiveSpanAttention.supports_causal is True
    assert AdaptiveSpanAttention.supports_bidirectional is True
    assert AdaptiveSpanAttention.supports_gqa is True
    assert AdaptiveSpanAttention.is_custom_attention is True
    assert AdaptiveSpanAttention.supports_rope is True
    print("[PASS] Test 8: Attention flags (causal, bidirectional, gqa, is_custom_attention, rope) verified")


if __name__ == "__main__":
    print("=" * 60)
    print("Running AdaptiveSpanAttention Comprehensive Unit Tests")
    print("=" * 60)
    test_1_instantiation_and_span_init()
    test_2_span_regularization_loss_positive()
    test_3_multi_module_span_loss_aggregation()
    test_4_forward_pass_output_shape()
    test_5_softmax_nan_safety_all_masked()
    test_6_gradient_flow_to_span_logits()
    test_7_registry_keys()
    test_8_attention_flags()
    print("=" * 60)
    print("ALL 8 UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
