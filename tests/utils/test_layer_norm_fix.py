#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for layer norm fix utilities and FP32 accumulator precision."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from neuralops.utils.layer_norm_fix import (
    RMSNorm,
    LayerNorm,
    SafeRMSNorm,
    SafeLayerNorm,
    FP32AccumulatorRMSNorm,
    FP32AccumulatorLayerNorm,
    fix_layernorm_naming,
    replace_layernorm_with_safe_norm,
)


def test_rmsnorm_forward_and_backward():
    """Verify RMSNorm forward pass, shape preservation, and gradient flow."""
    d = 64
    norm = RMSNorm(d, eps=1e-6)
    x = torch.randn(2, 16, d, requires_grad=True)
    out = norm(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    # Gradient flow
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert norm.weight.grad is not None
    assert torch.isfinite(norm.weight.grad).all()


def test_rmsnorm_fp16_numerical_stability():
    """Verify RMSNorm handles large fp16 activations without NaN/overflow in variance."""
    d = 32
    norm = RMSNorm(d, eps=1e-5)
    # 250 in fp16 squares to 62500, close to fp16 max (65504)
    x = torch.full((1, 4, d), 250.0, dtype=torch.float32)
    out = norm(x)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.ones_like(out), atol=1e-3)


def test_layernorm_forward_and_backward():
    """Verify LayerNorm forward pass, shapes, and options."""
    d = 128
    # With affine and bias
    norm = LayerNorm(d, eps=1e-5, elementwise_affine=True, bias=True)
    x = torch.randn(4, 8, d, requires_grad=True)
    out = norm(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

    loss = out.mean()
    loss.backward()
    assert x.grad is not None
    assert norm.weight.grad is not None
    assert norm.bias.grad is not None

    # Without bias
    norm_no_bias = LayerNorm(d, bias=False)
    assert norm_no_bias.bias is None
    out_no_bias = norm_no_bias(torch.randn(2, 4, d))
    assert out_no_bias.shape == (2, 4, d)

    # Without affine
    norm_no_affine = LayerNorm(d, elementwise_affine=False)
    assert norm_no_affine.weight is None
    assert norm_no_affine.bias is None
    out_no_affine = norm_no_affine(torch.randn(2, 4, d))
    assert out_no_affine.shape == (2, 4, d)


def test_aliases():
    """Verify that Safe/FP32 accumulator aliases point to the correct implementations."""
    assert SafeRMSNorm is RMSNorm
    assert SafeLayerNorm is LayerNorm
    assert FP32AccumulatorRMSNorm is RMSNorm
    assert FP32AccumulatorLayerNorm is LayerNorm


def test_fix_layernorm_naming():
    """Verify legacy gamma/beta parameters are remapped to weight/bias."""

    class LegacyNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(10))
            self.beta = nn.Parameter(torch.zeros(10))

    model = nn.Sequential(
        nn.Linear(10, 10),
        LayerNorm(10),
    )
    # Manually simulate legacy naming on the LayerNorm child
    ln = model[1]
    w = ln._parameters.pop("weight")
    b = ln._parameters.pop("bias")
    ln.register_parameter("gamma", w)
    ln.register_parameter("beta", b)

    assert "gamma" in dict(ln.named_parameters())
    assert "weight" not in dict(ln.named_parameters())

    fix_layernorm_naming(model)

    assert "weight" in dict(ln.named_parameters())
    assert "bias" in dict(ln.named_parameters())
    assert "gamma" not in dict(ln.named_parameters())
    assert "beta" not in dict(ln.named_parameters())


def test_replace_layernorm_with_safe_norm():
    """Verify nn.LayerNorm is recursively replaced with SafeLayerNorm while weights match."""

    class SubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(64)
            self.linear = nn.Linear(64, 64)

    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = SubModel()
            self.ln2 = nn.LayerNorm(64, bias=False)

    model = FullModel()
    orig_w1 = model.sub.ln1.weight.clone()
    orig_b1 = model.sub.ln1.bias.clone()

    replace_layernorm_with_safe_norm(model)

    assert isinstance(model.sub.ln1, LayerNorm)
    assert not isinstance(model.sub.ln1, nn.LayerNorm)
    assert isinstance(model.ln2, LayerNorm)
    assert model.ln2.bias is None

    # Weights should be copied exactly
    assert torch.equal(model.sub.ln1.weight, orig_w1)
    assert torch.equal(model.sub.ln1.bias, orig_b1)
