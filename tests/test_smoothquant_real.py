#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for production SmoothQuant."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.compression_engine.quantization.smoothquant import (
    SmoothQuantLinear,
    smoothquant_layer,
    smoothquant_model,
)


def test_smoothquant_layer_real():
    torch.manual_seed(42)
    layer = nn.Linear(32, 32, bias=False)
    x = torch.randn(16, 32)
    x[:, 5] *= 10.0

    orig_out = layer(x)
    mod, s = smoothquant_layer(layer, alpha=0.5, calib_data=x, quantize=True)

    assert hasattr(mod, "weight_int8")
    assert mod.weight_int8.dtype == torch.int8
    assert hasattr(mod, "weight_scale")
    assert hasattr(mod, "s_inv")

    new_out = mod(x)
    scaled_x = x / s.unsqueeze(0)
    manual_out = F.linear(scaled_x, mod.weight)
    assert torch.allclose(new_out, manual_out, atol=1e-4)

    rel_error = torch.norm(orig_out - new_out) / torch.norm(orig_out)
    assert rel_error < 0.05


def test_smoothquant_smoothing_exact():
    torch.manual_seed(42)
    layer = nn.Linear(32, 32, bias=False)
    x = torch.randn(16, 32)
    x[:, 5] *= 10.0

    orig_out = layer(x)
    mod, s = smoothquant_layer(layer, alpha=0.5, calib_data=x, quantize=False)
    new_out = mod(x)
    assert torch.allclose(orig_out, new_out, atol=1e-4)


def test_smoothquant_linear_module():
    torch.manual_seed(42)
    s = torch.ones(32) * 1.5
    sq_lin = SmoothQuantLinear(32, 16, s=s)
    x = torch.randn(8, 32)
    out = sq_lin(x)
    assert out.shape == (8, 16)
    assert hasattr(sq_lin, "weight_int8")
    assert sq_lin.weight_int8.dtype == torch.int8


def test_smoothquant_model_real():
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
    calib = [torch.randn(4, 16) for _ in range(2)]
    scales = smoothquant_model(model, calib, alpha=0.5)
    assert len(scales) == 2
