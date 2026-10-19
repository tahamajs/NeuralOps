# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for production SmoothQuant."""
import torch
import torch.nn as nn
from neuralops.compression_engine.quantization.smoothquant import smoothquant_layer, smoothquant_model


def test_smoothquant_layer_real():
    torch.manual_seed(42)
    layer = nn.Linear(32, 32, bias=False)
    x = torch.randn(16, 32)
    x[:, 5] *= 10.0

    orig_out = layer(x)
    mod, s = smoothquant_layer(layer, alpha=0.5, calib_data=x)

    new_out = mod(x)
    assert torch.allclose(orig_out, new_out, atol=1e-4)

    import torch.nn.functional as F
    scaled_x = x / s.unsqueeze(0)
    manual_out = F.linear(scaled_x, mod.weight)
    assert torch.allclose(orig_out, manual_out, atol=1e-4)


def test_smoothquant_model_real():
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
    calib = [torch.randn(4, 16) for _ in range(2)]
    scales = smoothquant_model(model, calib, alpha=0.5)
    assert len(scales) == 2
