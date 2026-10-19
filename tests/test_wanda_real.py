# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for production Wanda pruning."""
import torch
import torch.nn as nn
from neuralops.compression_engine.pruning.wanda import wanda_prune_layer, wanda_prune_model


def test_wanda_prune_layer_real():
    torch.manual_seed(42)
    layer = nn.Linear(32, 32, bias=False)
    x = torch.randn(16, 32)
    x[:, 0] *= 50.0

    mod, mask = wanda_prune_layer(layer, sparsity=0.5, calib_data=x)
    assert mask.shape == (32, 32)
    assert mask[:, 0].sum() > 20

    zeros_fraction = float((layer.weight.data == 0).sum()) / layer.weight.data.numel()
    assert 0.45 <= zeros_fraction <= 0.55


def test_wanda_prune_model_real():
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
    calib = [torch.randn(4, 16) for _ in range(2)]
    masks = wanda_prune_model(model, calib, sparsity=0.3)
    assert len(masks) == 2
