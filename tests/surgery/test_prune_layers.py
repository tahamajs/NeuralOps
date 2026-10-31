#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for prune_layers surgery operator.
"""

import pytest
import torch
import torch.nn as nn

from neuralops.surgery.layers.prune import prune_layers
from neuralops.surgery.registry import get_surgery


class SimpleBlock(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.lin(x)


class MockTransformer(nn.Module):
    def __init__(self, num_layers=6, dim=32):
        super().__init__()
        self.layers = nn.ModuleList([SimpleBlock(dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_prune_layers_metadata():
    meta = get_surgery("prune_layers")
    assert meta.name == "prune_layers"
    assert meta.category == "layers"
    assert meta._experimental is True


def test_prune_layers_keep_indices():
    model = MockTransformer(num_layers=6)
    x = torch.randn(2, 4, 32)
    assert len(model.layers) == 6

    with pytest.warns(UserWarning, match="experimental"):
        pruned = prune_layers(model, keep_layers=[0, 1, 4, 5])

    assert len(pruned.layers) == 4
    out = pruned(x)
    assert out.shape == (2, 4, 32)


def test_prune_layers_drop_count():
    model = MockTransformer(num_layers=6)
    with pytest.warns(UserWarning, match="experimental"):
        pruned = prune_layers(model, drop_count=2)
    assert len(pruned.layers) == 4
