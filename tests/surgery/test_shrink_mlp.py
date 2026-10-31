#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for shrink_mlp surgery operator.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.surgery.mlp.shrink import shrink_mlp
from neuralops.surgery.registry import get_surgery


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden=32, intermediate=128):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MockSwiGLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = SwiGLUMLP()

    def forward(self, x):
        return self.mlp(x)


class GeluMLP(nn.Module):
    def __init__(self, hidden=32, intermediate=128):
        super().__init__()
        self.c_fc = nn.Linear(hidden, intermediate)
        self.c_proj = nn.Linear(intermediate, hidden)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class MockGeluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = GeluMLP()

    def forward(self, x):
        return self.mlp(x)


def test_shrink_mlp_metadata():
    meta = get_surgery("shrink_mlp")
    assert meta.name == "shrink_mlp"
    assert meta.category == "mlp"
    assert meta._experimental is True


def test_shrink_swiglu():
    model = MockSwiGLUModel()
    x = torch.randn(2, 4, 32)
    assert model.mlp.gate_proj.out_features == 128

    with pytest.warns(UserWarning, match="experimental"):
        shrunk = shrink_mlp(model, ratio=0.5)

    assert shrunk.mlp.gate_proj.out_features == 64
    assert shrunk.mlp.up_proj.out_features == 64
    assert shrunk.mlp.down_proj.in_features == 64

    out = shrunk(x)
    assert out.shape == (2, 4, 32)


def test_shrink_gelu():
    model = MockGeluModel()
    x = torch.randn(2, 4, 32)
    assert model.mlp.c_fc.out_features == 128

    with pytest.warns(UserWarning, match="experimental"):
        shrunk = shrink_mlp(model, ratio=0.75)

    assert shrunk.mlp.c_fc.out_features == 96
    assert shrunk.mlp.c_proj.in_features == 96

    out = shrunk(x)
    assert out.shape == (2, 4, 32)
