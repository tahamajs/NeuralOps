#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for SwiftSVDActivation (Group 5, Key 77).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.svd.swift_svd_act import (
    SwiftSVDActivationCompressor,
    swift_svd_act_factory,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 256, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_registry():
    assert "swift_svd_act" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("swift_svd_act") is not None


def test_swift_svd_act_reduction():
    model = DummyMLP()
    orig_params = sum(p.numel() for p in model.parameters())
    calib = [torch.randn(4, 128) for _ in range(2)]
    compressor = SwiftSVDActivationCompressor(model, rank=16, calibration_loader=calib, max_error=1.0)
    compressed, stats = compressor.compress()

    comp_params = sum(p.numel() for p in compressed.parameters())
    assert orig_params / max(comp_params, 1) > 1.05

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_swift_svd_act_factory():
    model = DummyMLP()
    comp_model, stats = swift_svd_act_factory(model, rank=16, return_stats=True)
    assert comp_model is not None
    assert isinstance(stats, dict)
