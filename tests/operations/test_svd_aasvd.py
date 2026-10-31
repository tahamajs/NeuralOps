#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Activation-Aware SVD (AASVD) implementation.
"""

import pytest
import torch
import torch.nn as nn

from neuralops.core.low_rank.svd_linear import SVDLinear
from neuralops.operations.registry import get_operation
from neuralops.operations.weights.svd.aasvd import (
    aasvd_decompose,
    allocate_rank_auto,
    compute_activation_covariance,
)
from neuralops.operations.weights.svd_aasvd import WeightsCompressAASVDOperation


class ToyTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(hidden_size, hidden_size * 2)
        self.output = nn.Module()
        self.output.dense = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.intermediate.dense(x))
        return self.output.dense(h)


class ToyModel(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.config = nn.Module()
        self.config.hidden_size = hidden_size
        self.transformer = nn.Module()
        self.transformer.layer = nn.ModuleList([ToyTransformerBlock(hidden_size) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.transformer.layer:
            h = layer(h)
        return h


def test_aasvd_decompose_mathematics():
    torch.manual_seed(42)
    W = torch.randn(64, 128)
    U_r, S_r, V_r, energy = aasvd_decompose(W, rank=32, whitening=False)
    assert U_r.shape == (64, 32)
    assert S_r.shape == (32,)
    assert V_r.shape == (128, 32)
    assert 0.0 < energy <= 1.0

    # Reconstruction test
    W_recon = (U_r * S_r.unsqueeze(0)) @ V_r.t()
    assert W_recon.shape == (64, 128)


def test_aasvd_whitening_covariance():
    torch.manual_seed(42)
    model = ToyModel(64)
    targets = [("transformer.layer.0.intermediate.dense", model.transformer.layer[0].intermediate.dense)]
    covs = compute_activation_covariance(model, targets, calibration_tokens=128)
    assert "transformer.layer.0.intermediate.dense" in covs
    cov = covs["transformer.layer.0.intermediate.dense"]
    assert cov.shape == (64, 64)
    assert torch.all(torch.diag(cov) >= 0.0)


def test_weights_compress_aasvd_end_to_end():
    torch.manual_seed(42)
    model = ToyModel(64)
    x = torch.randn(2, 16, 64)
    out_before = model(x)

    base_params = sum(p.numel() for p in model.parameters())

    op = get_operation("weights.compress", "1.0.0")
    compressed_model, evidence = op.apply(
        model,
        {
            "method": "aasvd",
            "target": "*.intermediate.dense",
            "target_ratio": 0.5,
            "rank": 16,
            "whitening": True,
            "calibration_tokens": 128,
        },
    )

    out_after = compressed_model(x)
    assert out_after.shape == out_before.shape

    final_params = sum(p.numel() for p in compressed_model.parameters())
    assert final_params < base_params
    assert evidence["compression_ratio"] < 1.0
    assert evidence["modules_compressed"] == 2
    assert isinstance(compressed_model.transformer.layer[0].intermediate.dense, SVDLinear)


def test_weights_compress_determinism():
    x = torch.randn(2, 8, 64)

    torch.manual_seed(1234)
    m1 = ToyModel(64)
    op1 = WeightsCompressAASVDOperation()
    cm1, _ = op1.apply(m1, {"method": "aasvd", "target": "*dense*", "target_ratio": 0.5, "rank": 16})
    out1 = cm1(x)

    torch.manual_seed(1234)
    m2 = ToyModel(64)
    op2 = WeightsCompressAASVDOperation()
    cm2, _ = op2.apply(m2, {"method": "aasvd", "target": "*dense*", "target_ratio": 0.5, "rank": 16})
    out2 = cm2(x)

    assert torch.allclose(out1, out2, atol=1e-4)
