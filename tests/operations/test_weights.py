#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for Weights Operations (decompose, compress, prune, quantize)."""

import pytest
import torch
import torch.nn as nn

from neuralops.core.low_rank.svd_linear import SVDLinear
from neuralops.operations.registry import get_operation
from neuralops.operations.weights.compress import WeightsCompressOperation
from neuralops.operations.weights.decompose import WeightsDecomposeOperation
from neuralops.operations.weights.prune import (
    WeightsPruneOperation,
    prune_2_to_4,
    prune_channel,
    prune_unstructured,
)
from neuralops.operations.weights.quantize import WeightsQuantizeOperation


class SimpleMLP(nn.Module):
    def __init__(self, in_features: int = 32, hidden_features: int = 64, out_features: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class NestedModel(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.mlp = SimpleMLP(dim, dim * 2, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.mlp(x))


# ---------------------------------------------------------------------------
# 1. Tests for weights.decompose
# ---------------------------------------------------------------------------


def test_weights_decompose_basic():
    torch.manual_seed(42)
    model = SimpleMLP(32, 64, 32)
    x = torch.randn(2, 32)
    y_before = model(x)

    op = WeightsDecomposeOperation()
    compressed_model, evidence = op.apply(model, {"rank": 8})

    assert evidence["operation"] == "weights.decompose"
    assert evidence["modules_decomposed"] == 2
    assert isinstance(compressed_model.fc1, SVDLinear)
    assert isinstance(compressed_model.fc2, SVDLinear)
    assert compressed_model.fc1.rank == 8
    assert compressed_model.fc2.rank == 8

    y_after = compressed_model(x)
    assert y_after.shape == y_before.shape
    # Check low-rank approximation error is reasonable
    assert torch.allclose(y_before, y_after, atol=1.5)


def test_weights_decompose_with_target():
    torch.manual_seed(42)
    model = NestedModel(32)
    op = WeightsDecomposeOperation()
    # Decompose only the outer projection
    compressed_model, evidence = op.apply(model, {"rank": 8, "target": "proj"})

    assert evidence["modules_decomposed"] == 1
    assert isinstance(compressed_model.proj, SVDLinear)
    assert not isinstance(compressed_model.mlp.fc1, SVDLinear)


def test_weights_decompose_via_registry():
    op = get_operation("weights.decompose", "1.0.0")
    assert isinstance(op, WeightsDecomposeOperation)


# ---------------------------------------------------------------------------
# 2. Tests for weights.compress
# ---------------------------------------------------------------------------


def test_weights_compress_aasvd_and_svd():
    torch.manual_seed(42)
    model = NestedModel(32)
    op = WeightsCompressOperation()

    compressed_model, evidence = op.apply(
        model,
        {
            "method": "aasvd",
            "target": "*.fc1",
            "target_ratio": 0.5,
            "rank": 8,
            "whitening": False,
        },
    )

    assert evidence["operation"] == "weights.compress"
    assert evidence["method"] == "aasvd"
    assert evidence["modules_compressed"] >= 1
    assert isinstance(compressed_model.mlp.fc1, SVDLinear)


def test_weights_compress_asvd():
    torch.manual_seed(42)
    model = NestedModel(32)
    op = WeightsCompressOperation()

    compressed_model, evidence = op.apply(
        model,
        {
            "method": "asvd",
            "target_ratio": 0.5,
            "rank": 8,
        },
    )

    assert evidence["operation"] == "weights.compress"
    assert evidence["method"] == "asvd"
    assert evidence["modules_compressed"] >= 1


def test_weights_compress_fwsvd():
    torch.manual_seed(42)
    model = NestedModel(32)
    op = WeightsCompressOperation()

    compressed_model, evidence = op.apply(
        model,
        {
            "method": "fwsvd",
            "target_ratio": 0.5,
            "rank": 8,
        },
    )

    assert evidence["operation"] == "weights.compress"
    assert evidence["method"] == "fwsvd"
    assert evidence["modules_compressed"] >= 1


def test_weights_compress_via_registry():
    op = get_operation("weights.compress", "1.0.0")
    assert isinstance(op, WeightsCompressOperation)


# ---------------------------------------------------------------------------
# 3. Tests for weights.prune
# ---------------------------------------------------------------------------


def test_prune_unstructured_helper():
    torch.manual_seed(42)
    w = torch.randn(32, 32)
    pruned = prune_unstructured(w, sparsity=0.5)
    zeros = (pruned == 0).sum().item()
    assert zeros >= int(0.5 * 32 * 32) - 1


def test_prune_2_to_4_helper():
    torch.manual_seed(42)
    w = torch.randn(16, 16)
    pruned = prune_2_to_4(w)
    # Check that in each block of 4 elements, exactly 2 are zero
    flat = pruned.view(-1, 4)
    zeros_per_block = (flat == 0).sum(dim=1)
    assert torch.all(zeros_per_block == 2)


def test_prune_channel_helper():
    torch.manual_seed(42)
    w = torch.randn(10, 20)
    pruned = prune_channel(w, sparsity=0.4, norm="l2")
    row_zeros = (pruned.abs().sum(dim=1) == 0).sum().item()
    assert row_zeros == 4


def test_weights_prune_2_to_4_operation():
    torch.manual_seed(42)
    model = SimpleMLP(16, 32, 16)
    op = WeightsPruneOperation()

    pruned_model, evidence = op.apply(
        model,
        {
            "pattern": "2:4",
        },
    )

    assert evidence["operation"] == "weights.prune"
    assert evidence["pattern"] == "2:4"
    assert evidence["modules_pruned"] == 2
    # Verify 50% sparsity on linear layers
    assert abs(evidence["measured_sparsity"] - 0.5) < 0.05
    # Verify forward works
    x = torch.randn(2, 16)
    out = pruned_model(x)
    assert out.shape == (2, 16)


def test_weights_prune_channel_operation():
    torch.manual_seed(42)
    model = SimpleMLP(32, 64, 32)
    op = WeightsPruneOperation()

    pruned_model, evidence = op.apply(
        model,
        {
            "pattern": "channel",
            "sparsity": 0.5,
            "norm": "l1",
        },
    )

    assert evidence["operation"] == "weights.prune"
    assert evidence["pattern"] == "channel"
    assert evidence["measured_sparsity"] >= 0.45


def test_weights_prune_unstructured_operation():
    torch.manual_seed(42)
    model = SimpleMLP(32, 64, 32)
    op = WeightsPruneOperation()

    pruned_model, evidence = op.apply(
        model,
        {
            "pattern": "unstructured",
            "sparsity": 0.6,
        },
    )

    assert evidence["operation"] == "weights.prune"
    assert evidence["pattern"] == "unstructured"
    assert evidence["measured_sparsity"] >= 0.55


def test_weights_prune_via_registry():
    op = get_operation("weights.prune", "1.0.0")
    assert isinstance(op, WeightsPruneOperation)


# ---------------------------------------------------------------------------
# 4. Tests for weights.quantize
# ---------------------------------------------------------------------------


def test_weights_quantize_fp8():
    torch.manual_seed(42)
    model = SimpleMLP(32, 64, 32)
    op = WeightsQuantizeOperation()

    quant_model, evidence = op.apply(
        model,
        {
            "method": "fp8",
            "bits": 8,
            "fp8_format": "e4m3",
        },
    )

    assert evidence["operation"] == "weights.quantize"
    assert evidence["method"] == "fp8"
    assert evidence["bits"] == 8
    assert evidence["layers_quantized"] == 2


def test_weights_quantize_awq():
    torch.manual_seed(42)
    model = SimpleMLP(32, 64, 32)
    op = WeightsQuantizeOperation()

    quant_model, evidence = op.apply(
        model,
        {
            "method": "int4_awq",
            "bits": 4,
            "group_size": 32,
        },
    )

    assert evidence["operation"] == "weights.quantize"
    assert evidence["method"] == "int4_awq"
    assert evidence["bits"] == 4
    assert evidence["layers_quantized"] == 2


def test_weights_quantize_via_registry():
    op = get_operation("weights.quantize", "1.0.0")
    assert isinstance(op, WeightsQuantizeOperation)
