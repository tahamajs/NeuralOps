"""
Unit and Integration Tests for AutoSVD Automated Sensitivity Profiling and Pareto Search.
"""

import pytest
import torch
import torch.nn as nn

from neuralops.compression_engine.svd.auto_svd import (
    FisherSensitivityProfiler,
    solve_optimal_rank_allocation,
    AutoSVDCompressor,
)


def test_fisher_sensitivity_profiler():
    """Verify Fisher information and Gram matrix computation."""
    torch.manual_seed(42)

    class SmallMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 16)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = SmallMLP()
    profiler = FisherSensitivityProfiler(model, device=torch.device("cpu"))

    batches = [torch.randn(4, 32) for _ in range(3)]
    grams = profiler.compute_layer_gram_matrices(batches)

    assert "fc1" in grams
    assert grams["fc1"].shape == (32, 32)
    assert "fc2" in grams
    assert grams["fc2"].shape == (64, 64)


def test_solve_optimal_rank_allocation():
    """Verify Pareto knapsack rank optimization."""
    class MultiLayerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(64, 64)
            self.layer2 = nn.Linear(64, 64)
            self.layer3 = nn.Linear(64, 64)

        def forward(self, x):
            return self.layer3(self.layer2(self.layer1(x)))

    model = MultiLayerNet()
    allocated_ranks, stats = solve_optimal_rank_allocation(
        model,
        target_param_ratio=0.5,
        min_rank=16,
    )

    assert len(allocated_ranks) == 3
    assert all(r >= 16 for r in allocated_ranks.values())
    assert stats["compression_ratio"] > 1.0
    assert stats["compressed_params"] < stats["original_params"]


def test_auto_svd_compressor_pipeline():
    """Verify end-to-end AutoSVD compression replacement."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense1 = nn.Linear(64, 64)
            self.dense2 = nn.Linear(64, 32)

        def forward(self, x):
            return self.dense2(torch.relu(self.dense1(x)))

    model = SimpleModel()
    compressor = AutoSVDCompressor(model, target_param_ratio=0.5, min_rank=16)
    x = torch.randn(2, 64)
    orig_out = model(x)

    c_model, stats = compressor.compress()
    c_out = c_model(x)

    assert stats["compressed_layers"] == 2
    assert c_out.shape == orig_out.shape
