#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for DataParallelBenchmark executor.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.benchmarks.parallel_benchmark import BenchmarkResult, DataParallelBenchmark
from neuralops.distributed.context import DistributedContext


class ToyClassifier(nn.Module):
    """Deterministic linear classification module."""

    def __init__(self, in_features: int = 16, num_classes: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _accuracy_metric(model: nn.Module, dataset: dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        x = dataset["x"]
        y = dataset["y"]
        pred = model(x).argmax(dim=-1)
        return float((pred == y).float().mean().item())


def test_single_gpu_fallback():
    """Verify single-device fallback has is_distributed=False and matches non-parallel exactly."""
    torch.manual_seed(42)
    model = ToyClassifier(in_features=8, num_classes=2)
    dataset = {
        "x": torch.randn(64, 8),
        "y": torch.randint(0, 2, (64,)),
    }

    ctx = DistributedContext(init_if_available=False)
    # Force single-device execution
    bench = DataParallelBenchmark(ctx)
    bench.world_size = 1

    expected_acc = _accuracy_metric(model, dataset)
    result = bench.run(model, dataset, _accuracy_metric, device="cpu")

    assert isinstance(result, BenchmarkResult)
    assert result.is_distributed is False
    assert abs(result.metric_value - expected_acc) < 1e-6
    assert result.num_samples == 64
    assert len(result.per_gpu_metrics) == 1
    assert abs(result.per_gpu_metrics[0] - expected_acc) < 1e-6


def test_dataset_partitioning_disjoint():
    """Verify dataset shards are mutually disjoint across ranks and cover the full dataset."""
    N = 120
    dataset = {
        "x": torch.arange(N),
        "y": torch.arange(N),
    }

    for world_size in [2, 3, 4]:
        all_indices = []
        shards = []
        for r in range(world_size):
            shard, length = DataParallelBenchmark.partition_dataset(dataset, r, world_size)
            assert length == len(shard["x"])
            shard_idx = shard["x"].tolist()
            shards.append(set(shard_idx))
            all_indices.extend(shard_idx)

        # Disjointness check
        for i in range(world_size):
            for j in range(i + 1, world_size):
                assert shards[i].isdisjoint(shards[j]), f"Ranks {i} and {j} overlap!"

        # Completeness check
        assert sorted(all_indices) == list(range(N))


def test_per_gpu_metrics_has_world_size_entries():
    """Verify per_gpu_metrics list has exactly world_size entries."""
    torch.manual_seed(42)
    model = ToyClassifier(in_features=4, num_classes=2)
    dataset = {
        "x": torch.randn(32, 4),
        "y": torch.randint(0, 2, (32,)),
    }

    ctx = DistributedContext(init_if_available=False)
    bench = DataParallelBenchmark(ctx)
    result = bench.run(model, dataset, _accuracy_metric, device="cpu")

    assert len(result.per_gpu_metrics) == bench.world_size
    assert isinstance(result.per_gpu_metrics[0], float)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_two_gpu_accuracy_matches_single():
    """Verify aggregated accuracy on 2 GPUs is within 1e-6 of single-GPU accuracy."""
    torch.manual_seed(42)
    model = ToyClassifier(in_features=16, num_classes=4)
    N = 500
    dataset = {
        "x": torch.randn(N, 16),
        "y": torch.randint(0, 4, (N,)),
    }

    ctx = DistributedContext(init_if_available=False)
    bench = DataParallelBenchmark(ctx)

    # 1. Single-device run
    single_res = bench.run(model, dataset, _accuracy_metric, device="cuda:0")

    # 2. Parallel run across 2 GPUs
    bench.world_size = 2
    parallel_res = bench.run(model, dataset, _accuracy_metric)

    assert parallel_res.is_distributed is True
    assert abs(parallel_res.metric_value - single_res.metric_value) < 1e-6


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 CUDA devices")
def test_two_gpu_throughput_2x():
    """Verify 2-GPU throughput is >= 1.7x single GPU throughput allowing 15% overhead."""
    class BenchmarkMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    torch.manual_seed(42)
    model = BenchmarkMLP()
    N = 20000
    dataset = {
        "x": torch.randn(N, 256),
        "y": torch.randint(0, 10, (N,)),
    }

    ctx = DistributedContext(init_if_available=False)
    bench = DataParallelBenchmark(ctx)

    # 1. Warmup
    bench.world_size = 1
    _ = bench.run(model, dataset, _accuracy_metric, device="cuda:0")

    # 2. Single-device run
    single_res = bench.run(model, dataset, _accuracy_metric, device="cuda:0")

    # 3. Parallel run across 2 GPUs
    bench.world_size = 2
    parallel_res = bench.run(model, dataset, _accuracy_metric)

    assert parallel_res.is_distributed is True
    speedup = parallel_res.throughput_samples_per_sec / max(single_res.throughput_samples_per_sec, 1e-9)
    assert speedup >= 1.7, f"Speedup {speedup:.2f}x was below expected 1.7x gate"
