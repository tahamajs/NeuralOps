#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for Recipe Benchmark Runner (Sub-Agent C4).
"""

import pytest
import torch
import torch.nn as nn

from neuralops.benchmarks import run_recipe_benchmark
from neuralops.recipe.schema import Recipe


class BenchmarkToyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def test_c4_run_recipe_benchmark_on_cpu():
    """Verify run_recipe_benchmark produces real measured numbers on CPU."""
    toy_model = BenchmarkToyModel()

    recipe_dict = {
        "apiVersion": "neuralops/v1",
        "kind": "Recipe",
        "metadata": {
            "name": "test-toy-compress-benchmark",
            "version": "1.0.0",
            "description": "Benchmark recipe for compression",
        },
        "model": {
            "source": "toy-model",
            "dtype": "float32",
        },
        "operations": [
            {
                "id": "c1",
                "type": "weights.compress",
                "version": "1.0.0",
                "config": {"method": "svd", "target_ratio": 0.5},
            }
        ],
    }
    recipe = Recipe.from_dict(recipe_dict)

    res = run_recipe_benchmark(recipe, device="cpu", repeats=3, model=toy_model)

    assert res["success"] is True
    assert res["recipe_name"] == "test-toy-compress-benchmark"
    assert "cpu" in res["device"]

    # Baseline real measured metrics
    base = res["baseline"]
    assert base["latency_ms"] > 0.0
    assert base["throughput_tok_per_sec"] > 0.0
    assert base["vram"]["skipped"] is True  # zero-fabrication on CPU
    assert base["accuracy"]["skipped"] is True  # no dataset

    # Transformed real measured metrics
    trans = res["transformed"]
    assert trans["latency_ms"] > 0.0
    assert trans["throughput_tok_per_sec"] > 0.0
    assert trans["vram"]["skipped"] is True

    # Comparison metrics
    assert "speedup" in res["comparison"]
    assert len(res["stages"]) == 1


def test_c4_run_recipe_benchmark_with_synthetic_accuracy():
    """Verify accuracy is evaluated if a dataset is provided."""
    toy_model = BenchmarkToyModel()
    recipe_dict = {
        "apiVersion": "neuralops/v1",
        "kind": "Recipe",
        "metadata": {
            "name": "test-accuracy-benchmark",
            "version": "1.0.0",
            "description": "Benchmark recipe with accuracy evaluation",
        },
        "model": {
            "source": "toy-model",
            "dtype": "float32",
        },
        "operations": [
            {
                "id": "c1",
                "type": "weights.compress",
                "version": "1.0.0",
                "config": {"method": "svd", "target_ratio": 0.5},
            }
        ],
    }
    recipe = Recipe.from_dict(recipe_dict)

    x = torch.randn(10, 32)
    y = torch.randint(0, 32, (10,))
    dataset = [(x, y)]

    res = run_recipe_benchmark(recipe, device="cpu", repeats=2, model=toy_model, dataset=dataset)

    assert res["baseline"]["accuracy"]["skipped"] is False
    assert isinstance(res["baseline"]["accuracy"]["accuracy"], float)
    assert res["transformed"]["accuracy"]["skipped"] is False
    assert isinstance(res["transformed"]["accuracy"]["accuracy"], float)
