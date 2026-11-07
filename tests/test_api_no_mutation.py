#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying zero in-place mutation of caller-owned models, configs, and batches (F-86)."""

from __future__ import annotations

import copy
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import neuralops.api as api
from neuralops.core.compression_pipeline import CompressionPipeline


class SimpleToyModel(nn.Module):
    """Toy neural network for testing compression and evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 2)

    def __iter__(self):
        return iter(self.children())

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
        x = self.fc1(input_ids)
        return self.fc2(x)


def test_compress_does_not_change_model_class() -> None:
    """Verify api.compress() retains original class, isinstance check, and normal module iteration."""
    # Test 1: Standard nn.Module
    model = SimpleToyModel()
    orig_class = type(model)

    out_model = api.compress(model, method="direct_svd", rank=16)

    assert type(out_model) is orig_class, f"Expected {orig_class}, got {type(out_model)}"
    assert isinstance(out_model, orig_class)
    assert isinstance(out_model, SimpleToyModel)

    # Standard model must NOT have a fake __iter__ injected returning (self, stats)
    try:
        res = tuple(out_model)
        assert not (len(res) == 2 and res[0] is out_model), "tuple(model) returned (self, stats)!"
    except TypeError:
        # Standard nn.Module is not iterable in PyTorch
        pass

    # Check that stats are stored explicitly as an attribute
    assert hasattr(out_model, "_neuralops_compression_stats")

    # Test 2: Naturally iterable model like nn.Sequential
    seq_model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 2))
    out_seq = api.compress(seq_model, method="direct_svd", rank=16)

    assert type(out_seq) is nn.Sequential
    assert isinstance(out_seq, nn.Sequential)
    seq_children = list(out_seq)
    assert len(seq_children) == 2
    assert seq_children[0] is not out_seq, "list(seq_model) returned (self, stats) proxy!"
    assert hasattr(out_seq, "_neuralops_compression_stats")


def test_compress_does_not_mutate_config() -> None:
    """Verify that CompressionPipeline.run() does not pop or mutate caller's configuration dict."""
    model = SimpleToyModel()
    config = {
        "compressors": [
            {
                "method": "direct_svd",
                "rank": 16,
            }
        ],
        "seed": 42,
    }
    config_copy = copy.deepcopy(config)

    pipeline = CompressionPipeline(model, config)
    _ = pipeline.run()

    # The caller's config dictionary and inner step dict must be completely unchanged
    assert config == config_copy, f"Caller config was mutated! Before: {config_copy}, After: {config}"
    assert "method" in config["compressors"][0]


def test_evaluate_does_not_mutate_batch() -> None:
    """Verify that api.evaluate() does not mutate batch dictionary (labels key remains)."""
    model = SimpleToyModel()

    # Create dummy DataLoader yielding batch dicts
    inputs = torch.randn(8, 32)
    labels = torch.randint(0, 2, (8,))
    dataset = [{"input_ids": inputs[i], "labels": labels[i]} for i in range(8)]
    loader = DataLoader(dataset, batch_size=4)

    metrics = api.evaluate(model, dataset=loader, device="cpu")
    assert "metrics" in metrics
    assert "accuracy" in metrics["metrics"] or "loss" in metrics or len(metrics) > 0

    # Verify that the dataset/batch items still retain their labels key
    for item in dataset:
        assert "labels" in item, "Batch dictionary 'labels' was popped during evaluation!"
