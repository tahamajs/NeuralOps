#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Recursive Sequential Traversal in Layer Replacement (P053).

Verifies that attention layers nested inside nn.Sequential, nn.ModuleList,
or custom container blocks are correctly discovered and surgically replaced.
"""

import pytest
import torch
import torch.nn as nn

from neuralops.surgery.layer_replacement import (
    find_nested_attention_modules,
    replace_attention_recursive,
)


class DummyAttention(nn.Module):
    """Mock standard attention block."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ReplacementLinearAttention(nn.Module):
    """Mock replacement efficient attention block."""

    def __init__(self, dim: int = 32, old_module: nn.Module | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_map = nn.Linear(dim, dim, bias=False)
        self.is_linear_replacement = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel_map(x)


class DeepNestedModel(nn.Module):
    """
    Model with attention nested inside an nn.Sequential inside a container block,
    as well as inside an nn.ModuleList.
    """

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sequential(
                nn.LayerNorm(dim),
                DummyAttention(dim),
            ),
        )
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    DummyAttention(dim),
                    nn.Linear(dim, dim),
                )
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.stages[0](x)
        return x


def test_sequential_nested_attention_replacement():
    """P053: Verify attention inside nn.Sequential within container is replaced."""
    model = DeepNestedModel(dim=32)

    # Initial check: inner module is DummyAttention
    assert isinstance(model.encoder[1][1], DummyAttention)
    assert isinstance(model.stages[0][0], DummyAttention)

    # Execute recursive replacement
    replaced_model = replace_attention_recursive(
        model,
        new_attention_cls=lambda old_module=None, **kwargs: ReplacementLinearAttention(
            dim=32, old_module=old_module
        ),
    )

    # Verify both nested targets were replaced
    assert isinstance(replaced_model.encoder[1][1], ReplacementLinearAttention)
    assert getattr(replaced_model.encoder[1][1], "is_linear_replacement", False) is True

    assert isinstance(replaced_model.stages[0][0], ReplacementLinearAttention)
    assert getattr(replaced_model.stages[0][0], "is_linear_replacement", False) is True

    # Verify forward pass executes cleanly through the modified graph
    x = torch.randn(2, 4, 32)
    out = replaced_model(x)
    assert out.shape == (2, 4, 32)


def test_find_nested_attention_modules():
    """Verify scanner finds targets at all nesting levels."""
    model = DeepNestedModel(dim=32)
    targets = find_nested_attention_modules(model)

    target_paths = [path for path, _ in targets]
    assert "encoder.1.1" in target_paths
    assert "stages.0.0" in target_paths
    assert len(targets) == 2


def test_layer_index_filtering_nested():
    """Verify layer index filtering functions when digits exist in path."""
    container = nn.ModuleDict(
        {
            "layer_0": nn.Sequential(DummyAttention(16)),
            "layer_1": nn.Sequential(DummyAttention(16)),
        }
    )
    targets = find_nested_attention_modules(container, layer_indices=[1])
    assert len(targets) == 1
    assert "layer_1.0" in targets[0][0]
