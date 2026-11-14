#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for CUDAGraphDecodeRunner static cache validation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.models.cache import UnifiedHybridCache
from neuralops.models.cuda_graph import CUDAGraphDecodeRunner


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, input_ids: torch.Tensor, past_key_values: object = None, use_cache: bool = True) -> torch.Tensor:
        return self.fc(torch.zeros(input_ids.shape[0], 1, 16))


class DummyStaticCache:
    """Mock static cache representation."""
    is_static = True


def test_cuda_graph_runner_raises_type_error_for_non_static_cache() -> None:
    model = DummyModel()
    runner = CUDAGraphDecodeRunner(model=model, batch_size=1)

    # Passing UnifiedHybridCache should fail
    hybrid_cache = UnifiedHybridCache(num_layers=2)
    with pytest.raises(TypeError, match="CUDAGraphDecodeRunner requires a static-size KV cache"):
        runner.capture(sample_cache=hybrid_cache)

    # Passing None should fail
    with pytest.raises(TypeError, match="CUDAGraphDecodeRunner requires a static-size KV cache"):
        runner.capture(sample_cache=None)

    # Passing dynamic list or tuple should fail
    with pytest.raises(TypeError, match="CUDAGraphDecodeRunner requires a static-size KV cache"):
        runner.capture(sample_cache=[])


def test_cuda_graph_runner_accepts_static_cache() -> None:
    model = DummyModel()
    runner = CUDAGraphDecodeRunner(model=model, batch_size=1)
    static_cache = DummyStaticCache()

    # On non-CUDA / CPU machine, capture with static cache will pass the static cache check
    # and exit gracefully due to device not being CUDA
    runner.capture(sample_cache=static_cache)
    assert not runner.is_captured
