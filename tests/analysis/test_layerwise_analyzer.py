#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for neuralops.analysis.layerwise_analyzer module."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from neuralops.analysis.layerwise_analyzer import (
    LayerwiseAnalyzer,
    StreamingLayerAccumulator,
    estimate_intrinsic_dimension,
)


class TestStreamingLayerAccumulator:
    """Test online moment accumulator and rank estimation."""

    def test_accumulator_update_moments(self):
        acc = StreamingLayerAccumulator()
        t = torch.randn(4, 16, 32)
        acc.update(t)

        assert acc.total_batches == 1
        assert acc.total_tokens == 4 * 16
        assert acc.total_elements == 4 * 16 * 32
        assert acc.running_mean_vector is not None
        assert acc.running_mean_vector.shape == (32,)

    def test_accumulator_multiple_batches(self):
        acc = StreamingLayerAccumulator()
        for _ in range(3):
            acc.update(torch.randn(2, 8, 16))
        assert acc.total_batches == 3
        assert acc.total_tokens == 48

    def test_covariance_merge_includes_between_batch_mean_shift(self):
        acc = StreamingLayerAccumulator()
        acc.update(torch.tensor([[0.0, 0.0], [2.0, 0.0]]))
        acc.update(torch.tensor([[10.0, 0.0], [12.0, 0.0]]))

        expected = np.cov(np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]]).T, bias=True)
        assert np.allclose(acc.cov_matrix, expected)


class TestLayerwiseAnalyzer:
    """Test LayerwiseAnalyzer hook registration and model analysis."""

    def test_creation(self):
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
        tokenizer = MagicMock()
        analyzer = LayerwiseAnalyzer(model=model, tokenizer=tokenizer, device="cpu")
        assert analyzer is not None
        assert analyzer.device == "cpu"

    def test_estimate_intrinsic_dimension(self):
        rng = np_rng = torch.randn(100, 5)
        proj = torch.randn(5, 20)
        X = (rng @ proj).numpy()
        id_est = estimate_intrinsic_dimension(X)
        assert isinstance(id_est, (int, float))
        assert id_est >= 1.0

    def test_intrinsic_dimension_recovers_low_dimensional_line(self):
        samples = torch.arange(64, dtype=torch.float32).reshape(-1, 1).numpy()
        assert estimate_intrinsic_dimension(samples, k=5) < 2.0

    def test_intrinsic_dimension_rejects_invalid_neighbor_count(self):
        with pytest.raises(ValueError, match="at least 2"):
            estimate_intrinsic_dimension(np.ones((8, 2)), k=1)
