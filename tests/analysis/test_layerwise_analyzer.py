"""Tests for neuralops.analysis.layerwise_analyzer module."""

import pytest
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


class TestLayerwiseAnalyzer:
    """Test LayerwiseAnalyzer hook registration and model analysis."""

    def test_creation(self):
        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
        tokenizer = MagicMock()
        analyzer = LayerwiseAnalyzer(model=model, tokenizer=tokenizer, device="cpu")
        assert analyzer is not None
        assert analyzer.device == "cpu"

    def test_estimate_intrinsic_dimension(self):
        # 100 samples in 20 dimensions with effective rank ~5
        rng = np_rng = torch.randn(100, 5)
        proj = torch.randn(5, 20)
        X = (rng @ proj).numpy()
        id_est = estimate_intrinsic_dimension(X)
        assert isinstance(id_est, (int, float))
        assert id_est >= 1.0
