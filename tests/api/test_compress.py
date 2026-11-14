#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for public API compress."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.api import compress, list_compressors


class DummySequentialModel(nn.Module):
    def __init__(self, in_features: int = 64, hidden_features: int = 64, out_features: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class TestCompressAPI:
    """Verifies compress API surface, 3 compressor engines, and metadata attachment."""

    @pytest.mark.parametrize("method", ["direct_svd", "asvd", "auto_svd"])
    def test_compress_three_compressor_types(self, method: str):
        torch.manual_seed(42)
        model = DummySequentialModel(in_features=64, hidden_features=64, out_features=32)
        compressed = compress(model, method=method, rank=16)

        # Check returned model is nn.Module
        assert isinstance(compressed, nn.Module)

        # Verify execution and output shape
        x = torch.randn(2, 64)
        out = compressed(x)
        assert out.shape == (2, 32)
        assert torch.isfinite(out).all()

        # Verify operation recorded in metadata
        assert hasattr(compressed, "_neuralops_metadata")
        assert len(compressed._neuralops_metadata) > 0
        last_meta = compressed._neuralops_metadata[-1]
        assert last_meta["op"] == "compress"
        assert last_meta["type"] == method
        assert last_meta["kwargs"]["rank"] == 16

        # Verify stats attribute attached
        assert hasattr(compressed, "_neuralops_compression_stats")

    def test_compress_svd_alias(self):
        torch.manual_seed(42)
        model = DummySequentialModel(in_features=64, hidden_features=64, out_features=32)
        compressed = compress(model, method="svd", rank=16)
        assert hasattr(compressed, "_neuralops_metadata")
        assert compressed._neuralops_metadata[-1]["type"] == "svd"

        x = torch.randn(2, 64)
        out = compressed(x)
        assert out.shape == (2, 32)

    def test_compress_unknown_method_raises(self):
        model = DummySequentialModel()
        with pytest.raises(Exception):
            compress(model, method="non_existent_compressor_xyz", rank=16)

    def test_compress_layer_filtering(self):
        torch.manual_seed(42)
        model = DummySequentialModel(in_features=64, hidden_features=64, out_features=32)
        compressed = compress(model, method="direct_svd", rank=16, layers=[0])
        assert hasattr(compressed, "_neuralops_metadata")
        assert compressed._neuralops_metadata[-1]["kwargs"]["layers"] == [0]

        x = torch.randn(2, 64)
        out = compressed(x)
        assert out.shape == (2, 32)

    def test_list_compressors_is_non_empty(self):
        compressors = list_compressors()
        assert len(compressors) > 0
        assert "direct_svd" in compressors
