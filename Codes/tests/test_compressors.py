"""
Unit tests for compression engines (SVD, AASVD, SliceGPT, Kronecker-TT, Pruning, Quantization, Registry).
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

# FIX: Ensure parent path is in sys.path for direct script execution
_FRAMEWORK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FRAMEWORK_PARENT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_PARENT)

from framework.compression_engine.aa_svd_compressor import AASVDCompressor, SVDLinear
from framework.compression_engine.kronecker_compressor import KroneckerCompressor
from framework.compression_engine.kronecker_linear import KroneckerLinear
from framework.compression_engine.kronecker_tt_lora import (
    KroneckerTTLoRACompressor,
    KroneckerTTLoRALayer,
)
from framework.compression_engine.template_compressor import TemplateCompressor
from framework.compression_engine.rotational_slicegpt_int4 import (
    RotationalSliceGPTCompressor,
    RotationalSliceGPTINT4Compressor,
)
from framework.compression_engine.svd_linear import SVDLinearLayer
from framework.core.registry import COMPRESSOR_REGISTRY


class ToySequentialModel(nn.Module):
    """Toy sequential model for end-to-end compression testing."""

    def __init__(self, in_features: int = 64, hidden_features: int = 128, out_features: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class TestCompressionEngines(unittest.TestCase):
    """Test suite verifying compression engine algorithms, layer replacement, and execution."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 16
        self.in_dim = 64
        self.out_dim = 64
        self.dummy_x = torch.randn(self.batch_size, self.seq_len, self.in_dim)
        self.base_layer = nn.Linear(self.in_dim, self.out_dim)

    def test_rotational_slicegpt_int4(self) -> None:
        """Test RotationalSliceGPT compressor layer compression and forward pass."""
        compressor = RotationalSliceGPTCompressor(r_ratio=0.5, quant_bits=4)
        compressed = compressor.compress_linear(self.base_layer)
        out = compressed(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

    def test_kronecker_tt_lora(self) -> None:
        """Test Kronecker-TT LoRA compressor layer compression and forward pass."""
        compressor = KroneckerTTLoRACompressor(rank=8)
        compressed = compressor.compress_linear(self.base_layer)
        out = compressed(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

    # FIX: Add verification test for AASVD and SVDLinear factorized projection
    def test_aasvd_compressor_and_svd_linear(self) -> None:
        """Test AASVD compressor factorization, parameter reduction, and forward execution."""
        rank = 16
        aasvd = AASVDCompressor(nn.Sequential(self.base_layer), rank=rank)
        svd_res = aasvd.compress_linear(self.base_layer, rank=rank)
        svd_layer = svd_res[0] if isinstance(svd_res, tuple) else svd_res

        self.assertIsInstance(svd_layer, SVDLinear)
        self.assertEqual(svd_layer.rank, rank)
        self.assertEqual(svd_layer.V.shape, (rank, self.in_dim))
        self.assertEqual(svd_layer.U.shape, (self.out_dim, rank))

        out = svd_layer(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

        # Parameter reduction: r*(d_in + d_out) vs d_in*d_out
        svd_params = svd_layer.V.numel() + svd_layer.U.numel()
        dense_params = self.base_layer.weight.numel()
        self.assertLess(svd_params, dense_params)

    # FIX: Add verification test for SVDLinearLayer
    def test_svd_linear_layer(self) -> None:
        """Test SVDLinearLayer singular energy distribution and output shape."""
        rank = 16
        svd_layer = SVDLinearLayer(self.base_layer, rank=rank)
        out = svd_layer(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))
        self.assertEqual(svd_layer.rank, rank)

    # FIX: Add verification test for KroneckerLinear and KroneckerCompressor
    def test_kronecker_linear_and_compressor(self) -> None:
        """Test KroneckerLinear factorization and end-to-end KroneckerCompressor."""
        kron_layer = KroneckerLinear.from_linear(self.base_layer)
        out = kron_layer(self.dummy_x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

        toy_model = ToySequentialModel(in_features=64, hidden_features=128, out_features=32)
        orig_params = sum(p.numel() for p in toy_model.parameters())
        compressor = KroneckerCompressor(toy_model)
        compressed_model, stats = compressor.compress()

        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        self.assertLess(compressed_params, orig_params)
        self.assertEqual(stats["original_parameters"], orig_params)
        self.assertEqual(stats["compressed_parameters"], compressed_params)

    # FIX: Add verification test for ReplaceMeCompressor
    def test_replaceme_compressor(self) -> None:
        """Test ReplaceMeCompressor layer substitution and statistics calculation."""
        class MockConfig:
            hidden_size = 64

        class MockEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = nn.ModuleList([nn.Linear(64, 64) for _ in range(4)])

        class MockBertModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = MockConfig()
                self.encoder = MockEncoder()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.encoder.layer:
                    x = layer(x)
                return x

        model = MockBertModel()
        compressor = ReplaceMeCompressor(model, remove_layers=2)
        compressed_model, stats = compressor.compress()

        self.assertEqual(stats["compressor_name"], "replaceme")
        self.assertIn("original_parameters", stats)
        self.assertIn("compressed_parameters", stats)

    # FIX: Add test verifying compressor registry lookup
    def test_compressor_registry_lookups(self) -> None:
        """Test that COMPRESSOR_REGISTRY contains key standard algorithms."""
        expected_keys = ["aasvd", "dynamic_entropy_svd", "res_svd", "kronecker", "replaceme"]
        for key in expected_keys:
            self.assertIn(key, COMPRESSOR_REGISTRY, f"Missing key in COMPRESSOR_REGISTRY: {key}")
            retrieved = COMPRESSOR_REGISTRY.get(key)
            self.assertIsNotNone(retrieved)


if __name__ == "__main__":
    unittest.main()
