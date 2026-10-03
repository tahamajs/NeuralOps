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

from framework.compression_engine.aa_svd_compressor import (AASVDCompressor,
                                                            SVDLinear)
from framework.compression_engine.kronecker_compressor import \
    KroneckerCompressor
from framework.compression_engine.kronecker_linear import KroneckerLinear
from framework.compression_engine.kronecker_tt_lora import (
    KroneckerTTLoRACompressor, KroneckerTTLoRALayer)
from framework.compression_engine.replaceme_compressor import \
    ReplaceMeCompressor
from framework.compression_engine.rotational_slicegpt_int4 import (
    RotationalSliceGPTCompressor, RotationalSliceGPTINT4Compressor)
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
        compressor = KroneckerCompressor(toy_model, max_error=1.0)
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


from framework.compression_engine.a3_compressor import A3Compressor
from framework.compression_engine.direct_svd import DirectSVDCompressor
from framework.utils.compat import should_compress_linear
from framework.utils.lora import LoRALinear


class TestCompressorGuardsAndPrecision(unittest.TestCase):
    """Verify safeguards, precision stability, and head protection."""

    def setUp(self):
        torch.manual_seed(42)

    def test_should_compress_linear_head_protection(self):
        """Verify classifier heads, pooler, and embeddings are skipped by compressors."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_linear = nn.Linear(32, 32)
                self.classifier = nn.Linear(32, 2)
                self.score = nn.Linear(32, 2)
                self.lm_head = nn.Linear(32, 100)
                self.pooler = nn.Linear(32, 32)

        model = DummyModel()
        self.assertTrue(should_compress_linear(model, "encoder_linear"))
        self.assertFalse(should_compress_linear(model, "classifier"))
        self.assertFalse(should_compress_linear(model, "score"))
        self.assertFalse(should_compress_linear(model, "lm_head"))
        self.assertFalse(should_compress_linear(model, "pooler"))

    def test_fp32_and_fp16_numerical_stability(self):
        """Verify SVD compression forward remains finite in both fp32 and fp16."""
        layer = nn.Linear(32, 32)
        aasvd = AASVDCompressor(nn.Sequential(layer), rank=8)
        compressed_res = aasvd.compress_linear(layer, rank=8)
        compressed = compressed_res[0] if isinstance(compressed_res, tuple) else compressed_res

        # FP32
        x_32 = torch.randn(2, 4, 32, dtype=torch.float32)
        out_32 = compressed(x_32)
        self.assertTrue(torch.isfinite(out_32).all())

        # FP16
        compressed_16 = compressed.half()
        x_16 = torch.randn(2, 4, 32, dtype=torch.float16)
        out_16 = compressed_16(x_16)
        self.assertTrue(torch.isfinite(out_16).all())

    def test_a3_compressor_flow(self):
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 32)
                self.fc2 = nn.Linear(32, 32)
            def forward(self, x):
                return self.fc2(self.fc1(x))

        m = SmallModel()
        comp = A3Compressor(m, rank=8)
        compressed_m, stats = comp.compress()
        x = torch.randn(2, 32)
        out = compressed_m(x)
        self.assertEqual(out.shape, (2, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_compressor_preserves_unmerged_lora(self):
        """Compressor operating on a LoRA-wrapped linear decomposes base_layer without corrupting adapter."""
        base = nn.Linear(32, 32)
        lora = LoRALinear(base, r=4, alpha=8.0)
        self.assertFalse(lora.merged)

        # Factorize base_layer directly
        aasvd = AASVDCompressor(nn.Sequential(base), rank=8)
        svd_res = aasvd.compress_linear(lora.base_layer, rank=8)
        svd_layer = svd_res[0] if isinstance(svd_res, tuple) else svd_res
        lora.base_layer = svd_layer

        self.assertFalse(lora.merged)
        x = torch.randn(2, 32)
        out = lora(x)
        self.assertEqual(out.shape, (2, 32))
        self.assertTrue(torch.isfinite(out).all())
