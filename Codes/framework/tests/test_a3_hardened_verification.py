"""
Verification test script for A3 Compressor hardening.
Validates:
1. When error > max_error and fallback_to_original=True, compress_linear returns None.
2. When no calibration data is available, compress works cleanly using unweighted SVD.
3. Memory is freed after activation collection (activations dictionary cleared).
4. Scale normalization flag behavior (normalize_scale True vs False).
5. Conv1D architecture detection and transposition.
"""

import unittest
import torch
import torch.nn as nn

from framework.compression_engine.a3_compressor import (
    A3Compressor,
    A3FactorizedLinear,
    a3_compressor_factory,
)


class TestA3CompressorHardening(unittest.TestCase):

    def test_fallback_when_error_exceeds_threshold(self):
        """When reconstruction error exceeds max_error and fallback_to_original=True, returns None."""
        layer = nn.Linear(64, 64)
        # Force low rank 1 on random Gaussian weights, with tiny max_error 0.001
        compressor = A3Compressor(
            nn.Sequential(layer),
            rank=1,
            max_error=0.001,
            fallback_to_original=True,
            max_retries=0,
        )
        factorized, err, r = compressor.compress_linear(layer, name="test_linear", rank=1)
        self.assertIsNone(factorized, "Expected factorized layer to be None when error exceeds max_error")
        self.assertGreater(err, 0.001)

    def test_unweighted_svd_when_no_calibration_data(self):
        """When no calibration data is available, compress works smoothly using unweighted SVD."""
        model = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        # Without calibration_loader, calibration_texts, or tokenizer
        compressor = A3Compressor(
            model,
            rank=8,
            max_error=0.85,
            fallback_to_original=True,
        )
        compressed_model, stats = compressor.compress()

        self.assertEqual(stats["factorized_layers"], 2)
        self.assertEqual(stats["skipped_layers"], 0)
        self.assertFalse(stats["use_fisher"])
        self.assertFalse(stats["recalibrate_ln"])

        # Forward pass verification
        x = torch.randn(4, 32)
        out = compressed_model(x)
        self.assertEqual(out.shape, (4, 16))
        self.assertTrue(torch.isfinite(out).all())

    def test_memory_freed_after_activation_collection(self):
        """Activation dictionary is cleared after collecting covariances and energy proxies."""
        model = nn.Sequential(nn.Linear(16, 16))

        dummy_loader = [torch.randn(2, 16) for _ in range(3)]
        compressor = A3Compressor(
            model,
            rank=4,
            use_fisher=True,
            calibration_loader=dummy_loader,
        )
        cov_map, energy_map = compressor._collect_activations_and_energy()

        self.assertTrue(compressor.activations_cleared)
        self.assertIn("0", cov_map)
        self.assertIn("0", energy_map)

    def test_normalize_scale_options(self):
        """Verifies normalize_scale=True and normalize_scale=False behavior."""
        layer = nn.Linear(32, 32)
        dummy_energy = torch.ones(32) * 5.0

        # With scale normalization (unit RMS)
        comp_norm = A3Compressor(nn.Sequential(layer), rank=8, normalize_scale=True)
        layer_norm, err_norm, _ = comp_norm.compress_linear(
            layer, activation_energy=dummy_energy, rank=8
        )
        self.assertIsNotNone(layer_norm)

        # Without scale normalization
        comp_raw = A3Compressor(nn.Sequential(layer), rank=8, normalize_scale=False)
        layer_raw, err_raw, _ = comp_raw.compress_linear(
            layer, activation_energy=dummy_energy, rank=8
        )
        self.assertIsNotNone(layer_raw)

    def test_conv1d_detection_and_transposition(self):
        """Verifies robust Conv1D module support."""
        class MockConv1D(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(in_features, out_features))
                self.bias = nn.Parameter(torch.zeros(out_features))

        conv = MockConv1D(32, 64)
        comp = A3Compressor(nn.Sequential(conv), rank=8, max_error=0.85)
        fact_layer, err, r = comp.compress_linear(conv, rank=8)

        self.assertIsNotNone(fact_layer)
        self.assertEqual(fact_layer.in_features, 32)
        self.assertEqual(fact_layer.out_features, 64)
        self.assertEqual(fact_layer.rank, 8)

        x = torch.randn(2, 5, 32)
        out = fact_layer(x)
        self.assertEqual(out.shape, (2, 5, 64))


if __name__ == "__main__":
    unittest.main()
