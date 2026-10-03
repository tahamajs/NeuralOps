"""
Unit Tests for State-of-the-Art (SOTA) Compression Engines:
1. Dynamic Entropy SVD Compressor & Dynamic Rank Allocator (dynamic_rank.py).
2. Spectral Residual LoRA / Res-SVD Compressor (res_svd.py).
3. Central Compressor Registry & PEFT Aliases.
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure framework is in search path
_FRAMEWORK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FRAMEWORK_PARENT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_PARENT)

from framework.compression_engine.dynamic_rank import (
    DynamicEntropySVDCompressor, DynamicSVDLinear, allocate_entropy_ranks,
    compute_singular_entropy)
from framework.compression_engine.res_svd import (ResSVDCompressor,
                                                  ResSVDLinear,
                                                  SpectralResidualLoRA)
from framework.core.registry import COMPRESSOR_REGISTRY


class ToyTransformerBlock(nn.Module):
    """Multi-layer MLP / Attention-like block for end-to-end compressor testing."""

    def __init__(self, hidden_size: int = 64, intermediate_size: int = 128) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.mlp_in = nn.Linear(hidden_size, intermediate_size)
        self.mlp_out = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified attention + MLP forward
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / 8.0, dim=-1)
        attn_out = self.out_proj(torch.matmul(attn, v))
        h = x + attn_out
        mlp_h = torch.relu(self.mlp_in(h))
        out = h + self.mlp_out(mlp_h)
        return out


class ToyMultiLayerModel(nn.Module):
    """Multi-layer model containing several transformer blocks."""

    def __init__(self, num_layers: int = 3, hidden_size: int = 64) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ToyTransformerBlock(hidden_size=hidden_size) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h)
        logits = self.head(h[:, 0, :])
        return logits


class TestDynamicEntropySVD(unittest.TestCase):
    """Test suite for Dynamic Entropy SVD and rank allocation."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.batch_size = 4
        self.seq_len = 16
        self.hidden_dim = 64
        self.x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)

    # -------------------------------------------------------------------------
    # 1. Singular Entropy Mathematics Tests
    # -------------------------------------------------------------------------

    def test_singular_entropy_uniform_spectrum(self) -> None:
        """Uniform singular spectrum should achieve maximal entropy ln(K)."""
        k = 16
        uniform_spectrum = torch.ones(k)
        entropy = compute_singular_entropy(uniform_spectrum, normalize=False)
        expected_entropy = math.log(k)
        self.assertAlmostEqual(entropy, expected_entropy, places=5)

    def test_singular_entropy_single_spike(self) -> None:
        """Singular spectrum with 1-hot spike should have zero entropy."""
        spectrum = torch.zeros(16)
        spectrum[0] = 100.0
        entropy = compute_singular_entropy(spectrum, normalize=False)
        self.assertAlmostEqual(entropy, 0.0, places=5)

    def test_singular_entropy_normalization(self) -> None:
        """Normalized singular entropy must lie strictly in [0.0, 1.0]."""
        spectrum = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
        norm_entropy = compute_singular_entropy(spectrum, normalize=True)
        self.assertGreaterEqual(norm_entropy, 0.0)
        self.assertLessEqual(norm_entropy, 1.0)

    def test_singular_entropy_2d_covariance_and_weight(self) -> None:
        """Compute entropy on 2D covariance and 2D rectangular weight matrices."""
        # Square symmetric covariance
        x_samples = torch.randn(100, 32)
        cov = x_samples.t() @ x_samples / 100.0
        cov_entropy = compute_singular_entropy(cov)
        self.assertGreater(cov_entropy, 0.0)
        self.assertFalse(math.isnan(cov_entropy))

        # Rectangular weight matrix
        w = torch.randn(64, 32)
        w_entropy = compute_singular_entropy(w)
        self.assertGreater(w_entropy, 0.0)
        self.assertFalse(math.isnan(w_entropy))

    def test_singular_entropy_all_zeros_stability(self) -> None:
        """All-zeros matrix should return 0.0 without NaN or exceptions."""
        zeros_mat = torch.zeros(32, 32)
        entropy = compute_singular_entropy(zeros_mat)
        self.assertEqual(entropy, 0.0)

    # -------------------------------------------------------------------------
    # 2. Dynamic Rank Budget Allocator Tests
    # -------------------------------------------------------------------------

    def test_rank_allocation_proportional_and_bounded(self) -> None:
        """Higher entropy layers must receive higher rank, respecting [min_rank, max_rank]."""
        entropies = {
            "layer_low": 0.5,
            "layer_mid": 2.0,
            "layer_high": 4.5,
        }
        min_r = 8
        max_r = 32
        total_budget = 48  # Average 16 per layer

        ranks = allocate_entropy_ranks(
            entropies=entropies,
            min_rank=min_r,
            max_rank=max_r,
            total_rank_budget=total_budget,
        )

        # Monotonicity check
        self.assertLessEqual(ranks["layer_low"], ranks["layer_mid"])
        self.assertLessEqual(ranks["layer_mid"], ranks["layer_high"])

        # Boundary checks
        for name, r in ranks.items():
            self.assertGreaterEqual(r, min_r)
            self.assertLessEqual(r, max_r)

        # Total budget sum check
        self.assertEqual(sum(ranks.values()), total_budget)

    def test_rank_allocation_physical_dimension_clamping(self) -> None:
        """Layers with small physical dimensions cannot exceed min(d_in, d_out)."""
        entropies = {"layer_tiny": 5.0, "layer_large": 5.0}
        layer_dims = {"layer_tiny": (4, 4), "layer_large": (64, 64)}

        ranks = allocate_entropy_ranks(
            entropies=entropies,
            min_rank=16,
            max_rank=64,
            total_rank_budget=40,
            layer_dims=layer_dims,
        )
        self.assertLessEqual(ranks["layer_tiny"], 4)
        self.assertGreaterEqual(ranks["layer_large"], 16)

    def test_rank_allocation_parameter_budget(self) -> None:
        """Parameter budget water-filling produces valid ranks under target ratio."""
        layer_dims = {
            "l1": (64, 64),
            "l2": (64, 64),
            "l3": (64, 64),
        }
        entropies = {"l1": 1.0, "l2": 2.0, "l3": 3.0}

        ranks = allocate_entropy_ranks(
            entropies=entropies,
            min_rank=4,
            max_rank=32,
            target_param_ratio=0.5,
            layer_dims=layer_dims,
        )

        for name, r in ranks.items():
            self.assertGreaterEqual(r, 4)
            self.assertLessEqual(r, 32)

    # -------------------------------------------------------------------------
    # 3. DynamicSVDLinear Layer Tests
    # -------------------------------------------------------------------------

    def test_dynamic_svd_linear_forward_and_backward(self) -> None:
        """Factorized DynamicSVDLinear computes correct shapes and supports autograd."""
        in_f = 64
        out_f = 128
        rank = 16
        layer = DynamicSVDLinear(in_features=in_f, out_features=out_f, rank=rank, bias=True)

        self.assertEqual(layer.V.shape, (rank, in_f))
        self.assertEqual(layer.U.shape, (out_f, rank))
        self.assertEqual(layer.bias.shape, (out_f,))

        out = layer(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, out_f))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(layer.V.grad)
        self.assertIsNotNone(layer.U.grad)
        self.assertIsNotNone(layer.bias.grad)

    def test_dynamic_svd_linear_from_dense(self) -> None:
        """Conversion from dense nn.Linear preserves close representation."""
        dense = nn.Linear(64, 64, bias=True)
        svd_layer = DynamicSVDLinear.from_linear(dense, rank=32)

        dense_out = dense(self.x)
        svd_out = svd_layer(self.x)
        self.assertEqual(dense_out.shape, svd_out.shape)

        # Truncated SVD error check
        rel_diff = torch.norm(dense_out - svd_out) / torch.norm(dense_out)
        self.assertLess(rel_diff.item(), 0.5)

    def test_dynamic_svd_linear_with_covariance(self) -> None:
        """Activation-aware SVD initialization with empirical covariance matrix."""
        dense = nn.Linear(64, 64)
        x_calib = torch.randn(200, 64)
        cov = x_calib.t() @ x_calib / 200.0

        svd_layer = DynamicSVDLinear.from_linear(dense, rank=16, covariance=cov)
        out = svd_layer(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, 64))

    # -------------------------------------------------------------------------
    # 4. DynamicEntropySVDCompressor End-to-End Tests
    # -------------------------------------------------------------------------

    def test_dynamic_entropy_svd_compressor_end_to_end(self) -> None:
        """End-to-end compression of multi-layer model with calibration data."""
        model = ToyMultiLayerModel(num_layers=2, hidden_size=64)
        orig_params = sum(p.numel() for p in model.parameters())

        # Calibration dataset
        dummy_data = torch.randn(16, 8, 64)
        loader = DataLoader(TensorDataset(dummy_data), batch_size=4)

        compressor = DynamicEntropySVDCompressor(
            model=model,
            min_rank=8,
            max_rank=32,
            target_rank=16,
            calibration_loader=loader,
        )

        compressed_model, stats = compressor.compress()

        # Check parameter reduction
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        self.assertLess(compressed_params, orig_params)
        self.assertEqual(stats["original_parameters"], orig_params)
        self.assertEqual(stats["compressed_parameters"], compressed_params)
        self.assertIn("layer_entropies", stats)
        self.assertIn("layer_ranks", stats)

        # Check forward pass on compressed model
        test_in = torch.randn(2, 8, 64)
        logits = compressed_model(test_in)
        self.assertEqual(logits.shape, (2, 10))
        self.assertFalse(torch.isnan(logits).any())


class TestResSVD(unittest.TestCase):
    """Test suite for Spectral Residual LoRA (Res-SVD)."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.batch_size = 4
        self.seq_len = 16
        self.in_dim = 64
        self.out_dim = 64
        self.x = torch.randn(self.batch_size, self.seq_len, self.in_dim)
        self.dense = nn.Linear(self.in_dim, self.out_dim, bias=True)

    # -------------------------------------------------------------------------
    # 1. ResSVDLinear Layer Tests
    # -------------------------------------------------------------------------

    def test_res_svd_linear_construction_and_grad_freeze(self) -> None:
        """ResSVDLinear freezes base weights and enables grad on adapter matrices."""
        base_r = 16
        res_r = 4
        res_layer = ResSVDLinear.from_linear(
            linear=self.dense,
            base_rank=base_r,
            residual_rank=res_r,
            scaling=1.0,
            factorize_base=True,
            freeze_base=True,
        )

        # Base must be frozen
        self.assertFalse(res_layer.base_V.requires_grad)
        self.assertFalse(res_layer.base_U.requires_grad)

        # Adapters must be trainable
        self.assertTrue(res_layer.lora_A.requires_grad)
        self.assertTrue(res_layer.lora_B.requires_grad)

        self.assertEqual(res_layer.base_V.shape, (base_r, self.in_dim))
        self.assertEqual(res_layer.base_U.shape, (self.out_dim, base_r))
        self.assertEqual(res_layer.lora_A.shape, (res_r, self.in_dim))
        self.assertEqual(res_layer.lora_B.shape, (self.out_dim, res_r))

    def test_spectral_residual_tail_reconstruction_error(self) -> None:
        """Res-SVD adapter initialized from tail achieves lower error than base alone."""
        W = self.dense.weight.data.float()
        base_r = 8
        res_r = 8

        # Base only layer
        base_only = ResSVDLinear.from_linear(
            linear=self.dense,
            base_rank=base_r,
            residual_rank=0,
        )

        # Base + Spectral Residual LoRA
        res_svd = ResSVDLinear.from_linear(
            linear=self.dense,
            base_rank=base_r,
            residual_rank=res_r,
            scaling=1.0,
        )

        err_base = base_only.get_residual_error_norm(W)
        err_res = res_svd.get_residual_error_norm(W)

        # Tail inclusion must strictly reduce reconstruction error
        self.assertLess(err_res, err_base)

    def test_res_svd_linear_forward_and_backward(self) -> None:
        """ResSVDLinear forward pass and gradient backpropagation to adapters only."""
        res_layer = ResSVDLinear.from_linear(
            linear=self.dense,
            base_rank=16,
            residual_rank=4,
            scaling=1.0,
        )

        out = res_layer(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

        loss = out.sum()
        loss.backward()

        # Base factors should have no gradients
        self.assertIsNone(res_layer.base_V.grad)
        self.assertIsNone(res_layer.base_U.grad)

        # LoRA adapters must have valid gradients
        self.assertIsNotNone(res_layer.lora_A.grad)
        self.assertIsNotNone(res_layer.lora_B.grad)

    def test_res_svd_linear_merge_weights(self) -> None:
        """Weight merging produces standard nn.Linear with identical output."""
        res_layer = ResSVDLinear.from_linear(
            linear=self.dense,
            base_rank=16,
            residual_rank=8,
            scaling=1.0,
        )

        merged_linear = res_layer.merge_weights()
        self.assertIsInstance(merged_linear, nn.Linear)

        out_res = res_layer(self.x)
        out_merged = merged_linear(self.x)
        self.assertTrue(torch.allclose(out_res, out_merged, atol=1e-4))

    def test_res_svd_dense_base_mode(self) -> None:
        """ResSVDLinear supports non-factorized base mode (factorize_base=False)."""
        res_layer = ResSVDLinear.from_linear(
            linear=self.dense,
            base_rank=16,
            residual_rank=4,
            factorize_base=False,
            freeze_base=True,
        )

        self.assertFalse(res_layer.base_weight.requires_grad)
        self.assertTrue(res_layer.lora_A.requires_grad)
        out = res_layer(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

    # -------------------------------------------------------------------------
    # 2. ResSVDCompressor End-to-End Tests
    # -------------------------------------------------------------------------

    def test_res_svd_compressor_end_to_end(self) -> None:
        """End-to-end compression with ResSVDCompressor and parameter stats check."""
        model = ToyMultiLayerModel(num_layers=2, hidden_size=64)
        orig_params = sum(p.numel() for p in model.parameters())

        compressor = ResSVDCompressor(
            model=model,
            base_rank=16,
            residual_rank=4,
            scaling=1.0,
            factorize_base=True,
            freeze_base=True,
        )

        compressed_model, stats = compressor.compress()

        self.assertIn("trainable_parameters", stats)
        self.assertIn("frozen_parameters", stats)
        self.assertIn("replaced_layers", stats)
        self.assertGreater(stats["frozen_parameters"], 0)
        self.assertGreater(stats["trainable_parameters"], 0)

        # Trainable parameters should be a small fraction (PEFT)
        self.assertLess(stats["trainable_parameters"], stats["frozen_parameters"])

        # Forward pass verification
        test_x = torch.randn(2, 8, 64)
        logits = compressed_model(test_x)
        self.assertEqual(logits.shape, (2, 10))
        self.assertFalse(torch.isnan(logits).any())

    def test_spectral_residual_lora_alias_class(self) -> None:
        """SpectralResidualLoRA alias class works interchangeably."""
        model = ToyMultiLayerModel(num_layers=1, hidden_size=64)
        compressor = SpectralResidualLoRA(
            model=model,
            base_rank=16,
            residual_rank=4,
        )
        compressed_model, stats = compressor.compress()
        self.assertEqual(stats["compressor_name"], "spectral_residual_lora")


class TestCompressorRegistrations(unittest.TestCase):
    """Test suite verifying framework registry integration for all compressors."""

    def test_registry_lookups(self) -> None:
        """Verify dynamic_entropy_svd, entropy_rank_allocator, res_svd, and spectral_residual_lora."""
        # 1. dynamic_entropy_svd
        self.assertIn("dynamic_entropy_svd", COMPRESSOR_REGISTRY)
        cls_dynamic = COMPRESSOR_REGISTRY.get("dynamic_entropy_svd")
        self.assertTrue(issubclass(cls_dynamic, DynamicEntropySVDCompressor))

        # 2. entropy_rank_allocator
        self.assertIn("entropy_rank_allocator", COMPRESSOR_REGISTRY)
        cls_alloc = COMPRESSOR_REGISTRY.get("entropy_rank_allocator")
        self.assertTrue(issubclass(cls_alloc, DynamicEntropySVDCompressor))

        # 3. res_svd
        self.assertIn("res_svd", COMPRESSOR_REGISTRY)
        cls_res = COMPRESSOR_REGISTRY.get("res_svd")
        self.assertTrue(issubclass(cls_res, ResSVDCompressor))

        # 4. spectral_residual_lora
        self.assertIn("spectral_residual_lora", COMPRESSOR_REGISTRY)
        cls_spectral = COMPRESSOR_REGISTRY.get("spectral_residual_lora")
        self.assertTrue(issubclass(cls_spectral, ResSVDCompressor))


if __name__ == "__main__":
    unittest.main()
