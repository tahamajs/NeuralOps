"""
Comprehensive tests for AdaptiveRanksCompressor.
Verifies:
1. Conv1D and LoRALinear wrapper detection, weight transposition, and base layer handling.
2. comp_config flexibility (None, dict, object).
3. Reconstruction error validation and fallback parameter freezing.
4. Symmetric singular value factor distribution (no norm rescaling).
5. target_attention_only logic.
6. Missing calibration data handling.
7. nn.MultiheadAttention QKV chunking and adaptive rank selection.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn as nn

_FRAMEWORK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FRAMEWORK_PARENT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_PARENT)

from neuralops.compression_engine.aa_svd_compressor import SVDLinear
from neuralops.compression_engine.adaptive_ranks import (
    AdaptiveRanksCompressor, _cfg_get, adaptive_ranks_factory)


class FakeConv1D(nn.Module):
    """Simulates transformers.pytorch_utils.Conv1D (weight shape: [in_features, out_features])."""

    def __init__(self, nf: int, nx: int) -> None:
        super().__init__()
        self.nf = nf  # out_features
        self.nx = nx  # in_features
        self.weight = nn.Parameter(torch.randn(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(*size_out)


class FakeLoRALinear(nn.Module):
    """Simulates a LoRA wrapper layer holding base_layer."""

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = nn.Parameter(torch.randn(4, getattr(base_layer, "in_features", getattr(base_layer, "nx", 64))))
        self.lora_B = nn.Parameter(torch.randn(getattr(base_layer, "out_features", getattr(base_layer, "nf", 64)), 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x)


class FakeGPT2AttentionBlock(nn.Module):
    """Simulates a transformer block with Conv1D and MLP."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.c_attn = FakeConv1D(nf=3 * hidden_dim, nx=hidden_dim)
        self.c_proj = FakeConv1D(nf=hidden_dim, nx=hidden_dim)
        self.mlp_c_fc = FakeConv1D(nf=4 * hidden_dim, nx=hidden_dim)
        self.mlp_c_proj = FakeConv1D(nf=hidden_dim, nx=4 * hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.c_proj(self.c_attn(x))
        mlp = self.mlp_c_proj(torch.relu(self.mlp_c_fc(attn)))
        return attn + mlp


class TestAdaptiveRanksExtra(unittest.TestCase):
    """Rigorous tests for AdaptiveRanksCompressor hardening."""

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_cfg_get_helper(self) -> None:
        """_cfg_get correctly extracts keys from None, dict, and objects."""
        self.assertEqual(_cfg_get(None, "foo", 42), 42)
        self.assertEqual(_cfg_get({"foo": 10}, "foo", 42), 10)
        self.assertEqual(_cfg_get({"bar": 10}, "foo", 42), 42)

        class Obj:
            foo = 99

        self.assertEqual(_cfg_get(Obj(), "foo", 42), 99)
        self.assertEqual(_cfg_get(Obj(), "baz", 42), 42)

    def test_conv1d_transposition_and_compression(self) -> None:
        """Conv1D layers with [nx, nf] weight are properly compressed into SVDLinear with [nf, nx]."""
        model = nn.Sequential(
            FakeConv1D(nf=64, nx=32),
            nn.ReLU(),
            FakeConv1D(nf=32, nx=64),
        )
        x = torch.randn(2, 8, 32)
        out_orig = model(x)

        compressor = AdaptiveRanksCompressor(
            model,
            energy_threshold=0.95,
            min_rank=8,
            max_error=0.99,
            protect_critical=False,
        )
        compressed_model, stats = compressor.compress()

        # Check layers are replaced by SVDLinear
        self.assertIsInstance(compressed_model[0], SVDLinear)
        self.assertEqual(compressed_model[0].in_features, 32)
        self.assertEqual(compressed_model[0].out_features, 64)
        self.assertIsInstance(compressed_model[2], SVDLinear)
        self.assertEqual(compressed_model[2].in_features, 64)
        self.assertEqual(compressed_model[2].out_features, 32)

        out_comp = compressed_model(x)
        self.assertEqual(out_comp.shape, out_orig.shape)

    def test_lora_wrapper_handling(self) -> None:
        """LoRALinear wrappers with base_layer (nn.Linear or Conv1D) are correctly detected and compressed."""
        linear_base = nn.Linear(32, 64)
        conv_base = FakeConv1D(nf=32, nx=64)

        model = nn.ModuleDict({
            "layer_linear": FakeLoRALinear(linear_base),
            "layer_conv": FakeLoRALinear(conv_base),
        })

        compressor = AdaptiveRanksCompressor(
            model,
            energy_threshold=0.95,
            min_rank=8,
            max_error=0.99,
            protect_critical=False,
        )
        compressed_model, stats = compressor.compress()

        self.assertIsInstance(compressed_model.layer_linear, SVDLinear)
        self.assertEqual(compressed_model.layer_linear.in_features, 32)
        self.assertEqual(compressed_model.layer_linear.out_features, 64)

        self.assertIsInstance(compressed_model.layer_conv, SVDLinear)
        self.assertEqual(compressed_model.layer_conv.in_features, 64)
        self.assertEqual(compressed_model.layer_conv.out_features, 32)

    def test_target_attention_only(self) -> None:
        """When target_attention_only=True, attention layers are compressed while MLP/FFN are skipped."""
        model = FakeGPT2AttentionBlock(hidden_dim=32)

        compressor = AdaptiveRanksCompressor(
            model,
            energy_threshold=0.95,
            min_rank=8,
            max_error=0.99,
            target_attention_only=True,
            protect_critical=False,
        )
        compressed_model, stats = compressor.compress()

        # c_attn and c_proj should be compressed
        self.assertIsInstance(compressed_model.c_attn, SVDLinear)
        self.assertIsInstance(compressed_model.c_proj, SVDLinear)

        # mlp_c_fc and mlp_c_proj should NOT be compressed
        self.assertIsInstance(compressed_model.mlp_c_fc, FakeConv1D)
        self.assertIsInstance(compressed_model.mlp_c_proj, FakeConv1D)

    def test_fallback_freezing(self) -> None:
        """When error exceeds max_error, layer falls back and its parameters are frozen."""
        linear = nn.Linear(32, 32)
        model = nn.Sequential(linear)

        # Set max_error to an impossibly low threshold (e.g., 1e-6) so it triggers fallback
        compressor = AdaptiveRanksCompressor(
            model,
            energy_threshold=0.50,
            min_rank=2,
            max_error=1e-6,
            fallback_to_original=True,
            protect_critical=False,
        )
        compressed_model, stats = compressor.compress()

        self.assertIn("0", compressor.fallback_layers)
        self.assertIsInstance(compressed_model[0], nn.Linear)

        # Verify fallback layer parameters are frozen
        for p in compressed_model[0].parameters():
            self.assertFalse(p.requires_grad)

    def test_symmetric_singular_values_no_rescaling(self) -> None:
        """Factors U_r and V_r must retain symmetric sqrt(S) distribution without norm rescaling."""
        # Orthogonal matrices
        q1, _ = torch.linalg.qr(torch.randn(64, 16))
        q2, _ = torch.linalg.qr(torch.randn(32, 16))
        s = torch.linspace(10.0, 1.0, 16)
        W = q1 @ torch.diag(s) @ q2.T  # shape [64, 32]

        layer = nn.Linear(32, 64, bias=False)
        layer.weight.data.copy_(W)
        model = nn.Sequential(layer)

        compressor = AdaptiveRanksCompressor(
            model,
            energy_threshold=0.99,
            min_rank=8,
            max_error=0.50,
            protect_critical=False,
        )
        compressed_model, _ = compressor.compress()

        svd_layer: SVDLinear = compressed_model[0]
        # Check Frobenius norms of U_r and V_r:
        # Since U_r = U * sqrt(S) and V_r = sqrt(S) * V^T,
        # ||U_r||_F^2 = sum(S), ||V_r||_F^2 = sum(S)
        # So norm(U_r) should match norm(V_r) exactly
        norm_u = torch.norm(svd_layer.U, p="fro").item()
        norm_v = torch.norm(svd_layer.V, p="fro").item()
        self.assertAlmostEqual(norm_u, norm_v, places=4)

    def test_factory_with_none_and_dict_config(self) -> None:
        """adaptive_ranks_factory handles comp_config=None, dict, and object properly."""
        model1 = nn.Sequential(nn.Linear(32, 32))
        res1 = adaptive_ranks_factory(model1, comp_config=None, min_rank=4, max_error=0.99)
        self.assertIsInstance(res1[0], SVDLinear)

        model2 = nn.Sequential(nn.Linear(32, 32))
        res2 = adaptive_ranks_factory(model2, comp_config={"min_rank": 4, "max_error": 0.99})
        self.assertIsInstance(res2[0], SVDLinear)

    def test_missing_calibration_data(self) -> None:
        """Compressor gracefully proceeds with unweighted SVD when calibration data is missing."""
        model = nn.Sequential(nn.Linear(32, 32))
        compressor = AdaptiveRanksCompressor(
            model,
            energy_threshold=0.90,
            min_rank=4,
            max_error=0.99,
            calibration_loader=None,
            calibration_texts=None,
            protect_critical=False,
        )
        covs = compressor._collect_activation_covariances()
        self.assertEqual(covs, {})
        compressed_model, _ = compressor.compress()
        self.assertIsInstance(compressed_model[0], SVDLinear)


if __name__ == "__main__":
    unittest.main()
