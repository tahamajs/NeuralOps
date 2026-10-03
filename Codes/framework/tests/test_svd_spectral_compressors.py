"""
Comprehensive Unit Tests for all SVD, Spectral & Low-Rank Compression Engine Modules.

Tests:
1. AASVDCompressor (aasvd)
2. AdaSVDCompressor (adasvd)
3. DynamicEntropySVDCompressor (dynamic_entropy_svd, entropy_rank_allocator)
4. EntropyBudgetCompressor (entropy_budget)
5. ResSVDCompressor & SpectralResidualLoRA (res_svd, spectral_residual_lora)
6. RPCACompressor (rpca)
7. SwiftSVDCompressor (swift_svd)
8. SwiftSVDActivationCompressor (swift_svd_act)
9. DirectSVDCompressor (direct_svd)
10. SAESSVDCompressorV2 (saes_svd_v2, saes)
11. MGIECompressor (mgie)
12. MGIESVDBudgetCompressor (mgie_budget)
13. ERCCompressor (erc)
14. DobiSVDCompressor (dobi)
15. HybridSVDCompressor & DirectSVDEncoderLayer (hybrid_svd, direct_svd_encoder)
16. SALRCompressor & SALRLayer (salr)
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict

import torch
import torch.nn as nn

_FRAMEWORK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FRAMEWORK_PARENT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_PARENT)

from framework.compression_engine.aa_svd_compressor import (AASVDCompressor,
                                                            SVDLinear,
                                                            aasvd_factory)
from framework.compression_engine.adaptive_compressor import (
    AdaptiveTransformerCompressor, adaptive_svd_factory)
from framework.compression_engine.adaptive_ranks import (
    AdaptiveRanksCompressor, adaptive_ranks_factory)
from framework.compression_engine.adasvd_compressor import (AdaSVDCompressor,
                                                            adasvd_factory)
from framework.compression_engine.direct_svd import (DirectSVDCompressor,
                                                     direct_svd_factory)
from framework.compression_engine.dobi_svd import (DobiSVDCompressor,
                                                   DobiSVDLinearLayer,
                                                   dobi_factory)
from framework.compression_engine.dynamic_rank import (
    DynamicEntropySVDCompressor, DynamicSVDLinear, allocate_entropy_ranks,
    compute_singular_entropy, dynamic_entropy_svd_factory)
from framework.compression_engine.entropy_budget_allocation import (
    EntropyBudgetCompressor, entropy_budget_factory)
from framework.compression_engine.erc_svd import ERCCompressor, erc_factory
from framework.compression_engine.hybrid_svd_layer import (
    DirectSVDEncoderLayer, HybridSVDCompressor, SVDSingleHead,
    hybrid_svd_factory)
from framework.compression_engine.mgie_svd import MGIECompressor, mgie_factory
from framework.compression_engine.mgie_svd_compressor import (
    MGIESVDBudgetCompressor, mgie_budget_factory)
from framework.compression_engine.res_svd import (
    ResSVDCompressor, ResSVDLinear, SpectralResidualLoRA, res_svd_factory,
    spectral_residual_lora_factory)
from framework.compression_engine.rpca_compressor import (
    RPCACompressor, rpca_compressor_factory)
from framework.compression_engine.saes_svd_v2 import (SAESCompressor,
                                                      SAESSVDCompressorV2,
                                                      saes_svd_v2_factory)
from framework.compression_engine.salr import (SALRCompressor, SALRLayer,
                                               salr_factory)
from framework.compression_engine.svd_linear import (SVDLinearCompressor,
                                                     SVDLinearLayer,
                                                     svd_linear_factory)
from framework.compression_engine.swift_svd import (SwiftSVDCompressor,
                                                    swift_svd_factory)
from framework.compression_engine.swift_svd_act import (
    SwiftSVDActivationCompressor, swift_svd_act_factory)
from framework.core.registry import COMPRESSOR_REGISTRY


class MockTransformerBlock(nn.Module):
    """Mock multi-head transformer block with Linear projections and FFN."""

    def __init__(self, hidden_size: int = 64, intermediate_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        with torch.no_grad():
            for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.fc1, self.fc2]:
                out_f, in_f = m.weight.shape
                min_dim = min(out_f, in_f)
                u = torch.randn(out_f, min_dim)
                u, _ = torch.linalg.qr(u)
                v = torch.randn(in_f, min_dim)
                v, _ = torch.linalg.qr(v)
                s = torch.exp(-0.25 * torch.arange(min_dim, dtype=torch.float32))
                m.weight.data.copy_(u @ torch.diag(s) @ v.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        attn_out = torch.matmul(torch.softmax(attn, dim=-1), v)
        x = self.norm1(res + self.out_proj(attn_out))
        res = x
        ffn = self.fc2(self.act(self.fc1(x)))
        return self.norm2(res + ffn)


class MockSequentialModel(nn.Module):
    """Mock sequential feed-forward model."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 128, out_dim: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class TestSVDSpectralCompressionEngines(unittest.TestCase):
    """Exhaustive unit test suite verifying all 13 SVD and spectral compression engines."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 16
        self.hidden_size = 64
        self.x = torch.randn(self.batch_size, self.seq_len, self.hidden_size)

    def _verify_compressed_model(self, comp_model: nn.Module, orig_params: int, stats: Dict[str, Any]) -> None:
        """Helper to verify model forward pass and parameter stats."""
        out = comp_model(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertFalse(torch.isnan(out).any(), "Compressed model produced NaNs in output")
        self.assertIn("original_parameters", stats)
        self.assertIn("compressed_parameters", stats)
        self.assertIn("compression_ratio", stats)

    # 1. AASVD Compressor
    def test_aasvd_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = AASVDCompressor(model, rank=16)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 2. AdaSVD Compressor
    def test_adasvd_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = AdaSVDCompressor(model, rank=16, num_iters=3)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 3. Dynamic Entropy SVD Compressor & Rank Allocator
    def test_dynamic_entropy_svd_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = DynamicEntropySVDCompressor(model, min_rank=8, max_rank=32, target_rank=16)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        self.assertIn("layer_entropies", stats)
        self.assertIn("layer_ranks", stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 4. Entropy Budget Compressor
    def test_entropy_budget_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = EntropyBudgetCompressor(model, total_budget_ratio=0.5, rank=16)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 5. Res-SVD Compressor & Spectral Residual LoRA
    def test_res_svd_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = ResSVDCompressor(model, base_rank=16, residual_rank=4)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        self.assertIn("trainable_parameters", stats)
        self.assertIn("frozen_parameters", stats)

    def test_spectral_residual_lora_alias(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        compressor = SpectralResidualLoRA(model, base_rank=16, residual_rank=4)
        comp_model, stats = compressor.compress()
        out = comp_model(self.x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))

    # 6. RPCA Compressor
    def test_rpca_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = RPCACompressor(model, rank=16, max_iter=3)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 7. Swift-SVD Compressor
    def test_swift_svd_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = SwiftSVDCompressor(model, rank=16)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 8. Swift-SVD Activation-Aware Compressor
    def test_swift_svd_act_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        dummy_loader = [self.x]
        compressor = SwiftSVDActivationCompressor(model, calibration_loader=dummy_loader, rank=16)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 9. Direct SVD Compressor
    def test_direct_svd_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = DirectSVDCompressor(model, rank=16)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 10. SAES-SVD v2 Compressor
    def test_saes_svd_v2_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = SAESSVDCompressorV2(model, rank=16, correcting_steps=2)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 11. MGIE Compressor
    def test_mgie_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = MGIECompressor(model, rank=16)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 12. MGIE Budget Compressor
    def test_mgie_budget_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = MGIESVDBudgetCompressor(model, total_rank_budget=100)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 13. ERC Compressor (Error-Controlled SVD)
    def test_erc_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = ERCCompressor(model, allowed_error=0.1, rank=16)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 14. Dobi-SVD Compressor
    def test_dobi_svd_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = DobiSVDCompressor(model, rank=16)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 15. Hybrid SVD Compressor & DirectSVDEncoderLayer
    def test_hybrid_svd_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = HybridSVDCompressor(model, svd_rank=16)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)

    # 16. SALR Compressor & SALRLayer
    def test_salr_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = SALRCompressor(model, rank=16, sparsity=0.5)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)
        comp_params = sum(p.numel() for p in comp_model.parameters())
        self.assertLess(comp_params, orig_params)

    # 17. Adaptive Transformer Compressor & Adaptive Ranks
    def test_adaptive_transformer_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = AdaptiveTransformerCompressor(model, energy_threshold=0.9, min_rank=16)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)

    def test_adaptive_ranks_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = AdaptiveRanksCompressor(model, energy_threshold=0.9, min_rank=16)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)

    # 18. SVD Linear Compressor & Layer
    def test_svd_linear_compressor(self) -> None:
        model = MockTransformerBlock(self.hidden_size)
        orig_params = sum(p.numel() for p in model.parameters())
        compressor = SVDLinearCompressor(model, rank=16)
        comp_model, stats = compressor.compress()
        self._verify_compressed_model(comp_model, orig_params, stats)

    # 19. Exhaustive Verification of All Factory Functions
    def test_all_svd_factory_functions(self) -> None:
        factories = [
            ("aasvd_factory", aasvd_factory),
            ("adasvd_factory", adasvd_factory),
            ("adaptive_svd_factory", adaptive_svd_factory),
            ("adaptive_ranks_factory", adaptive_ranks_factory),
            ("direct_svd_factory", direct_svd_factory),
            ("dobi_factory", dobi_factory),
            ("dynamic_entropy_svd_factory", dynamic_entropy_svd_factory),
            ("entropy_budget_factory", entropy_budget_factory),
            ("erc_factory", erc_factory),
            ("hybrid_svd_factory", hybrid_svd_factory),
            ("mgie_factory", mgie_factory),
            ("mgie_budget_factory", mgie_budget_factory),
            ("res_svd_factory", res_svd_factory),
            ("spectral_residual_lora_factory", spectral_residual_lora_factory),
            ("rpca_compressor_factory", rpca_compressor_factory),
            ("saes_svd_v2_factory", saes_svd_v2_factory),
            ("salr_factory", salr_factory),
            ("svd_linear_factory", svd_linear_factory),
            ("swift_svd_factory", swift_svd_factory),
            ("swift_svd_act_factory", swift_svd_act_factory),
        ]

        for name, fn in factories:
            model = MockTransformerBlock(self.hidden_size)
            # Verify factory accepts (model, comp_config=None, **kwargs)
            comp_model = fn(model, comp_config=None, rank=16)
            self.assertIsInstance(comp_model, nn.Module, f"Factory {name} did not return nn.Module")
            out = comp_model(self.x)
            self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size), f"Factory {name} output shape mismatch")
            self.assertFalse(torch.isnan(out).any(), f"Factory {name} output contained NaNs")

    # 20. Registry Lookups
    def test_all_svd_registry_lookups(self) -> None:
        expected_keys = [
            "aasvd",
            "adasvd",
            "adaptive_svd",
            "adaptive_ranks",
            "dynamic_entropy_svd",
            "entropy_rank_allocator",
            "entropy_budget",
            "res_svd",
            "spectral_residual_lora",
            "rpca",
            "swift_svd",
            "swift_svd_act",
            "direct_svd",
            "saes_svd_v2",
            "saes",
            "mgie",
            "mgie_budget",
            "erc",
            "dobi",
            "hybrid_svd",
            "direct_svd_encoder",
            "salr",
            "svd_linear_compressor",
        ]
        for key in expected_keys:
            comp_cls_or_fn = COMPRESSOR_REGISTRY.get(key)
            self.assertIsNotNone(comp_cls_or_fn, f"Registry key '{key}' not found in COMPRESSOR_REGISTRY")


if __name__ == "__main__":
    unittest.main()
