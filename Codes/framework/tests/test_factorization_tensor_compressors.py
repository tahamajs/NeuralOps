"""
Comprehensive Unit Tests for Tensor Decomposition, Kronecker & Factorization Compression Engine.

Covers:
1. Kronecker (KroneckerLinear, KroneckerCompressor, KroneckerTTLoRALayer, KroneckerTTLoRACompressor)
2. Tensor-Train (TTLinearLayer, TTCompressor)
3. CUR Cross-Approximation (CURLinear, CURCompressor, CURMoECompressor)
4. Non-Negative Matrix Factorization (NMFLinear, NMFCompressor)
5. Subspace Projection (SubspaceLinear, SubspaceProjectionCompressor)
6. HashedNet (HashedLinear, HashedNetCompressor)
7. COMPOT (COMPOTLinear, COMPOTCompressor)
8. CR Interpolative Decomposition (CRLinear, CRCompressor)
9. FRanDI (FRanDILinear, FRanDICompressor)
10. GreCoDeC (GreCoDeCLinear, GreCoDeCCompressor)
11. Registry Factory invocations
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict, List

import torch
import torch.nn as nn

_FRAMEWORK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FRAMEWORK_PARENT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_PARENT)

from framework.compression_engine.compot_compressor import (COMPOTCompressor,
                                                            COMPOTLinear)
from framework.compression_engine.cr_compressor import CRCompressor, CRLinear
from framework.compression_engine.cur_compressor import CURCompressor
from framework.compression_engine.cur_linear import (CURLinear,
                                                     compute_energy_scores,
                                                     compute_leverage_scores)
from framework.compression_engine.cur_moe_compressor import CURMoECompressor
from framework.compression_engine.frandi_compressor import (FRanDICompressor,
                                                            FRanDILinear)
from framework.compression_engine.grecodec_compressor import (
    GreCoDeCCompressor, GreCoDeCLinear)
from framework.compression_engine.hashed_linear import HashedLinear
from framework.compression_engine.hashed_net_compressor import \
    HashedNetCompressor
from framework.compression_engine.kronecker_compressor import \
    KroneckerCompressor
from framework.compression_engine.kronecker_linear import KroneckerLinear
from framework.compression_engine.kronecker_tt_lora import (
    KroneckerTTLoRACompressor, KroneckerTTLoRALayer)
from framework.compression_engine.nmf_compressor import (NMFCompressor,
                                                         NMFLinear)
from framework.compression_engine.subspace_projection import (
    SubspaceLinear, SubspaceProjectionCompressor)
from framework.compression_engine.tt_compressor import TTCompressor
from framework.compression_engine.tt_linear import TTLinearLayer
from framework.core.registry import COMPRESSOR_REGISTRY


class ToySequentialModel(nn.Module):
    """Toy multi-layer neural network for end-to-end model compression testing."""

    def __init__(self, in_features: int = 64, hidden_features: int = 128, out_features: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class ToyMoEModel(nn.Module):
    """Toy Mixture of Experts model for testing MoE compressor."""

    def __init__(self, num_experts: int = 4, in_features: int = 64, out_features: int = 64) -> None:
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Linear(out_features, in_features),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for exp in self.experts:
            out = out + exp(x)
        return out / len(self.experts)


class TestTensorAndFactorizationCompressors(unittest.TestCase):
    """Comprehensive test suite for all 9 Tensor & Factorization compression engines."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 8
        self.in_dim = 64
        self.out_dim = 64
        self.dummy_input = torch.randn(self.batch_size, self.seq_len, self.in_dim)
        self.base_layer = nn.Linear(self.in_dim, self.out_dim)

    # 1. Kronecker Tests
    def test_kronecker_linear_and_compressor(self) -> None:
        # Layer level
        kron_layer = KroneckerLinear.from_linear(self.base_layer)
        out = kron_layer(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

        # Model level
        model = ToySequentialModel(in_features=64, hidden_features=128, out_features=32)
        compressor = KroneckerCompressor(model)
        comp_model, stats = compressor.compress()
        x = torch.randn(self.batch_size, 16, 64)
        out_m = comp_model(x)
        self.assertEqual(out_m.shape, (self.batch_size, 16, 32))
        self.assertIn("compression_ratio", stats)

    # 2. Kronecker-TT-LoRA Tests
    def test_kronecker_tt_lora(self) -> None:
        comp = KroneckerTTLoRACompressor(rank=8)
        layer = comp.compress_linear(self.base_layer)
        out = layer(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

        # Test rank adaptation
        layer.adapt_tt_ranks(delta_r=1)

        # Full model
        model = ToySequentialModel()
        c_model, stats = KroneckerTTLoRACompressor(model, rank=8).compress()
        x = torch.randn(2, 4, 64)
        out_m = c_model(x)
        self.assertEqual(out_m.shape, (2, 4, 32))

    # 3. Tensor-Train (TT) Tests
    def test_tt_linear_and_compressor(self) -> None:
        tt_layer = TTLinearLayer.from_linear(self.base_layer, tt_rank=8)
        out = tt_layer(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

        model = ToySequentialModel()
        c_model, stats = TTCompressor(model, tt_rank=8).compress()
        x = torch.randn(self.batch_size, 64)
        out_m = c_model(x)
        self.assertEqual(out_m.shape, (self.batch_size, 32))
        self.assertIn("tt_rank", stats)

    # 4. CUR Cross-Approximation Tests
    def test_cur_linear_and_compressor(self) -> None:
        # Leverage scores
        cur_lev = CURLinear.from_linear(self.base_layer, rank=16, method="leverage")
        out_lev = cur_lev(self.dummy_input)
        self.assertEqual(out_lev.shape, (self.batch_size, self.seq_len, self.out_dim))

        # Energy sampling
        cur_nrg = CURLinear.from_linear(self.base_layer, rank=16, method="energy")
        out_nrg = cur_nrg(self.dummy_input)
        self.assertEqual(out_nrg.shape, (self.batch_size, self.seq_len, self.out_dim))

        # Score computation functions
        col_s, row_s = compute_leverage_scores(self.base_layer.weight.data, rank=8)
        self.assertEqual(col_s.numel(), self.in_dim)
        self.assertEqual(row_s.numel(), self.out_dim)

        col_e, row_e = compute_energy_scores(self.base_layer.weight.data)
        self.assertEqual(col_e.numel(), self.in_dim)
        self.assertEqual(row_e.numel(), self.out_dim)

        # Compressor
        model = ToySequentialModel()
        c_model, stats = CURCompressor(model, rank=16, method="leverage").compress()
        x = torch.randn(self.batch_size, 10, 64)
        out_m = c_model(x)
        self.assertEqual(out_m.shape, (self.batch_size, 10, 32))

    # 5. CUR MoE Tests
    def test_cur_moe_compressor(self) -> None:
        moe_model = ToyMoEModel(num_experts=3, in_features=64, out_features=64)
        c_model, stats = CURMoECompressor(moe_model, rank=16, method="leverage").compress()
        x = torch.randn(self.batch_size, 4, 64)
        out_m = c_model(x)
        self.assertEqual(out_m.shape, (self.batch_size, 4, 64))
        self.assertGreater(stats["compressed_layer_count"], 0)

    # 6. NMF Tests
    def test_nmf_linear_and_compressor(self) -> None:
        nmf_layer = NMFLinear.from_linear(self.base_layer, rank=16, max_iter=20)
        out = nmf_layer(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

        model = ToySequentialModel()
        c_model, stats = NMFCompressor(model, rank=16, max_iter=20).compress()
        x = torch.randn(self.batch_size, 64)
        out_m = c_model(x)
        self.assertEqual(out_m.shape, (self.batch_size, 32))

    # 7. Subspace Projection Tests
    def test_subspace_projection(self) -> None:
        sub_layer = SubspaceLinear.from_linear(self.base_layer, rank=16)
        out = sub_layer(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

        model = ToySequentialModel()
        c_model, stats = SubspaceProjectionCompressor(model, rank=16).compress()
        x = torch.randn(self.batch_size, 64)
        out_m = c_model(x)
        self.assertEqual(out_m.shape, (self.batch_size, 32))

    # 8. HashedNet Tests
    def test_hashed_net(self) -> None:
        h_layer = HashedLinear.from_linear(self.base_layer, n_buckets=256, use_sign=True)
        out = h_layer(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

        model = ToySequentialModel()
        c_model, stats = HashedNetCompressor(model, n_buckets=256).compress()
        x = torch.randn(self.batch_size, 64)
        out_m = c_model(x)
        self.assertEqual(out_m.shape, (self.batch_size, 32))

    # 9. COMPOT Tests
    def test_compot(self) -> None:
        comp_layer = COMPOTLinear.from_linear_and_activations(self.base_layer, rank=16)
        out = comp_layer(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

        model = ToySequentialModel()
        c_model, stats = COMPOTCompressor(model, rank=16).compress()
        x = torch.randn(self.batch_size, 64)
        out_m = c_model(x)
        self.assertEqual(out_m.shape, (self.batch_size, 32))

    # 10. CR (Interpolative Decomposition) Tests
    def test_cr_compressor(self) -> None:
        cr_layer = CRLinear.from_linear(self.base_layer, rank=16)
        out = cr_layer(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

        model = ToySequentialModel()
        c_model, stats = CRCompressor(model, rank=16).compress()
        x = torch.randn(self.batch_size, 64)
        out_m = c_model(x)
        self.assertEqual(out_m.shape, (self.batch_size, 32))

    # 11. FRanDI Tests
    def test_frandi(self) -> None:
        fr_layer = FRanDILinear.from_linear(self.base_layer, rank=16, energy_threshold=0.95)
        out = fr_layer(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

        model = ToySequentialModel()
        c_model, stats = FRanDICompressor(model, rank=16, energy_threshold=0.95).compress()
        x = torch.randn(self.batch_size, 64)
        out_m = c_model(x)
        self.assertEqual(out_m.shape, (self.batch_size, 32))

    # 12. GreCoDeC Tests
    def test_grecodec(self) -> None:
        gre_layer = GreCoDeCLinear.from_linear(self.base_layer, rank=16, max_iter=5)
        out = gre_layer(self.dummy_input)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_dim))

        model = ToySequentialModel()
        c_model, stats = GreCoDeCCompressor(model, rank=16, max_iter=5).compress()
        x = torch.randn(self.batch_size, 64)
        out_m = c_model(x)
        self.assertEqual(out_m.shape, (self.batch_size, 32))

    # 13. COMPRESSOR_REGISTRY Factory Invocations
    def test_registry_factories(self) -> None:
        keys = [
            "kronecker", "tt", "kronecker_tt_lora",
            "cur", "cur_moe", "nmf", "subspace_projection",
            "hashed_net", "compot", "cr", "frandi", "grecodec"
        ]
        class Config:
            pass

        cfg = Config()
        cfg.rank = 16

        for key in keys:
            factory = COMPRESSOR_REGISTRY.get(key)
            self.assertIsNotNone(factory, f"Factory for {key} should be registered in COMPRESSOR_REGISTRY")
            model = ToySequentialModel()
            compressed = factory(model, comp_config=cfg)
            self.assertIsInstance(compressed, nn.Module)
            x = torch.randn(2, 64)
            y = compressed(x)
            self.assertEqual(y.shape, (2, 32))


if __name__ == "__main__":
    unittest.main()
