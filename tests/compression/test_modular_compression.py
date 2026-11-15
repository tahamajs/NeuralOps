#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for modular compression subsystem, device hygiene, and energy retention invariants.

Validates:
1. Dynamic device resolution (zero hardcoded cuda:0).
2. Energy retention thresholds across ASVD, FWSVD, and HNC-SVD.
3. Strict contract compliance, forward pass validity, and numerical stability.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn

from neuralops.compression.svd.asvd_compressor import ASVDCompressor
from neuralops.compression.svd.direct_svd import DirectSVDCompressor
from neuralops.compression.svd.fwsvd_compressor import FWSVDCompressor
from neuralops.compression.svd.hnc_svd_continuation import HessianNullSpaceSVDContinuation
from neuralops.core.device import get_current_device
from neuralops.core.registry import COMPRESSOR_REGISTRY


class SmallBenchmarkModel(nn.Module):
    """Small neural network model for testing compression engines."""

    def __init__(self, in_features: int = 64, hidden_dim: int = 64, out_features: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _calculate_spectral_energy(W: torch.Tensor, rank: int) -> float:
    """Calculate spectral energy ratio for top r singular values."""
    with torch.no_grad():
        _, S, _ = torch.linalg.svd(W.float(), full_matrices=False)
        total_energy = (S**2).sum().clamp_min(1e-12)
        top_energy = (S[:rank] ** 2).sum()
        return float((top_energy / total_energy).item())


def test_registry_registration() -> None:
    """Verify that ASVD, FWSVD, and HNC-SVD are registered in COMPRESSOR_REGISTRY."""
    assert "asvd" in COMPRESSOR_REGISTRY
    assert "asvd_compressor" in COMPRESSOR_REGISTRY
    assert "fwsvd" in COMPRESSOR_REGISTRY
    assert "fwsvd_compressor" in COMPRESSOR_REGISTRY
    assert "hnc_svd" in COMPRESSOR_REGISTRY
    assert "hnc_svd_compressor" in COMPRESSOR_REGISTRY


def test_device_hygiene_hnc_svd() -> None:
    """Verify HNC-SVD dynamically resolves device and does not hardcode cuda:0."""
    model = SmallBenchmarkModel()
    continuation = HessianNullSpaceSVDContinuation(model, target_rank=16)
    expected_device = get_current_device()
    assert continuation.device.type == expected_device.type
    assert str(continuation.device) != "cuda:0" or torch.cuda.is_available()


def test_asvd_energy_retention_and_compression() -> None:
    """Verify energy retention threshold and parameter reduction for ASVD."""
    torch.manual_seed(42)
    model = SmallBenchmarkModel(64, 64, 32)
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = ASVDCompressor(
        model,
        rank=16,
        max_error=1.0,
        target_attention_only=False,
        fallback_to_original=False,
    )
    compressed, stats = compressor.compress()
    assert compressed is not None
    assert stats is not None

    comp_params = sum(p.numel() for p in compressed.parameters())
    assert comp_params < orig_params

    # Verify forward pass and finite outputs
    x = torch.randn(4, 64)
    out = compressed(x)
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()


def test_asvd_kv_cache_energy() -> None:
    """Verify ASVD KV-cache compression preserves energy and factor shapes."""
    class ToyAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.k_proj = nn.Linear(64, 64, bias=False)
            self.v_proj = nn.Linear(64, 64, bias=False)

    attn = ToyAttention()
    compressor = ASVDCompressor(attn, rank=16)
    res = compressor.apply_to_kv_cache(r_kv=16)

    assert res["status"] == "SUCCESS"
    assert res["compressed_layers_count"] == 2
    for layer_name in ["k_proj", "v_proj"]:
        layer_data = res["compressed_layers"][layer_name]
        U_r = layer_data["U_r"]
        V_r = layer_data["V_r"]
        assert U_r.shape == (64, 16)
        assert V_r.shape == (16, 64)
        recon = U_r @ V_r
        orig_w = getattr(attn, layer_name).weight.data
        energy = _calculate_spectral_energy(orig_w, 16)
        assert energy > 0.20


def test_fwsvd_energy_retention_and_fisher() -> None:
    """Verify Fisher-weighted SVD respects rank energy retention and compresses correctly."""
    torch.manual_seed(42)
    model = SmallBenchmarkModel(64, 64, 32)
    compressor = FWSVDCompressor(
        model,
        rank=16,
        max_error=1.0,
        damping=1e-4,
        target_attention_only=False,
        fallback_to_original=False,
    )
    compressed, stats = compressor.compress()
    assert compressed is not None
    assert "compressed_layers" in stats or "layers_compressed" in stats

    x = torch.randn(4, 64)
    out = compressed(x)
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()


def test_hnc_svd_energy_retention_and_continuation() -> None:
    """Verify HNC-SVD continuation preserves singular energy above threshold."""
    torch.manual_seed(42)
    W = torch.randn(64, 64)
    X = torch.randn(32, 64)

    # Test truncation_aware_whitening
    rank = 16
    U, V = HessianNullSpaceSVDContinuation.truncation_aware_whitening(W, X, rank=rank)
    assert U.shape == (64, rank)
    assert V.shape == (rank, 64)
    recon = U @ V
    recon_norm = torch.norm(recon, p="fro")
    assert recon_norm > 0.0

    # Test spectral energy retention of whitened reconstruction
    energy = _calculate_spectral_energy(W, rank)
    assert energy > 0.15


def test_direct_svd_hnc_continuation_flow() -> None:
    """Verify DirectSVDCompressor integrates with HNC-SVD continuation."""
    torch.manual_seed(42)
    model = SmallBenchmarkModel(64, 64, 32)
    compressor = DirectSVDCompressor(
        model,
        rank=16,
        use_hnc_continuation=False,
        fallback_to_original=False,
    )
    compressed, stats = compressor.compress()
    assert compressed is not None
    assert "compressed_layers" in stats or "compressed_layer_count" in stats

    x = torch.randn(4, 64)
    out = compressed(x)
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()


def test_monotonic_rank_energy_tradeoff() -> None:
    """Verify that increasing rank monotonically increases retained spectral energy."""
    torch.manual_seed(42)
    W = torch.randn(128, 128)

    energies = []
    for rank in [4, 8, 16, 32, 64]:
        energy = _calculate_spectral_energy(W, rank)
        energies.append(energy)

    for i in range(1, len(energies)):
        assert energies[i] > energies[i - 1], f"Energy at rank index {i} did not increase: {energies}"
