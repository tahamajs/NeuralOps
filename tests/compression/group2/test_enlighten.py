#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive test suite for ENLighten compressor.
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression.low_rank.enlighten_compressor import (
    ENLightenCompressor,
    enlighten_factory,
    unstructured_prune,
    vectorized_24_prune,
)
from neuralops.core.registry import COMPRESSOR_REGISTRY


class DummyLinearModel(nn.Module):
    def __init__(self, in_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim * 2)
        self.fc2 = nn.Linear(out_dim * 2, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.q_proj(x) + self.k_proj(x) + self.v_proj(x)
        h = self.out_proj(h)
        h = self.fc2(torch.relu(self.fc1(h)))
        return h


def test_registry():
    """Verify registration under both canonical names."""
    assert "enlighten" in COMPRESSOR_REGISTRY
    assert "enlighten_compressor" in COMPRESSOR_REGISTRY
    assert COMPRESSOR_REGISTRY.get("enlighten") is not None
    assert COMPRESSOR_REGISTRY.get("enlighten_compressor") is not None


def test_vectorized_24_pruning():
    """Verify exactly 2 out of every 4 entries are non-zero in each block."""
    W = torch.randn(16, 32)
    pruned = vectorized_24_prune(W)
    assert pruned.shape == W.shape
    W_blocks = pruned.view(-1, 4)
    nonzeros = (W_blocks != 0).sum(dim=-1)
    assert (nonzeros == 2).all()


def test_unstructured_prune():
    """Verify target fraction of weights with smallest magnitude are zeroed."""
    W = torch.randn(32, 32)
    pruned = unstructured_prune(W, sparsity=0.5)
    zeros = (pruned == 0).sum().item()
    assert abs(zeros - (32 * 32 * 0.5)) <= 2


def test_enlighten_compression_forward():
    """Verify compressed model forward pass produces valid tensors and finite error."""
    model = DummyLinearModel(64, 64)
    compressor = ENLightenCompressor(model, rank=16, structured_24=True, max_error=1.0)
    comp_model, stats = compressor.compress()

    x = torch.randn(2, 64)
    out = comp_model(x)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert len(stats["compressed_layers"]) > 0


def test_no_scale_distortion():
    """Verify that _decompose_and_prune does not multiply by an arbitrary Frobenius norm scale factor."""
    model = DummyLinearModel(64, 64)
    compressor = ENLightenCompressor(model, rank=16, structured_24=True)
    W = torch.randn(64, 64)
    U_factor, V_factor, err = compressor._decompose_and_prune(W, rank=16)
    W_recon = U_factor @ V_factor
    expected_err = float(torch.linalg.norm(W - W_recon) / torch.linalg.norm(W))
    assert abs(err - expected_err) < 1e-6


def test_target_attention_only():
    """Verify target_attention_only restricts compression to attention projections."""
    model = DummyLinearModel(64, 64)
    compressor = ENLightenCompressor(model, rank=16, target_attention_only=True, max_error=0.8)
    comp_model, stats = compressor.compress()

    compressed_names = stats["compressed_layers"]
    for name in compressed_names:
        assert any(k in name for k in ("q_proj", "k_proj", "v_proj", "out_proj", "attn", "attention"))
    assert not any("fc1" in name or "fc2" in name for name in compressed_names)


def test_fallback_on_unreachable_error():
    """Verify fallback to original module if max_error is impossibly low."""
    model = DummyLinearModel(64, 64)
    compressor = ENLightenCompressor(model, rank=2, max_error=0.0001, fallback_to_original=True)
    comp_model, stats = compressor.compress()

    assert len(stats["fallback_layers"]) > 0


def test_factory_function():
    """Verify factory invocation with config dictionary."""
    model = DummyLinearModel(64, 64)
    cfg = {"rank": 16, "structured_24": True, "max_error": 1.0}
    comp_model = enlighten_factory(model, comp_config=cfg)
    assert comp_model is not None
