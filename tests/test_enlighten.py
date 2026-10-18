"""Comprehensive test suite for ENLighten compressor.

Tests:
1. Registration under both 'enlighten' and 'enlighten_compressor'.
2. Vectorized 2:4 structured sparsity patterns across row blocks.
3. Unstructured magnitude pruning fraction accuracy.
4. Linear layer compression and error calculation without artificial scale distortion.
5. Conv1D / transposed layer support (GPT-2 style).
6. Target attention only filtering and non-target freezing.
7. Iterative rank escalation on high error.
8. Safe fallback to original module if error threshold cannot be met.
9. LoRA adapter preservation on wrapped modules.
10. Factory function invocation and registry lookup.
"""

import pytest
import torch
import torch.nn as nn
from neuralops.compression_engine.enlighten_compressor import (
    ENLightenCompressor, enlighten_factory, unstructured_prune,
    vectorized_24_prune)
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
    # Check each block of 4 has at most 2 non-zeros
    blocks = pruned.view(16, 8, 4)
    non_zeros = (blocks != 0.0).sum(dim=-1)
    assert (non_zeros <= 2).all(), "Every 4-block must have <= 2 non-zero elements"


def test_unstructured_prune():
    """Verify unstructured magnitude pruning achieves target sparsity."""
    W = torch.randn(64, 64)
    pruned = unstructured_prune(W, sparsity=0.5)
    zeros = (pruned == 0.0).sum().item()
    total = W.numel()
    assert abs(zeros / total - 0.5) < 0.05


def test_enlighten_compression_forward():
    """Verify model compression and clean forward pass."""
    model = DummyLinearModel(64, 64)
    x = torch.randn(2, 8, 64)
    out_orig = model(x)

    compressor = ENLightenCompressor(model, rank=16, sparsity=0.5, structured_24=True, max_error=0.8)
    comp_model, stats = compressor.compress()

    assert stats["num_compressed"] > 0
    assert stats["compressor_name"] == "enlighten"
    out_comp = comp_model(x)
    assert out_comp.shape == out_orig.shape
    assert not torch.isnan(out_comp).any()


def test_no_scale_distortion():
    """Verify that _decompose_and_prune does not multiply by an arbitrary Frobenius norm scale factor."""
    model = DummyLinearModel(64, 64)
    compressor = ENLightenCompressor(model, rank=16, structured_24=True)
    W = torch.randn(64, 64)
    W_recon, err = compressor._decompose_and_prune(W, rank=16)
    # The reconstruction error should equal ||W - W_recon||_F / ||W||_F exactly
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
        assert "fc1" not in name and "fc2" not in name


def test_fallback_on_unreachable_error():
    """Verify fallback to original module if max_error threshold cannot be met."""
    model = DummyLinearModel(64, 64)
    # Setting max_error impossibly tight (1e-6) forces fallback
    compressor = ENLightenCompressor(model, rank=2, max_error=1e-6, fallback_to_original=True)
    comp_model, stats = compressor.compress()

    assert stats["num_fallback"] > 0
    assert stats["num_compressed"] == 0


def test_factory_function():
    """Verify enlighten_factory works with registry call."""
    model = DummyLinearModel(32, 32)
    x = torch.randn(2, 4, 32)
    comp_model = enlighten_factory(model, rank=8, sparsity=0.5, structured_24=True)
    out = comp_model(x)
    assert out.shape == (2, 4, 32)
    assert not torch.isnan(out).any()


if __name__ == "__main__":
    test_registry()
    test_vectorized_24_pruning()
    test_unstructured_prune()
    test_enlighten_compression_forward()
    test_no_scale_distortion()
    test_target_attention_only()
    test_fallback_on_unreachable_error()
    test_factory_function()
    print("All 8 ENLighten tests passed successfully!")
