#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration test suite verifying all 15 Group 5 Compressors (Keys 68-82).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.core.registry import COMPRESSOR_REGISTRY

GROUP5_KEYS = [
    "slicegpt_pca",
    "slim_compressor",
    "smoothquant",
    "soka_compressor",
    "sparsegpt",
    "subspace_projection",
    "svd_linear",
    "svd_llm_compressor",
    "swift_svd",
    "swift_svd_act",
    "tomoe_compressor",
    "tt_compressor",
    "uniql_compressor",
    "wanda",
    "zerokv_neo_compressor",
]


class BenchmarkMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 256, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


@pytest.mark.parametrize("key", GROUP5_KEYS)
def test_group5_registration(key: str):
    """Verify that every compressor in Group 5 is properly registered in COMPRESSOR_REGISTRY."""
    assert key in COMPRESSOR_REGISTRY, f"Compressor {key} missing from COMPRESSOR_REGISTRY"
    cls = COMPRESSOR_REGISTRY.get(key)
    assert cls is not None, f"Compressor class for {key} is None"


@pytest.mark.parametrize("key", GROUP5_KEYS)
def test_group5_compression_and_forward(key: str):
    """Verify compression, forward pass, output shape, finiteness, and physical reduction."""
    cls = COMPRESSOR_REGISTRY.get(key)
    model = BenchmarkMLP()
    orig_params = sum(p.numel() for p in model.parameters())

    kwargs = {}
    if key == "smoothquant":
        kwargs = {"alpha": 0.5}
    elif key in ("wanda", "sparsegpt"):
        kwargs = {"sparsity": 0.5}
    elif key == "slicegpt_pca":
        kwargs = {"sparsity": 0.5, "pca_ratio": 0.5}
    elif key == "slim_compressor":
        kwargs = {"bits": 4, "sparsity": 0.5, "rank": 8}
    elif key == "zerokv_neo_compressor":
        kwargs = {"keep_ratio": 0.3, "max_error": 1.0}
    elif key == "tomoe_compressor":
        kwargs = {"num_experts": 4, "top_k": 2, "rank": 8, "max_error": 1.0}
    elif key == "uniql_compressor":
        kwargs = {"rank": 16, "bits": 8, "max_error": 1.0}
    elif key == "swift_svd_act":
        calib = [torch.randn(4, 128) for _ in range(2)]
        kwargs = {"rank": 16, "calibration_loader": calib, "max_error": 1.0}
    else:
        kwargs = {"rank": 16, "max_error": 1.0}

    compressor = cls(model, **kwargs)
    compressed, stats = compressor.compress()

    # Forward pass check
    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64), f"Wrong output shape for {key}: {out.shape}"
    assert torch.isfinite(out).all(), f"Non-finite outputs encountered in {key}"

    # Physical reduction check
    comp_params = sum(p.numel() for p in compressed.parameters())
    param_ratio = orig_params / max(comp_params, 1)

    total_zeros = sum((p == 0).sum().item() for p in compressed.parameters())
    total_elements = sum(p.numel() for p in compressed.parameters())
    model_sparsity = float(total_zeros / max(1, total_elements))
    stats_sparsity = stats.get("sparsity", 0.0)
    sparsity = max(model_sparsity, stats_sparsity)
    sparsity_ratio = 1.0 / (1.0 - sparsity + 1e-9) if sparsity > 0.0 else 1.0

    bit_reduction = stats.get("bit_reduction", 1.0)
    if bit_reduction == 1.0:
        if hasattr(compressor, "effective_bits") and getattr(compressor, "effective_bits", 32) < 32:
            bit_reduction = 32.0 / compressor.effective_bits
        elif hasattr(compressor, "bits") and getattr(compressor, "bits", 32) < 32:
            bit_reduction = 32.0 / compressor.bits
        elif hasattr(compressed, "effective_bits") and getattr(compressed, "effective_bits", 32) < 32:
            bit_reduction = 32.0 / compressed.effective_bits

    stats_eff = stats.get("effective_ratio", 1.0)
    eff_ratio = max(param_ratio, bit_reduction, sparsity_ratio, stats_eff)
    assert eff_ratio > 1.05, f"Compressor {key} failed reduction: effective_ratio={eff_ratio:.2f}x <= 1.05x"
