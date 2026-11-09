#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration test suite verifying all 17 Group 2 Compressors (Keys 18-34).
"""

import pytest
import torch
import torch.nn as nn
from neuralops.core.registry import COMPRESSOR_REGISTRY

GROUP2_KEYS = [
    "coopq_quantizer",
    "cr_compressor",
    "cur_compressor",
    "cur_moe_compressor",
    "delta_kv",
    "direct_svd",
    "dobi_svd",
    "drone_compressor",
    "dynamic_rank",
    "enlighten_compressor",
    "entropy_budget_allocation",
    "erc_svd",
    "faq_quantizer",
    "flap",
    "flat_llm",
    "fp8",
    "frandi_compressor",
]


class BenchmarkMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 256, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


@pytest.mark.parametrize("key", GROUP2_KEYS)
def test_group2_registration(key: str):
    """Verify that every compressor in Group 2 is properly registered in COMPRESSOR_REGISTRY."""
    assert key in COMPRESSOR_REGISTRY, f"Compressor {key} missing from COMPRESSOR_REGISTRY"
    cls = COMPRESSOR_REGISTRY.get(key)
    assert cls is not None, f"Compressor class for {key} is None"


@pytest.mark.parametrize("key", GROUP2_KEYS)
def test_group2_compression_and_forward(key: str):
    """Verify compression, forward pass, output shape, finiteness, and physical reduction."""
    cls = COMPRESSOR_REGISTRY.get(key)
    model = BenchmarkMLP()
    orig_params = sum(p.numel() for p in model.parameters())

    kwargs = {}
    if "quant" in key or key in ("fp8", "coopq_quantizer", "faq_quantizer"):
        if key == "fp8":
            kwargs = {"fp8_format": "e4m3"}
        elif key == "faq_quantizer":
            kwargs = {"bits": 3}
        else:
            kwargs = {"bits": 4}
    elif key in ("flap", "flat_llm"):
        if key == "flap":
            kwargs = {"sparsity": 0.2, "max_error": 1.0}
        else:
            kwargs = {"prune_ratio": 0.3, "max_error": 1.0}
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
    bit_reduction = stats.get("bit_reduction", 1.0)
    sparsity = stats.get("sparsity", 0.0)
    sparsity_ratio = 1.0 / max(1e-6, 1.0 - sparsity)
    eff_ratio = max(param_ratio, bit_reduction, sparsity_ratio)
    assert eff_ratio > 1.05, f"Compressor {key} failed reduction: effective_ratio={eff_ratio:.2f}x <= 1.05x"
