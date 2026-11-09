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

from typing import Any
import pytest
import torch
import torch.nn as nn
from neuralops.core.registry import COMPRESSOR_REGISTRY, create_compressor

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


class TwoLayerMLP(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _apply_compressor(name: str, model: nn.Module, rank: int = 32) -> tuple[nn.Module, dict[str, Any]]:
    cls = COMPRESSOR_REGISTRY.get(name)
    kwargs = {
        "rank": rank,
        "fallback_to_original": False,
        "max_error": 1.0,
        "target_attention_only": False,
    }
    if name == "fp8":
        kwargs["fp8_format"] = "e4m3"
    elif name == "faq_quantizer":
        kwargs["bits"] = 3
    elif name == "coopq_quantizer":
        kwargs["bits"] = 4
    elif name == "flap":
        kwargs["sparsity"] = 0.2
    elif name == "flat_llm":
        kwargs["prune_ratio"] = 0.3

    try:
        comp_obj = create_compressor(name, model=model, **kwargs)
    except Exception:
        try:
            comp_obj = cls(model=model, **kwargs)
        except Exception:
            comp_obj = cls(model, **kwargs)

    res = comp_obj.compress()
    if isinstance(res, tuple):
        compressed = res[0]
        stats = res[1] if len(res) > 1 else {}
    else:
        compressed = res
        stats = getattr(comp_obj, "stats", {})
    return compressed, stats


@pytest.mark.parametrize("key", GROUP2_KEYS)
def test_group2_registration(key: str):
    """Verify that every compressor in Group 2 is properly registered in COMPRESSOR_REGISTRY."""
    assert key in COMPRESSOR_REGISTRY, f"Compressor {key} missing from COMPRESSOR_REGISTRY"
    cls = COMPRESSOR_REGISTRY.get(key)
    assert cls is not None, f"Compressor class for {key} is None"


@pytest.mark.parametrize("key", GROUP2_KEYS)
def test_group2_compression_and_forward(key: str):
    """Verify compression, forward pass, output shape, finiteness, and physical reduction."""
    model = TwoLayerMLP(hidden_size=256)
    orig_params = sum(p.numel() for p in model.parameters())

    compressed, stats = _apply_compressor(key, model, rank=32)

    # Forward pass check
    x = torch.randn(2, 256)
    out = compressed(x)
    assert out.shape == (2, 256), f"Wrong output shape for {key}: {out.shape}"
    assert torch.isfinite(out).all(), f"Non-finite outputs encountered in {key}"

    # Physical reduction check
    comp_params = sum(p.numel() for p in compressed.parameters())
    param_ratio = orig_params / max(comp_params, 1)
    bit_reduction = stats.get("bit_reduction", 1.0)
    sparsity = stats.get("sparsity", 0.0)
    sparsity_ratio = 1.0 / max(1e-6, 1.0 - sparsity)
    eff_ratio = max(param_ratio, bit_reduction, sparsity_ratio)
    assert eff_ratio > 1.05, f"Compressor {key} failed reduction: effective_ratio={eff_ratio:.2f}x <= 1.05x"
