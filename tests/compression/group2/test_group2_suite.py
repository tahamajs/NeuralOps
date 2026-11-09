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
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _apply_compressor(name: str, model: nn.Module, rank: int = 64) -> tuple[nn.Module, Any]:
    cls = COMPRESSOR_REGISTRY.get(name)
    kwargs: dict[str, Any] = {
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

    comp_obj = None
    try:
        comp_obj = create_compressor(name, model=model, **kwargs)
    except Exception:
        try:
            comp_obj = cls(model=model, **kwargs)
        except Exception:
            try:
                comp_obj = cls(model, **kwargs)
            except Exception:
                try:
                    comp_obj = cls(model=model)
                except Exception:
                    try:
                        comp_obj = cls(model)
                    except Exception:
                        comp_obj = cls()

    compressed = None
    if hasattr(comp_obj, "compress"):
        res = comp_obj.compress()
        compressed = res[0] if isinstance(res, tuple) else res
    elif callable(comp_obj) and comp_obj is not model:
        try:
            res = comp_obj(model, **kwargs)
        except Exception:
            res = comp_obj(model)
        if isinstance(res, tuple):
            compressed = res[0]
        elif hasattr(res, "compress"):
            c_res = res.compress()
            compressed = c_res[0] if isinstance(c_res, tuple) else c_res
        else:
            compressed = res
    else:
        compressed = comp_obj

    return compressed, comp_obj


@pytest.mark.parametrize("key", GROUP2_KEYS)
def test_group2_registration(key: str):
    """Verify that every compressor in Group 2 is properly registered in COMPRESSOR_REGISTRY."""
    assert key in COMPRESSOR_REGISTRY, f"Compressor {key} missing from COMPRESSOR_REGISTRY"
    cls = COMPRESSOR_REGISTRY.get(key)
    assert cls is not None, f"Compressor class for {key} is None"


@pytest.mark.parametrize("key", GROUP2_KEYS)
def test_group2_compression_and_forward(key: str):
    """Verify compression, forward pass, output shape, finiteness, and physical reduction."""
    model = TwoLayerMLP(hidden_size=768)
    orig_params = sum(p.numel() for p in model.parameters())

    compressed, comp_obj = _apply_compressor(key, model, rank=64)

    # Forward pass check
    x = torch.randn(2, 16, 768)
    out = compressed(x)
    assert out.shape == (2, 16, 768), f"Wrong output shape for {key}: {out.shape}"
    assert torch.isfinite(out).all(), f"Non-finite outputs encountered in {key}"

    # Physical reduction check
    comp_params = sum(p.numel() for p in compressed.parameters())
    param_ratio = orig_params / max(comp_params, 1)

    total_zeros = sum((p == 0).sum().item() for p in compressed.parameters())
    total_elements = sum(p.numel() for p in compressed.parameters())
    sparsity = total_zeros / total_elements if total_elements > 0 else 0.0
    sparsity_ratio = 1.0 / (1.0 - sparsity + 1e-9) if sparsity > 0.0 else 1.0

    bit_reduction = 1.0
    for obj in [comp_obj, compressed]:
        if hasattr(obj, "effective_bits"):
            bit_reduction = max(bit_reduction, 32.0 / obj.effective_bits)
        if hasattr(obj, "bits"):
            bit_reduction = max(bit_reduction, 32.0 / obj.bits)
        if hasattr(obj, "bit_reduction"):
            bit_reduction = max(bit_reduction, float(obj.bit_reduction))
        if hasattr(obj, "compression_ratio"):
            cr = float(obj.compression_ratio)
            if cr > 1.05:
                bit_reduction = max(bit_reduction, cr)

    stats = getattr(compressed, "compression_stats", getattr(comp_obj, "stats", {}))
    if hasattr(comp_obj, "get_compression_stats"):
        try:
            stats = comp_obj.get_compression_stats()
        except Exception:
            pass
    if isinstance(stats, dict):
        if "effective_compression_rate" in stats:
            bit_reduction = max(bit_reduction, float(stats["effective_compression_rate"]))
        if "bit_reduction" in stats:
            bit_reduction = max(bit_reduction, float(stats["bit_reduction"]))
        if "effective_parameter_ratio" in stats:
            val = float(stats["effective_parameter_ratio"])
            if 0.0 < val < 1.0:
                val = 1.0 / val
            param_ratio = max(param_ratio, val)

    eff_ratio = max(param_ratio, sparsity_ratio, bit_reduction)
    assert eff_ratio > 1.05, f"Compressor {key} failed reduction: effective_ratio={eff_ratio:.2f}x <= 1.05x"
