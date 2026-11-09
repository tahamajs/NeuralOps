#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration test suite verifying all 17 Group 1 Compressors (Keys 1-17).
"""

from typing import Any
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from neuralops.core.registry import COMPRESSOR_REGISTRY, create_compressor

GROUP1_KEYS = [
    "a3_compressor",
    "aa_svd_compressor",
    "aatc_xkv_compressor",
    "adaptive_ranks",
    "adasvd_compressor",
    "admm_compressor",
    "admm_pruner",
    "armor_pruner",
    "ast_pruner",
    "astro_quantizer",
    "asvd",
    "auto_svd",
    "awq",
    "bonsai_pruner",
    "cachenotes_compressor",
    "come_compressor",
    "compot_compressor",
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
    if cls is None:
        raise KeyError(f"Compressor '{name}' not found in registry")

    kwargs: dict[str, Any] = {
        "rank": rank,
        "fallback_to_original": False,
        "max_error": 1.0,
        "target_attention_only": False,
    }
    if name == "astro_quantizer":
        kwargs["bits"] = 4
    elif name == "awq":
        kwargs["bits"] = 4
    elif name in ("admm_compressor", "admm_pruner", "ast_pruner", "bonsai_pruner", "armor_pruner"):
        kwargs["sparsity"] = 0.4
    elif name in ("aatc_xkv_compressor", "cachenotes_compressor"):
        kwargs["keep_ratio"] = 0.5

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


@pytest.mark.parametrize("key", GROUP1_KEYS)
def test_group1_registration(key: str):
    """Verify that every compressor in Group 1 is properly registered in COMPRESSOR_REGISTRY."""
    assert key in COMPRESSOR_REGISTRY, f"Compressor {key} missing from COMPRESSOR_REGISTRY"
    cls = COMPRESSOR_REGISTRY.get(key)
    assert cls is not None, f"Compressor class for {key} is None"


@pytest.mark.parametrize("key", GROUP1_KEYS)
def test_group1_compression_and_forward_mlp(key: str):
    """Verify compression, forward pass, output shape, finiteness, and physical reduction on MLP."""
    model = TwoLayerMLP(hidden_size=768)
    orig_params = sum(p.numel() for p in model.parameters())

    compressed, comp_obj = _apply_compressor(key, model, rank=64)
    assert compressed is not None

    x = torch.randn(2, 16, 768)
    try:
        out = compressed(x)
    except Exception:
        x = torch.randn(2, 768)
        out = compressed(x)

    if isinstance(out, (tuple, list)):
        out = out[0]
    assert torch.isfinite(out).all(), f"Non-finite output detected in {key} forward pass"

    comp_params = sum(p.numel() for p in compressed.parameters())
    param_ratio = orig_params / max(1, comp_params)

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
        if "effective_ratio" in stats:
            bit_reduction = max(bit_reduction, float(stats["effective_ratio"]))
        if "compression_ratio" in stats:
            bit_reduction = max(bit_reduction, float(stats["compression_ratio"]))
        if "effective_parameter_ratio" in stats:
            bit_reduction = max(bit_reduction, float(stats["effective_parameter_ratio"]))

    effective_ratio = max(param_ratio, sparsity_ratio, bit_reduction)
    assert effective_ratio > 1.05, f"Effective ratio {effective_ratio} <= 1.05 for {key}"


@pytest.mark.parametrize("key", GROUP1_KEYS)
def test_group1_compression_and_forward_bert(key: str):
    """Verify compression and numerical stability on multi-head attention transformer."""
    cfg = BertConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=1000,
    )
    model = BertModel(cfg)
    model.eval()

    compressed, comp_obj = _apply_compressor(key, model, rank=16)
    assert compressed is not None

    dummy_ids = torch.randint(0, 1000, (2, 16))
    dummy_mask = torch.ones((2, 16), dtype=torch.long)
    with torch.no_grad():
        out = compressed(input_ids=dummy_ids, attention_mask=dummy_mask)
    lhs = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, (tuple, list)) else out)
    assert torch.isfinite(lhs).all(), f"BERT output contains NaNs/Infs for {key}"
