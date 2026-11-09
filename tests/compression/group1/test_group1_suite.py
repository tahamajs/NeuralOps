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

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from neuralops.core.registry import COMPRESSOR_REGISTRY

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


class BenchmarkMLP(nn.Module):
    def __init__(self, in_features: int = 128, hidden: int = 256, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _get_compressor_kwargs(key: str) -> dict:
    if "quant" in key or key in ("astro_quantizer", "awq"):
        if key == "astro_quantizer":
            return {"bits": 4, "max_error": 1.0}
        return {"bits": 4}
    elif "prun" in key or "admm" in key:
        return {"sparsity": 0.4, "max_error": 1.0}
    elif "aatc" in key or "cachenotes" in key:
        return {"keep_ratio": 0.5, "max_error": 1.0, "rank": 16}
    else:
        return {"rank": 16, "max_error": 1.0}


@pytest.mark.parametrize("key", GROUP1_KEYS)
def test_group1_registration(key: str):
    """Verify that every compressor in Group 1 is properly registered in COMPRESSOR_REGISTRY."""
    assert key in COMPRESSOR_REGISTRY, f"Compressor {key} missing from COMPRESSOR_REGISTRY"
    cls = COMPRESSOR_REGISTRY.get(key)
    assert cls is not None, f"Compressor class for {key} is None"


@pytest.mark.parametrize("key", GROUP1_KEYS)
def test_group1_compression_and_forward_mlp(key: str):
    """Verify compression, forward pass, output shape, finiteness, and physical reduction on MLP."""
    cls = COMPRESSOR_REGISTRY.get(key)
    model = BenchmarkMLP()
    orig_params = sum(p.numel() for p in model.parameters())

    kwargs = _get_compressor_kwargs(key)
    compressor = cls(model, **kwargs)
    compressed, stats = compressor.compress()
    assert compressed is not None

    comp_params = sum(p.numel() for p in compressed.parameters())
    reduction_ratio = orig_params / max(1, comp_params)
    effective_ratio = float(stats.get("effective_ratio", stats.get("compression_ratio", reduction_ratio)))
    assert effective_ratio > 1.05, f"Effective ratio {effective_ratio} <= 1.05 for {key}"

    x = torch.randn(2, 128)
    out = compressed(x)
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all(), f"Non-finite output detected in {key} forward pass"


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

    cls = COMPRESSOR_REGISTRY.get(key)
    kwargs = _get_compressor_kwargs(key)
    compressor = cls(model, **kwargs)
    compressed, stats = compressor.compress()
    assert compressed is not None

    dummy_ids = torch.randint(0, 1000, (2, 16))
    dummy_mask = torch.ones((2, 16), dtype=torch.long)
    with torch.no_grad():
        out = compressed(input_ids=dummy_ids, attention_mask=dummy_mask)
    lhs = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)
    assert torch.isfinite(lhs).all(), f"BERT output contains NaNs/Infs for {key}"
