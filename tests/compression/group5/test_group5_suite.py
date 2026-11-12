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

    # Forward pass check with gradient tracking
    x = torch.randn(2, 128, requires_grad=True)
    out = compressed(x)
    assert out.shape == (2, 64), f"Wrong output shape for {key}: {out.shape}"
    assert torch.isfinite(out).all(), f"Non-finite outputs encountered in {key}"

    # Autograd backward gate: verify gradient propagation
    loss = out.sum()
    loss.backward()
    trainable_params = [p for p in compressed.parameters() if p.requires_grad]
    if trainable_params:
        grads = [p.grad for p in trainable_params if p.grad is not None]
        assert len(grads) > 0, f"No parameter gradients computed during backward pass for {key}"
        assert all(torch.isfinite(g).all() for g in grads), f"Non-finite gradient detected in {key} parameters"
    assert x.grad is not None, f"Input gradient missing during backward pass for {key}"
    assert torch.isfinite(x.grad).all(), f"Non-finite input gradient detected in {key}"

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


GROUP5_RANK_KEYS = [
    "soka_compressor",
    "subspace_projection",
    "svd_linear",
    "svd_llm_compressor",
    "swift_svd",
    "swift_svd_act",
    "tomoe_compressor",
    "tt_compressor",
]


@pytest.mark.parametrize("key", GROUP5_RANK_KEYS)
def test_group5_monotonic_pareto(key: str):
    """Verify that decreasing ranks [32, 16, 8, 4] yield strictly decreasing parameter counts and increasing reduction ratios."""
    ranks = [32, 16, 8, 4]
    prev_params = float("inf")
    prev_ratio = 0.0
    cls = COMPRESSOR_REGISTRY.get(key)

    for rank in ranks:
        model = BenchmarkMLP()
        orig_params = sum(p.numel() for p in model.parameters())

        kwargs = {"rank": rank, "max_error": 1.0}
        if key == "tomoe_compressor":
            kwargs["num_experts"] = 4
            kwargs["top_k"] = 2

        compressor = cls(model, **kwargs)
        compressed, stats = compressor.compress()
        assert compressed is not None, f"Compressor {key} returned None at rank {rank}"

        curr_params = sum(p.numel() for p in compressed.parameters())
        curr_ratio = orig_params / max(1, curr_params)

        assert curr_params < prev_params, (
            f"Pareto rank monotonicity violated for {key} at rank {rank}: "
            f"current params ({curr_params}) >= previous params ({prev_params})"
        )
        assert curr_ratio > prev_ratio, (
            f"Pareto reduction ratio monotonicity violated for {key} at rank {rank}: "
            f"current ratio ({curr_ratio:.3f}) <= previous ratio ({prev_ratio:.3f})"
        )
        assert curr_ratio > 1.05, (
            f"Compressor {key} at rank {rank} failed physical reduction contract: {curr_ratio:.3f} <= 1.05"
        )

        # Autograd backward gate across each Pareto rank
        x = torch.randn(2, 128, requires_grad=True)
        out = compressed(x)
        assert out.shape == (2, 64), f"Wrong output shape at rank {rank} for {key}"
        assert torch.isfinite(out).all(), f"Non-finite output at rank {rank} for {key}"
        loss = out.sum()
        loss.backward()
        trainable = [p for p in compressed.parameters() if p.requires_grad]
        if trainable:
            grads = [p.grad for p in trainable if p.grad is not None]
            assert len(grads) > 0, f"No gradients at rank {rank} for {key}"
            assert all(torch.isfinite(g).all() for g in grads), f"Non-finite gradients at rank {rank} for {key}"
        assert x.grad is not None and torch.isfinite(x.grad).all(), f"Invalid input grad at rank {rank} for {key}"

        prev_params = curr_params
        prev_ratio = curr_ratio

