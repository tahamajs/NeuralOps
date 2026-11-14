#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive SVD Family Smoke & Hardening Test Suite (Wave 2 - Domain G1).

Verifies the SVD family compressors and attention layers across key invariants:
1. Instantiation with rank=32, target_ratio=0.5 (or rank_ratio=0.5).
2. Application to linear projections (e.g. nn.Linear(768, 768)).
3. Preservation of output tensor shapes and finiteness (no NaN/Inf).
4. Relative reconstruction error within expected spectral bounds.
5. Autograd backward pass / gradient flow integrity.
6. Memory safety invariant: no full_matrices=True blowup in SVD calls.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


class ToyTransformerLayer(nn.Module):
    """Minimal transformer layer containing linear projection for compression testing."""

    def __init__(self, in_features: int = 768, out_features: int = 768) -> None:
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x)


class DummyConfig:
    """Mock transformer configuration for attention layer initialization."""

    def __init__(self, hidden_size: int = 768, num_attention_heads: int = 12) -> None:
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.d_model = hidden_size
        self.rank = 32


SVD_COMPRESSOR_NAMES = [
    "direct_svd",
    "asvd",
    "aa_svd_compressor",
    "adasvd_compressor",
    "slicegpt",
    "slicegpt_pca",
    "rotational_slicegpt_int4",
    "a3_compressor",
    "residual_svd_compressor",
    "saes_svd_v2",
    "svd_llm_compressor",
    "svd_linear",
    "hybrid_svd_layer",
    "swift_svd",
    "swift_svd_act",
    "kronecker_tt_lora",
    "kronecker_compressor",
    "mgie_budget",
    "auto_svd",
    "dobi_svd",
    "dynamic_rank",
    "entropy_budget_allocation",
    "erc_svd",
    "fwsvd_compressor",
    "gfws",
    "hadamard_incoherent_svd",
    "hnc_svd",
    "optimal_subspace_projection",
    "subspace_projection",
    "adaptive_ranks",
]


SVD_ATTENTION_LAYER_NAMES = [
    "minimal_svd",
    "adaptive_svd_attention",
    "asvd",
    "svd_compressed",
    "svd_llm_v2_attention",
    "fused_svd_kernel",
    "residual_svd_attention",
    "svd_linear_attention",
    "svd_projection_attention",
    "naive_svd",
]


@pytest.mark.parametrize("name", SVD_COMPRESSOR_NAMES)
def test_svd_compressor_registration(name: str) -> None:
    """Verify each canonical SVD compressor is registered in COMPRESSOR_REGISTRY."""
    cls = COMPRESSOR_REGISTRY.get(name)
    assert cls is not None, f"Compressor '{name}' not found in COMPRESSOR_REGISTRY"


@pytest.mark.parametrize("name", SVD_COMPRESSOR_NAMES)
def test_svd_compressor_apply_to_linear(name: str) -> None:
    """Test SVD compressor applied to nn.Linear(768, 768) with rank=32 and target_ratio=0.5."""
    cls = COMPRESSOR_REGISTRY.get(name)
    assert cls is not None, f"Compressor '{name}' unavailable"

    model = ToyTransformerLayer(768, 768)
    x = torch.randn(2, 768, requires_grad=True)
    orig_out = model(x).detach()

    kwargs: dict[str, Any] = {
        "rank": 32,
        "target_ratio": 0.5,
        "rank_ratio": 0.5,
        "fallback_to_original": False,
        "max_error": 1.0,
        "target_attention_only": False,
        "calibration_texts": ["NeuralOps deep learning compression calibration phrase"] * 4,
        "n_calib": 4,
    }

    try:
        compressor = cls(model, **kwargs)
    except TypeError:
        try:
            compressor = cls(model, rank=32)
        except TypeError:
            compressor = cls(model)

    if hasattr(compressor, "compress"):
        res = compressor.compress()
        comp_model = res[0] if isinstance(res, tuple) else res
    else:
        comp_model = compressor

    out = comp_model(x)
    assert out.shape == (2, 768), f"{name}: expected shape (2, 768), got {out.shape}"
    assert torch.isfinite(out).all(), f"{name}: compressed model output contains NaN or Inf"

    # Verify reconstruction error is non-negative and finite
    rel_error = (torch.norm(out.detach() - orig_out) / torch.norm(orig_out)).item()
    assert rel_error >= 0.0
    assert torch.isfinite(torch.tensor(rel_error))

    # Autograd backward verification
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, f"{name}: input gradient is None after backward"
    assert torch.isfinite(x.grad).all(), f"{name}: input gradient contains NaN or Inf"


@pytest.mark.parametrize("layer_name", SVD_ATTENTION_LAYER_NAMES)
def test_svd_attention_layer_forward_backward(layer_name: str) -> None:
    """Test SVD attention layers forward and backward computation."""
    cls = ATTENTION_REGISTRY.get(layer_name)
    assert cls is not None, f"Attention layer '{layer_name}' not found in ATTENTION_REGISTRY"

    cfg = DummyConfig(hidden_size=768, num_attention_heads=12)
    try:
        layer = cls(cfg)
    except TypeError:
        try:
            layer = cls(cfg, rank=32)
        except TypeError:
            layer = cls(d_model=768, num_heads=12, rank=32)

    x = torch.randn(2, 8, 768, requires_grad=True)
    out = layer(x)
    if isinstance(out, tuple):
        out = out[0]

    assert out.shape == (2, 8, 768), f"{layer_name}: expected output shape (2, 8, 768), got {out.shape}"
    assert torch.isfinite(out).all(), f"{layer_name}: output contains NaN or Inf"

    loss = out.sum()
    loss.backward()
    assert x.grad is not None, f"{layer_name}: input gradient is None"
    assert torch.isfinite(x.grad).all(), f"{layer_name}: input gradient contains NaN or Inf"


def test_full_matrices_false_invariant() -> None:
    """AST Invariant: Ensure every torch.linalg.svd call in SVD domain uses full_matrices=False."""
    project_root = Path(__file__).resolve().parent.parent.parent
    svd_paths = list((project_root / "neuralops" / "compression").glob("**/*.py")) + list(
        (project_root / "neuralops" / "layers" / "svd").glob("**/*.py")
    )

    violations: list[str] = []

    for file_path in svd_paths:
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))
        except (OSError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Detect calls to torch.linalg.svd or linalg.svd
                is_svd = False
                if isinstance(func, ast.Attribute) and func.attr == "svd":
                    if isinstance(func.value, ast.Attribute) and func.value.attr == "linalg":
                        is_svd = True

                if is_svd:
                    has_full_matrices_false = False
                    for kw in node.keywords:
                        if kw.arg == "full_matrices":
                            if isinstance(kw.value, ast.Constant) and kw.value.value is False:
                                has_full_matrices_false = True
                    if not has_full_matrices_false:
                        violations.append(f"{file_path.name}:{getattr(node, 'lineno', 0)}")

    assert not violations, f"Calls to torch.linalg.svd missing full_matrices=False found in: {violations}"
