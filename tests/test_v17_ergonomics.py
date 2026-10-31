#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive test suite for NeuralOps v1.7.0 high-level ergonomics and public API contracts.

Tests:
1. Canonical package version check: assert `neuralops.__version__ == "1.7.0"` and `neuralops.get_version() == "1.7.0"`.
2. Public symbol exposure in `neuralops`: assert that `transplant`, `export`, `compress`, `replace_attention`
   are directly importable from `neuralops` and listed in `__all__`.
3. `transplant` alias behavior delegating to `replace_attention`.
4. `export` alias behavior delegating to `export_model`.
5. End-to-end integration of high-level ergonomics on a toy PyTorch model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

import neuralops
from neuralops import (
    compress,
    export,
    export_model,
    get_version,
    replace_attention,
    transplant,
    transplant_attention,
)


class DummyAttention(nn.Module):
    """Toy attention layer for testing transplantation and replacement."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-1, -2))
        return torch.matmul(scores, v)


class ToyTransformerBlock(nn.Module):
    """Toy transformer block containing self_attn and mlp."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.self_attn = DummyAttention(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.self_attn(x))


class ToyModel(nn.Module):
    """Toy model containing stacked blocks."""

    def __init__(self, num_layers: int = 2, dim: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyTransformerBlock(dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def test_canonical_package_version() -> None:
    """Verify that neuralops.__version__ and neuralops.get_version() report exactly '1.7.0'."""
    assert neuralops.__version__ == "1.7.0"
    assert get_version() == "1.7.0"
    assert neuralops.get_version() == "1.7.0"


def test_public_symbol_exposure() -> None:
    """Verify that transplant, export, compress, replace_attention are exposed in __all__ and importable."""
    expected_symbols = ["transplant", "export", "compress", "replace_attention"]
    for sym in expected_symbols:
        assert sym in neuralops.__all__, f"Symbol '{sym}' is missing from neuralops.__all__"
        attr = getattr(neuralops, sym, None)
        assert attr is not None, f"neuralops has no attribute '{sym}'"
        assert callable(attr), f"neuralops.{sym} is not callable"

    # Verify that directly imported symbols match top-level module attributes
    assert transplant is neuralops.transplant
    assert export is neuralops.export
    assert compress is neuralops.compress
    assert replace_attention is neuralops.replace_attention


def test_transplant_alias_identity_and_delegation() -> None:
    """Verify that transplant is an alias for transplant_attention and delegates to replace_attention."""
    # Identity check
    assert transplant is transplant_attention
    assert neuralops.transplant is neuralops.transplant_attention

    # Delegation check with mock
    dummy_model = ToyModel()
    with patch("neuralops.api.replace_attention") as mock_replace:
        mock_replace.return_value = dummy_model

        result = transplant(
            model=dummy_model,
            attention="gla",
            rank=16,
            copy_weights=False,
            layers=[0],
            custom_arg="test_val",
        )

        mock_replace.assert_called_once_with(
            model=dummy_model,
            attention="gla",
            rank=16,
            copy_weights=False,
            layers=[0],
            custom_arg="test_val",
        )
        assert result is dummy_model

    # Check argument alias resolution (attention_type and target_attention)
    with patch("neuralops.api.replace_attention") as mock_replace:
        mock_replace.return_value = dummy_model

        transplant(model=dummy_model, attention_type="cosformer")
        mock_replace.assert_called_once_with(
            model=dummy_model,
            attention="cosformer",
            rank=None,
            copy_weights=True,
            layers=None,
        )

    with patch("neuralops.api.replace_attention") as mock_replace:
        mock_replace.return_value = dummy_model

        transplant(model=dummy_model, target_attention="retnet")
        mock_replace.assert_called_once_with(
            model=dummy_model,
            attention="retnet",
            rank=None,
            copy_weights=True,
            layers=None,
        )


def test_export_alias_identity_and_delegation() -> None:
    """Verify that export is an alias for export_model and delegates to _export_model."""
    # Identity check
    assert export is export_model
    assert neuralops.export is neuralops.export_model

    # Delegation check with mock
    dummy_model = ToyModel()
    with patch("neuralops.api._export_model") as mock_export_impl:
        mock_export_impl.return_value = Path("/tmp/mock_export_dir")

        out = export(
            model=dummy_model,
            output_dir="/tmp/mock_export_dir",
            format="safetensors",
            extra_meta="ok",
        )

        mock_export_impl.assert_called_once_with(
            model=dummy_model,
            output_dir="/tmp/mock_export_dir",
            format="safetensors",
            extra_meta="ok",
        )
        assert out == Path("/tmp/mock_export_dir")


def test_export_end_to_end_toy_model() -> None:
    """Verify that neuralops.export works end-to-end saving safetensors / neuralops_config."""
    model = ToyModel(num_layers=1, dim=16)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = export(model=model, output_dir=tmp_dir, format="safetensors")
        assert Path(out_path).exists()
        config_file = Path(tmp_dir) / "neuralops_config.json"
        assert config_file.exists(), f"Expected config file at {config_file}"


def test_compress_symbol_callable() -> None:
    """Verify that neuralops.compress is callable and delegates to api.compress."""
    dummy_model = MagicMock(spec=nn.Module)
    with patch("neuralops.api.compress") as mock_compress:
        mock_compress.return_value = dummy_model

        res = compress(model=dummy_model, method="svd", ratio=0.5)
        mock_compress.assert_called_once_with(model=dummy_model, method="svd", ratio=0.5)
        assert res is dummy_model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
