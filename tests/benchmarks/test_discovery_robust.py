# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying discovery robustness when submodules or catalogs are missing or broken."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

import neuralops.benchmarks.discovery as discovery
from neuralops.benchmarks.discovery import (
    ModuleMetadata,
    discover_all_modules,
    discover_modules,
    inspect_module_capabilities,
)


def test_discover_all_modules_handles_missing_attention_submodule(tmp_path: pytest.TempPathFactory) -> None:
    """Mock missing attention submodule and verify discover_all_modules runs gracefully."""
    fake_catalog = tmp_path / "missing_catalog.json"  # type: ignore[operator]

    mock_attention_registry = MagicMock()
    mock_attention_registry.keys.return_value = ["good_attention", "broken_missing_submodule"]

    class GoodAttention:
        supports_causal = True
        supports_gqa = False

    def fake_getitem(key: str) -> type:
        if key == "broken_missing_submodule":
            raise ModuleNotFoundError("No module named 'neuralops.layers.broken_missing'")
        return GoodAttention

    mock_attention_registry.__getitem__.side_effect = fake_getitem

    mock_compressor_registry = MagicMock()
    mock_compressor_registry.keys.return_value = []

    with patch.object(discovery, "CATALOG_PATH", fake_catalog):
        with patch.dict(
            "sys.modules",
            {
                "neuralops.core.registry": MagicMock(
                    ATTENTION_REGISTRY=mock_attention_registry,
                    COMPRESSOR_REGISTRY=mock_compressor_registry,
                )
            },
        ):
            modules = discover_all_modules()
            assert isinstance(modules, list)
            names = [m.name for m in modules]
            assert "good_attention" in names
            assert "broken_missing_submodule" not in names


def test_discover_all_modules_handles_missing_compressor_submodule(tmp_path: pytest.TempPathFactory) -> None:
    """Mock missing compressor submodule and verify discover_all_modules runs gracefully."""
    fake_catalog = tmp_path / "missing_catalog.json"  # type: ignore[operator]

    mock_attention_registry = MagicMock()
    mock_attention_registry.keys.return_value = []

    mock_compressor_registry = MagicMock()
    mock_compressor_registry.keys.return_value = ["good_compressor", "broken_compressor"]

    class GoodCompressor:
        pass

    def fake_getitem(key: str) -> type:
        if key == "broken_compressor":
            raise ImportError("Simulated DLL or CUDA driver failure for compressor")
        return GoodCompressor

    mock_compressor_registry.__getitem__.side_effect = fake_getitem

    with patch.object(discovery, "CATALOG_PATH", fake_catalog):
        with patch.dict(
            "sys.modules",
            {
                "neuralops.core.registry": MagicMock(
                    ATTENTION_REGISTRY=mock_attention_registry,
                    COMPRESSOR_REGISTRY=mock_compressor_registry,
                )
            },
        ):
            modules = discover_all_modules()
            assert isinstance(modules, list)
            names = [m.name for m in modules]
            assert "good_compressor" in names
            assert "broken_compressor" not in names


def test_inspect_module_capabilities_robust_to_broken_objects() -> None:
    """Verify inspect_module_capabilities handles non-standard or broken classes without raising."""

    class BrokenInit:
        def __init__(self) -> None:
            pass

    # Break signature introspection
    BrokenInit.__init__ = None  # type: ignore[assignment]

    meta = inspect_module_capabilities("broken_init", BrokenInit, "attention")
    assert isinstance(meta, ModuleMetadata)
    assert meta.name == "broken_init"
    assert meta.type == "attention"

    # None object
    meta_none = inspect_module_capabilities("none_cls", None, "compressor")
    assert isinstance(meta_none, ModuleMetadata)
    assert meta_none.name == "none_cls"
    assert meta_none.type == "compressor"


def test_corrupt_static_catalog_graceful_fallback(tmp_path: pytest.TempPathFactory) -> None:
    """Verify corrupted static catalog triggers warning and graceful fallback."""
    corrupt_catalog = tmp_path / "corrupt_catalog.json"  # type: ignore[operator]
    corrupt_catalog.write_text("{invalid json", encoding="utf-8")

    with patch.object(discovery, "CATALOG_PATH", corrupt_catalog):
        modules = discover_all_modules()
        assert isinstance(modules, list)
        # Even with corrupt catalog, falls back to dynamic discovery and returns modules
        assert len(modules) > 0


def test_discover_modules_mapping() -> None:
    """Verify discover_modules dictionary mapping functionality."""
    by_name = discover_modules(key_by="name")
    assert isinstance(by_name, dict)
    assert len(by_name) >= 200

    by_key = discover_modules(key_by="unique_key")
    assert isinstance(by_key, dict)
    assert len(by_key) >= 200
    assert all(":" in k for k in by_key)
