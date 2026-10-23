#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for lazy loading in neuralops.core.cuda_kernels (Phase 1 / P001, P011).
"""

from __future__ import annotations

import sys
import pytest


def test_no_eager_imports():
    """Verify bare import does not load any kernel subpackages into sys.modules."""
    for mod in list(sys.modules.keys()):
        if mod.startswith("neuralops.core.cuda_kernels."):
            del sys.modules[mod]

    import neuralops.core.cuda_kernels as nck  # noqa: F401

    eager_submodules = [
        m for m in sys.modules
        if m.startswith("neuralops.core.cuda_kernels.")
        and not m.endswith("._map")
    ]
    assert len(eager_submodules) == 0, f"Found eager submodules loaded: {eager_submodules}"


def test_all_symbols_resolve():
    """Verify every symbol listed in __all__ resolves via getattr."""
    import neuralops.core.cuda_kernels as nck

    for symbol in nck.__all__:
        val = getattr(nck, symbol, None)
        assert val is not None, f"Failed to resolve symbol {symbol} via getattr"


def test_catalog_preserved():
    """Verify CUDA_KERNEL_CATALOG is present and list_cuda_kernels() returns it."""
    import neuralops.core.cuda_kernels as nck

    assert hasattr(nck, "CUDA_KERNEL_CATALOG")
    assert isinstance(nck.CUDA_KERNEL_CATALOG, list)
    assert len(nck.CUDA_KERNEL_CATALOG) > 50

    catalog = nck.list_cuda_kernels()
    assert isinstance(catalog, list)
    assert len(catalog) == len(nck.CUDA_KERNEL_CATALOG)


def test_aliases_and_special_exports():
    """Verify aliases and special wrappers resolve properly."""
    import neuralops.core.cuda_kernels as nck

    assert hasattr(nck, "retnet_retention")
    assert hasattr(nck, "moe_dispatch")
    assert hasattr(nck, "hgrn2_gated_deltanet")
    assert callable(nck.retnet_retention)
    assert callable(nck.moe_dispatch)
    assert callable(nck.hgrn2_gated_deltanet)


def test_annotations_not_resolvable():
    """Verify 'annotations' from __future__ is not a valid exported symbol."""
    import neuralops.core.cuda_kernels as nck

    with pytest.raises(AttributeError):
        _ = nck.annotations


def test_unknown_names_raise_attribute_error():
    """Verify non-existent symbols raise AttributeError."""
    import neuralops.core.cuda_kernels as nck

    with pytest.raises(AttributeError, match="has no attribute 'totally_unknown_kernel_name'"):
        _ = nck.totally_unknown_kernel_name
