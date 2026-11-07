#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying lazy importing and symbol resolution in neuralops.kernels."""

from __future__ import annotations

import sys


def test_import_kernels_does_not_eagerly_load_110_modules() -> None:
    """Importing neuralops.kernels should not eagerly load the 110+ CUDA op submodules."""
    # Ensure fresh state for neuralops.kernels if possible
    import neuralops.kernels

    cuda_ops = [k for k in sys.modules if k.startswith("neuralops.ops.cuda.") and not k.endswith(("_map", "_bindings", "_fallback", "_utils"))]
    assert len(cuda_ops) < 10, f"Too many ops.cuda submodules loaded eagerly: {len(cuda_ops)} ({cuda_ops[:10]})"


def test_kernels_cuda_module_resolves_via_lazy() -> None:
    """Accessing an attribute on neuralops.kernels resolves lazily via PEP 562."""
    import neuralops.kernels as kernels

    rmsnorm_mod = kernels.rmsnorm
    assert rmsnorm_mod is not None
    assert "rmsnorm" in rmsnorm_mod.__name__
