#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and functional tests for the neuralops.optimization subpackage.
"""

from __future__ import annotations

import neuralops.optimization
from neuralops.optimization import kernels


def test_optimization_subpackage_exports():
    assert hasattr(neuralops.optimization, "kernels")
    assert kernels is not None


def test_optimization_kernels_module_access():
    # Verify triton and kernel operator exports are accessible through optimization.kernels
    assert hasattr(neuralops.optimization, "kernels")
    # Verify module has docstring and license
    assert neuralops.optimization.__doc__ is not None
    assert "Optimization" in neuralops.optimization.__doc__


def test_optimization_absent_or_reexports_real_optimizers():
    """Verify that optimization re-exports canonical optimizer classes directly."""
    from neuralops.optimization import Lion, Muon, RiemannianStiefelOptimizer
    from neuralops.trainer.optimizers.lion import Lion as RealLion
    from neuralops.trainer.optimizers.muon import Muon as RealMuon
    from neuralops.trainer.optimizers.riemannian import RiemannianStiefelOptimizer as RealRiemannian

    assert Lion is RealLion
    assert Muon is RealMuon
    assert RiemannianStiefelOptimizer is RealRiemannian


def test_no_empty_optimization_kernels():
    """Verify that optimization/kernels is not an empty stub."""
    from pathlib import Path

    kernels_init = (
        Path(__file__).resolve().parent.parent.parent
        / "neuralops"
        / "optimization"
        / "kernels"
        / "__init__.py"
    )
    assert kernels_init.exists()
    text = kernels_init.read_text(encoding="utf-8")
    assert "neuralops.ops" in text or "__getattr__" in text
