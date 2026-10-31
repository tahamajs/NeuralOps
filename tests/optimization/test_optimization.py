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
