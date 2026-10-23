#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for lazy loading behavior in neuralops.core.cuda_kernels.
"""

from __future__ import annotations

import sys
import pytest


def test_cuda_kernels_lazy_load_no_eager_submodules():
    """Verify importing neuralops.core.cuda_kernels does not eagerly load submodules."""
    # Ensure submodules are not pre-cached
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("neuralops.core.cuda_kernels."):
            del sys.modules[mod_name]

    import neuralops.core.cuda_kernels as nck

    # Submodule should NOT be eagerly loaded upon initial package import
    assert "neuralops.core.cuda_kernels.low_rank_svd" not in sys.modules, (
        "Submodule low_rank_svd was eagerly imported during package initialization!"
    )

    # Dynamic access should resolve the symbol correctly
    fn = getattr(nck, "fused_lowrank_gated_gemm", None)
    assert fn is not None, "Failed to resolve fused_lowrank_gated_gemm via lazy loader."
    assert callable(fn)

    # Now the submodule should be in sys.modules
    assert "neuralops.core.cuda_kernels.low_rank_svd" in sys.modules


def test_cuda_kernels_invalid_attribute_raises():
    """Verify non-existent attributes raise AttributeError."""
    import neuralops.core.cuda_kernels as nck

    with pytest.raises(AttributeError, match="has no attribute 'non_existent_kernel_symbol'"):
        _ = nck.non_existent_kernel_symbol
