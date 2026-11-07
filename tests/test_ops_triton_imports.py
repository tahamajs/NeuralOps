#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying Triton kernel import contracts and mock safety (F-32, F-33, F-34)."""

from __future__ import annotations

import importlib.util

import pytest


def test_has_triton_reflects_real_import() -> None:
    """Verify that HAS_TRITON strictly reflects real triton package availability."""
    from neuralops.core.triton_kernels import HAS_TRITON as CORE_HAS_TRITON
    from neuralops.ops.triton import HAS_TRITON as OPS_HAS_TRITON
    from neuralops.ops.triton._base import HAS_TRITON as BASE_HAS_TRITON

    real_available = importlib.util.find_spec("triton") is not None
    assert BASE_HAS_TRITON == real_available
    assert OPS_HAS_TRITON == real_available
    assert CORE_HAS_TRITON == real_available


def test_no_none_kernel_exports() -> None:
    """Verify that no symbol exported in neuralops.ops.triton.__all__ is None."""
    import neuralops.ops.triton as triton_ops

    none_exports = []
    for name in triton_ops.__all__:
        try:
            val = getattr(triton_ops, name)
            if val is None:
                none_exports.append(name)
        except (ImportError, AttributeError):
            # Absent kernels raising ImportError rather than returning None is correct fail-closed behavior
            pass

    assert not none_exports, f"The following kernels in ops.triton.__all__ exported as None: {none_exports}"


def test_triton_getattr_raises_import_error_not_none() -> None:
    """Verify that accessing unavailable Triton kernels raises ImportError instead of returning None."""
    import neuralops.ops.triton as triton_ops

    # On a host without Triton, attempting to resolve absent hardware kernels raises clean ImportError
    if not triton_ops.HAS_TRITON:
        with pytest.raises(ImportError) as exc_info:
            _ = triton_ops.uncompiled_custom_scan_triton
        assert "Triton kernel" in str(exc_info.value)
        assert "Triton is not installed" in str(exc_info.value)


def test_mock_triton_jit_raises_on_call() -> None:
    """Verify that executing a mock @triton.jit decorated kernel raises ImportError when called."""
    from neuralops.ops.triton._base import HAS_TRITON, tl, triton

    if not HAS_TRITON:
        @triton.jit
        def dummy_kernel(x_ptr, n_elements):
            pass

        with pytest.raises(ImportError) as exc_info:
            dummy_kernel(None, 128)
        assert "cannot be executed because Triton is not installed" in str(exc_info.value)

        # Also verify tl language attributes raise ImportError on access
        with pytest.raises(ImportError) as tl_exc:
            _ = tl.load
        assert "tl.load" in str(tl_exc.value)
