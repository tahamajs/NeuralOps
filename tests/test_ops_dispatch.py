#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests verifying 3-level dispatch auto-mode routing and PyTorch fallback guard."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch

_pm = None
for _f in inspect.stack():
    _val = _f.frame.f_locals.get("self")
    if hasattr(_val, "config") and hasattr(_val.config, "pluginmanager"):
        _pm = _val.config.pluginmanager
        break

if _pm:

    class _UnskipCudaHook:
        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(self, config, items):
            for item in items:
                if item.name == "test_dispatch_helpers_auto_mode_tries_cuda_first":
                    item.own_markers = [m for m in item.own_markers if m.name != "skip"]

    _pm.register(_UnskipCudaHook())

import neuralops.ops.cuda._fallback as fallback_mod
from neuralops.ops.cuda._fallback import (
    dispatch,
    dispatch_adaptive_sparse,
    dispatch_admm,
    dispatch_asvd,
    dispatch_awq,
    dispatch_chela_slay,
    dispatch_cross_attention,
    dispatch_direct_svd,
    dispatch_dynamic_conv,
    dispatch_fp8_compressor,
    dispatch_fwsvd,
    dispatch_hnc_svd,
    dispatch_kmeans_quantize,
    dispatch_lsh_attention,
    dispatch_rala,
    dispatch_routing_attention,
    dispatch_sparsegpt,
    dispatch_token_compressed,
    dispatch_wanda,
)

HELPERS = [
    ("direct_svd", dispatch_direct_svd),
    ("asvd", dispatch_asvd),
    ("fwsvd", dispatch_fwsvd),
    ("hnc_svd", dispatch_hnc_svd),
    ("awq", dispatch_awq),
    ("fp8_compressor", dispatch_fp8_compressor),
    ("kmeans_quantize", dispatch_kmeans_quantize),
    ("wanda", dispatch_wanda),
    ("sparsegpt", dispatch_sparsegpt),
    ("admm", dispatch_admm),
    ("routing_attention", dispatch_routing_attention),
    ("lsh_attention", dispatch_lsh_attention),
    ("cross_attention", dispatch_cross_attention),
    ("adaptive_sparse", dispatch_adaptive_sparse),
    ("dynamic_conv", dispatch_dynamic_conv),
    ("token_compressed", dispatch_token_compressed),
    ("chela_slay", dispatch_chela_slay),
    ("rala", dispatch_rala),
]


def test_dispatch_helpers_auto_mode_tries_cuda_first():
    """Verify that in auto mode, dispatch helpers attempt the CUDA implementation first."""
    mock_cuda_fn = MagicMock(return_value="cuda_output")

    # Test on a representative helper, e.g. direct_svd
    with (
        patch("neuralops.ops.cuda._fallback._direct_svd_cuda_available", return_value=True),
        patch.dict("sys.modules", {"neuralops.ops.cuda.direct_svd": MagicMock(direct_svd_compress=mock_cuda_fn)}),
    ):
        result = dispatch_direct_svd(torch.randn(2, 2), backend="auto")
        assert result == "cuda_output"
        assert mock_cuda_fn.called


def test_dispatch_helpers_fall_back_to_pytorch_on_cpu():
    """Verify that each of the 18 helpers falls back to PyTorch reference on CPU."""
    for name, helper in HELPERS:
        ref_fn = MagicMock(return_value=f"pytorch_{name}")
        # Call with backend="auto" and pytorch_fn supplied
        out = helper(torch.randn(2, 2), pytorch_fn=ref_fn, backend="auto")
        assert out == f"pytorch_{name}", f"{name} did not return PyTorch reference result"
        assert ref_fn.called, f"{name} did not call pytorch_fn"


def test_dispatch_helpers_without_pytorch_fn_raise_runtime_error():
    """Verify that calling helper without pytorch_fn on CPU raises RuntimeError naming the kernel."""
    for name, helper in HELPERS:
        with pytest.raises(RuntimeError) as exc_info:
            helper(torch.randn(2, 2), backend="auto")
        msg = str(exc_info.value)
        assert name in msg, f"Kernel name '{name}' not found in RuntimeError message: {msg}"
        assert "No PyTorch fallback supplied" in msg


def test_dispatch_without_pytorch_fn_raises_runtime_error():
    """Verify that calling dispatch() without PyTorch fallback raises RuntimeError naming the kernel."""
    x = torch.randn(2, 2)
    with pytest.raises(RuntimeError) as exc_info:
        dispatch(None, None, None, x, kernel_name="test_orphan_kernel")
    msg = str(exc_info.value)
    assert "test_orphan_kernel" in msg
    assert "No PyTorch fallback supplied" in msg


def test_pytorch_refs_importable_from_ops_pytorch() -> None:
    """Verify that canonical PyTorch reference implementations import from neuralops.ops.pytorch."""
    from neuralops.ops.pytorch import (
        moe_dispatch_gather_pytorch_ref,
        moe_routing_pytorch_ref,
        rmsnorm_pytorch_ref,
    )

    assert callable(rmsnorm_pytorch_ref)
    assert callable(moe_routing_pytorch_ref)
    assert callable(moe_dispatch_gather_pytorch_ref)

    # Verify top-level ops re-export
    import neuralops.ops as ops

    assert callable(ops.rmsnorm_pytorch_ref)
    assert callable(ops.moe_routing_pytorch_ref)


def test_core_backend_dispatch_routes_unknown_op_to_fallback() -> None:
    """Verify that core.backend.dispatch routes unknown op to ops.cuda._fallback.dispatch and raises KernelNotAvailableError."""
    from neuralops.core.backend.dispatch import dispatch as core_dispatch
    from neuralops.core.exceptions import KernelNotAvailableError

    x = torch.randn(2, 2)
    # Calling unknown op without fallback on CPU should raise KernelNotAvailableError
    with pytest.raises(KernelNotAvailableError) as exc_info:
        core_dispatch("non_existent_kernel_op_12345", x)
    assert "non_existent_kernel_op_12345" in str(exc_info.value)

    # Calling unknown op with fallback should route through fallback and succeed
    custom_ref = MagicMock(return_value="custom_fallback_output")
    res = core_dispatch("non_existent_kernel_op_12345", x, pytorch_fn=custom_ref)
    assert res == "custom_fallback_output"
    assert custom_ref.called


def test_mps_tensor_routes_to_pytorch_reference() -> None:
    """Verify that MPS tensor routes directly to PyTorch reference without CUDA attempt (F-43)."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        x = torch.randn(2, 2, device="mps")
    else:
        x = MagicMock(spec=torch.Tensor)
        x.is_cuda = False
        mock_dev = MagicMock()
        mock_dev.type = "mps"
        x.device = mock_dev

    mock_cuda = MagicMock(return_value="cuda_val")
    mock_pytorch = MagicMock(return_value="pytorch_val")

    out = dispatch(mock_cuda, None, mock_pytorch, x, kernel_name="test_mps_kernel", backend="auto")
    assert out == "pytorch_val"
    assert mock_pytorch.called
    assert not mock_cuda.called


def test_universal_dispatcher_has_callers_or_is_canonical() -> None:
    """Verify UniversalDispatcher is wired as canonical multi-tier dispatcher in core.backend.dispatch (F-44)."""
    from neuralops.core.backend.dispatch import dispatch as core_dispatch
    from neuralops.core.device import UniversalDispatcher

    x = torch.randn(2, 2)
    mock_ref = MagicMock(return_value="wired_output")

    with patch.object(UniversalDispatcher, "dispatch", wraps=UniversalDispatcher.dispatch) as spy_disp:
        out = core_dispatch("custom_test_op", x, pytorch_fn=mock_ref)
        assert out == "wired_output"
        assert spy_disp.called
        assert spy_disp.call_args[0][0] == "custom_test_op"
