# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Factorised Matrix State Scan Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "fms_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fms_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_fms_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fms_scan_fwd"],
        )
    return _MODULE


def fms_scan_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.fms_scan import fms_scan_pytorch
    return fms_scan_pytorch(q, k, v, decay)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.fms_scan import fms_scan
        return fms_scan
    except ImportError:
        return None


def fms_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    """3-Level Dispatched FMS Recurrent Scan."""
    def cuda_fn(q, k, v, decay, **kwargs):
        mod = _get_module()
        return mod.fms_scan_fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            decay.contiguous(),
        )

    triton_fn = _get_triton_impl()
    if q.shape[-1] > 128:
        cuda_fn = None

    return dispatch(
        cuda_fn,
        triton_fn,
        fms_scan_pytorch_ref,
        q,
        k,
        v,
        decay,
        kernel_name="fms_scan",
    )
