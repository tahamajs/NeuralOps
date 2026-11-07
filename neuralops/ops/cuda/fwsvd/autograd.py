#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for FWSVD CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.autograd import Function

_OPS = None


def _get_fwsvd_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import fwsvd_ops

        _OPS = fwsvd_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="fwsvd_ops",
            sources=[str(pkg_dir / "fwsvd.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class FWSVDFunction(Function):
    """Autograd Function wrapping CUDA fwsvd_fwd kernel."""

    @staticmethod
    def forward(ctx, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(U, V)

        ops = _get_fwsvd_ops()
        if ops is not None and hasattr(ops, "fwsvd_fwd"):
            return ops.fwsvd_fwd(U.contiguous(), V.contiguous())

        # Level 3 PyTorch reference fallback
        return torch.matmul(U, V)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        U, V = ctx.saved_tensors
        grad_U = torch.matmul(grad_output, V.t())
        grad_V = torch.matmul(U.t(), grad_output)
        return grad_U, grad_V
