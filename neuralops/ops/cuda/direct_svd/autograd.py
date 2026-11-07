#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for Direct SVD CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.autograd import Function

_OPS = None


def _get_direct_svd_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import direct_svd_ops

        _OPS = direct_svd_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="direct_svd_ops",
            sources=[str(pkg_dir / "direct_svd.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class DirectSVDFunction(Function):
    """Autograd Function wrapping CUDA direct_svd_fwd kernel."""

    @staticmethod
    def forward(ctx, U: torch.Tensor, V: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(U, V)
        ctx.scale = scale

        ops = _get_direct_svd_ops()
        if ops is not None and hasattr(ops, "direct_svd_fwd"):
            return ops.direct_svd_fwd(U.contiguous(), V.contiguous(), float(scale))

        # Level 3 PyTorch reference fallback
        return torch.matmul(U, V) * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        U, V = ctx.saved_tensors
        scale = ctx.scale

        grad_scaled = grad_output * scale
        grad_U = torch.matmul(grad_scaled, V.t())
        grad_V = torch.matmul(U.t(), grad_scaled)
        return grad_U, grad_V, None
