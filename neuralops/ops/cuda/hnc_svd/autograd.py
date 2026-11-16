#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for HNC-SVD CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.autograd import Function

_OPS = None


def _get_hnc_svd_ops() -> Any:
    global _OPS
    if _OPS is not None:
        return _OPS

    if not torch.cuda.is_available():
        return None

    try:
        import hnc_svd_ops

        _OPS = hnc_svd_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="hnc_svd_ops",
            sources=[str(pkg_dir / "hnc_svd.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class HNCSVDFunction(Function):
    """Autograd Function wrapping CUDA hnc_svd_fwd kernel."""

    @staticmethod
    def forward(ctx: Any, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(A, B)

        ops = _get_hnc_svd_ops()
        if (
            ops is not None
            and hasattr(ops, "hnc_svd_fwd")
            and A.is_cuda
            and B.is_cuda
            and A.dtype == torch.float32
            and B.dtype == torch.float32
        ):
            return ops.hnc_svd_fwd(A.contiguous(), B.contiguous())

        # Level 3 PyTorch reference fallback
        return torch.matmul(A, B)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        A, B = ctx.saved_tensors
        grad_A = torch.matmul(grad_output, B.t())
        grad_B = torch.matmul(A.t(), grad_output)
        return grad_A, grad_B
