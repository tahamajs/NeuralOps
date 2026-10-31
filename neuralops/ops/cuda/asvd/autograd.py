#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for ASVD CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.autograd import Function

_OPS = None


def _get_asvd_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import asvd_ops

        _OPS = asvd_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="asvd_ops",
            sources=[str(pkg_dir / "asvd.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-gencode=arch=compute_75,code=sm_75", "-O3"],
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class ASVDFunction(Function):
    """Autograd Function wrapping CUDA asvd_fwd kernel."""

    @staticmethod
    def forward(
        ctx,
        U: torch.Tensor,
        V: torch.Tensor,
        row_gain: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ctx.save_for_backward(U, V, row_gain)

        ops = _get_asvd_ops()
        if ops is not None and hasattr(ops, "asvd_fwd"):
            return ops.asvd_fwd(U.contiguous(), V.contiguous(), row_gain)

        # Level 3 PyTorch reference fallback
        out = torch.matmul(U, V)
        if row_gain is not None:
            out = out * row_gain.unsqueeze(-1)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        U, V, row_gain = ctx.saved_tensors

        grad_scaled = grad_output
        if row_gain is not None:
            grad_scaled = grad_output * row_gain.unsqueeze(-1)

        grad_U = torch.matmul(grad_scaled, V.t())
        grad_V = torch.matmul(U.t(), grad_scaled)
        grad_gain = None
        if row_gain is not None:
            recon = torch.matmul(U, V)
            grad_gain = (grad_output * recon).sum(dim=-1)
        return grad_U, grad_V, grad_gain
