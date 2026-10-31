#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for AWQ CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path
import torch
from torch.autograd import Function

_OPS = None


def _get_awq_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import awq_ops

        _OPS = awq_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="awq_ops",
            sources=[str(pkg_dir / "awq.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-gencode=arch=compute_75,code=sm_75", "-O3"],
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class AWQFunction(Function):
    """Autograd Function wrapping CUDA awq_compress_forward kernel."""

    @staticmethod
    def forward(ctx, W: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> torch.Tensor:
        ctx.save_for_backward(W, scales)
        ctx.group_size = group_size

        ops = _get_awq_ops()
        if ops is not None and hasattr(ops, "awq_compress_forward"):
            return ops.awq_compress_forward(W.contiguous(), scales.contiguous(), group_size)

        # Level 3 PyTorch reference fallback
        W_fp32 = W.to(torch.float32)
        q = torch.clamp(torch.round(W_fp32 / scales), -8.0, 7.0)
        return (q * scales).to(dtype=W.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE (Straight-Through Estimator)
        return grad_output, None, None
