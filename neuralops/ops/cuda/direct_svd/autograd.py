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

import torch
from torch.autograd import Function


def _get_direct_svd_ops():
    try:
        import direct_svd_ops

        return direct_svd_ops
    except ImportError:
        try:
            from pathlib import Path

            from neuralops.ops.cuda._bindings import load_cuda_module

            pkg_dir = Path(__file__).parent
            with open(pkg_dir / "binding.cpp", "r", encoding="utf-8") as f:
                cpp_src = f.read()
            with open(pkg_dir / "direct_svd.cu", "r", encoding="utf-8") as f:
                cuda_src = f.read()

            return load_cuda_module(
                name="direct_svd_ops",
                cpp_sources=cpp_src,
                cuda_sources=cuda_src,
                functions=["direct_svd_fwd"],
            )
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
