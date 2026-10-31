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

from typing import Optional

import torch
from torch.autograd import Function


def _get_asvd_ops():
    try:
        import asvd_ops

        return asvd_ops
    except ImportError:
        try:
            from pathlib import Path

            from neuralops.ops.cuda._bindings import load_cuda_module

            pkg_dir = Path(__file__).parent
            with open(pkg_dir / "binding.cpp", "r", encoding="utf-8") as f:
                cpp_src = f.read()
            with open(pkg_dir / "asvd.cu", "r", encoding="utf-8") as f:
                cuda_src = f.read()

            return load_cuda_module(
                name="asvd_ops",
                cpp_sources=cpp_src,
                cuda_sources=cuda_src,
                functions=["asvd_fwd"],
            )
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
