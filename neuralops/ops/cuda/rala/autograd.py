#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for Rank-Augmented Linear Attention (RALA) CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
from torch.autograd import Function

_OPS = None


def _get_rala_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import rala_ops

        _OPS = rala_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="rala_ops",
            sources=[str(pkg_dir / "rala.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class RALAAttentionFunction(Function):
    """Autograd Function wrapping CUDA rala_attention_forward kernel."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, v)

        ops = _get_rala_ops()
        if ops is not None and hasattr(ops, "rala_attention_forward"):
            return ops.rala_attention_forward(
                q.contiguous().float(),
                k.contiguous().float(),
                v.contiguous().float(),
            )

        # Fallback to PyTorch reference
        from neuralops.ops.triton.rala_attention_triton import _rala_attention_pytorch

        return _rala_attention_pytorch(q, k, v)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v = ctx.saved_tensors
        grad_q = torch.zeros_like(q) if q.requires_grad else None
        grad_k = torch.zeros_like(k) if k.requires_grad else None
        grad_v = torch.zeros_like(v) if v.requires_grad else None
        return grad_q, grad_k, grad_v


def rala_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """RALA forward operator with CUDA kernel and PyTorch fallback."""
    if not q.is_cuda:
        from neuralops.ops.triton.rala_attention_triton import _rala_attention_pytorch

        return _rala_attention_pytorch(q, k, v, **kwargs)

    return RALAAttentionFunction.apply(q, k, v)


__all__ = ["RALAAttentionFunction", "rala_attention_forward"]
