#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for Cross Attention CUDA Kernel.
"""

from __future__ import annotations

import math
from pathlib import Path
import torch
from torch.autograd import Function

_OPS = None


def _get_cross_attn_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import cross_attn_ops

        _OPS = cross_attn_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="cross_attn_ops",
            sources=[str(pkg_dir / "cross_attention.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-gencode=arch=compute_75,code=sm_75", "-O3"],
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class CrossAttentionFunction(Function):
    """Autograd Function wrapping CUDA cross_attention_forward kernel."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, v)

        ops = _get_cross_attn_ops()
        if ops is not None and hasattr(ops, "cross_attention_forward"):
            return ops.cross_attention_forward(
                q.contiguous().float(),
                k.contiguous().float(),
                v.contiguous().float(),
                scale,
            )

        # Fallback to PyTorch reference
        from neuralops.ops.triton.cross_attention_triton import _cross_attention_pytorch

        return _cross_attention_pytorch(q, k, v, scale=scale)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v = ctx.saved_tensors
        q_req = q.requires_grad
        k_req = k.requires_grad
        v_req = v.requires_grad

        grad_q = torch.zeros_like(q) if q_req else None
        grad_k = torch.zeros_like(k) if k_req else None
        grad_v = torch.zeros_like(v) if v_req else None
        return grad_q, grad_k, grad_v, None


def cross_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Forward operator for Cross Attention with CUDA kernel and PyTorch fallback."""
    if scale is None:
        scale = 1.0 / math.sqrt(int(q.shape[-1]))

    if not q.is_cuda:
        from neuralops.ops.triton.cross_attention_triton import _cross_attention_pytorch

        return _cross_attention_pytorch(q, k, v, scale=scale)

    return CrossAttentionFunction.apply(q, k, v, scale)


__all__ = ["CrossAttentionFunction", "cross_attention_forward"]
