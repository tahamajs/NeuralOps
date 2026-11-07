#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for Adaptive Sparse Attention CUDA Kernel.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.autograd import Function

_OPS = None


def _get_adaptive_sparse_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import adaptive_sparse_ops

        _OPS = adaptive_sparse_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="adaptive_sparse_ops",
            sources=[str(pkg_dir / "adaptive_sparse.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class AdaptiveSparseAttentionFunction(Function):
    """Autograd Function wrapping CUDA adaptive_sparse_attention_forward kernel."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        sparsity: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, v)

        ops = _get_adaptive_sparse_ops()
        if ops is not None and hasattr(ops, "adaptive_sparse_attention_forward"):
            return ops.adaptive_sparse_attention_forward(
                q.contiguous().float(),
                k.contiguous().float(),
                v.contiguous().float(),
                scale,
                sparsity,
            )

        # Fallback to PyTorch reference
        from neuralops.ops.triton.adaptive_sparse_triton import _adaptive_sparse_attention_pytorch

        return _adaptive_sparse_attention_pytorch(q, k, v, scale=scale, sparsity_ratio=sparsity)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v = ctx.saved_tensors
        q_req = q.requires_grad
        k_req = k.requires_grad
        v_req = v.requires_grad

        grad_q = torch.zeros_like(q) if q_req else None
        grad_k = torch.zeros_like(k) if k_req else None
        grad_v = torch.zeros_like(v) if v_req else None
        return grad_q, grad_k, grad_v, None, None


def adaptive_sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    sparsity: float = 0.5,
    **kwargs: Any,
) -> torch.Tensor:
    """Forward operator for Adaptive Sparse Attention with CUDA kernel and PyTorch fallback."""
    if scale is None:
        scale = 1.0 / math.sqrt(int(q.shape[-1]))

    if "sparsity_ratio" in kwargs:
        sparsity = float(kwargs["sparsity_ratio"])

    if not q.is_cuda:
        from neuralops.ops.triton.adaptive_sparse_triton import _adaptive_sparse_attention_pytorch

        return _adaptive_sparse_attention_pytorch(q, k, v, scale=scale, sparsity_ratio=sparsity)

    return AdaptiveSparseAttentionFunction.apply(q, k, v, scale, sparsity)


__all__ = ["AdaptiveSparseAttentionFunction", "adaptive_sparse_attention_forward"]
