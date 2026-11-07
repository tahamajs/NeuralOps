#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for CHELA-SLAY Attention CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.autograd import Function

_OPS = None


def _get_chela_slay_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import chela_slay_ops

        _OPS = chela_slay_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="chela_slay_ops",
            sources=[str(pkg_dir / "chela_slay.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class ChelaSlayAttentionFunction(Function):
    """Autograd Function wrapping CUDA chela_slay_attention_forward kernel."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, v)

        ops = _get_chela_slay_ops()
        if ops is not None and hasattr(ops, "chela_slay_attention_forward"):
            return ops.chela_slay_attention_forward(
                q.contiguous().float(),
                k.contiguous().float(),
                v.contiguous().float(),
                chunk_size,
            )

        # Fallback to PyTorch reference
        from neuralops.ops.triton.chela_slay_triton import _chela_slay_pytorch

        return _chela_slay_pytorch(q, k, v, chunk_size=chunk_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v = ctx.saved_tensors
        grad_q = torch.zeros_like(q) if q.requires_grad else None
        grad_k = torch.zeros_like(k) if k.requires_grad else None
        grad_v = torch.zeros_like(v) if v.requires_grad else None
        return grad_q, grad_k, grad_v, None


def chela_slay_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 64,
    **kwargs: Any,
) -> torch.Tensor:
    """CHELA-SLAY attention forward operator with CUDA kernel and PyTorch fallback."""
    if not q.is_cuda:
        from neuralops.ops.triton.chela_slay_triton import _chela_slay_pytorch

        return _chela_slay_pytorch(q, k, v, chunk_size=chunk_size, **kwargs)

    return ChelaSlayAttentionFunction.apply(q, k, v, chunk_size)


__all__ = ["ChelaSlayAttentionFunction", "chela_slay_attention_forward"]
