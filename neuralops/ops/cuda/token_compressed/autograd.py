#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for Token-Compressed Attention CUDA Kernel.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import torch
from torch.autograd import Function

_OPS = None


def _get_token_compressed_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import token_compressed_ops

        _OPS = token_compressed_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="token_compressed_ops",
            sources=[str(pkg_dir / "token_compressed.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class TokenCompressedAttentionFunction(Function):
    """Autograd Function wrapping CUDA token_compressed_attention_forward kernel."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_projected_tokens: int,
        scale: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, v)

        ops = _get_token_compressed_ops()
        if ops is not None and hasattr(ops, "token_compressed_attention_forward"):
            return ops.token_compressed_attention_forward(
                q.contiguous().float(),
                k.contiguous().float(),
                v.contiguous().float(),
                num_projected_tokens,
                scale,
            )

        # Fallback to PyTorch reference
        from neuralops.ops.triton.token_compressed_triton import _token_compressed_attention_pytorch

        return _token_compressed_attention_pytorch(
            q, k, v, num_projected_tokens=num_projected_tokens, scale=scale
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v = ctx.saved_tensors
        grad_q = torch.zeros_like(q) if q.requires_grad else None
        grad_k = torch.zeros_like(k) if k.requires_grad else None
        grad_v = torch.zeros_like(v) if v.requires_grad else None
        return grad_q, grad_k, grad_v, None, None


def token_compressed_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_projected_tokens: int = 4,
    scale: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Token-compressed attention forward operator with CUDA kernel and PyTorch fallback."""
    if scale is None:
        scale = 1.0 / math.sqrt(int(q.shape[-1]))

    if not q.is_cuda:
        from neuralops.ops.triton.token_compressed_triton import _token_compressed_attention_pytorch

        return _token_compressed_attention_pytorch(
            q, k, v, num_projected_tokens=num_projected_tokens, scale=scale, **kwargs
        )

    return TokenCompressedAttentionFunction.apply(q, k, v, num_projected_tokens, scale)


__all__ = ["TokenCompressedAttentionFunction", "token_compressed_attention_forward"]
