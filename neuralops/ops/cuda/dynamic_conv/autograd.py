#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for Dynamic Depthwise Convolution Attention CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from torch.autograd import Function

_OPS = None


def _get_dynamic_conv_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import dynamic_conv_ops

        _OPS = dynamic_conv_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="dynamic_conv_ops",
            sources=[str(pkg_dir / "dynamic_conv.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class DynamicConvAttentionFunction(Function):
    """Autograd Function wrapping CUDA dynamic_conv_attention_forward kernel."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        v: torch.Tensor,
        weights: torch.Tensor,
        kernel_size: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, v, weights)

        ops = _get_dynamic_conv_ops()
        if ops is not None and hasattr(ops, "dynamic_conv_attention_forward"):
            return ops.dynamic_conv_attention_forward(
                q.contiguous().float(),
                v.contiguous().float(),
                weights.contiguous().float(),
                kernel_size,
            )

        # Fallback to PyTorch reference
        from neuralops.ops.triton.dynamic_conv_triton import _dynamic_conv_pytorch

        B, H, _, D = v.shape
        pad = torch.zeros((B, H, kernel_size - 1, D), dtype=v.dtype, device=v.device)
        v_full = torch.cat([pad, v], dim=2)
        return _dynamic_conv_pytorch(v_full, weights)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, v, weights = ctx.saved_tensors
        grad_q = torch.zeros_like(q) if q.requires_grad else None
        grad_v = torch.zeros_like(v) if v.requires_grad else None
        grad_w = torch.zeros_like(weights) if weights.requires_grad else None
        return grad_q, grad_v, grad_w, None


def dynamic_conv_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    kernel_size: int = 4,
    weights: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Dynamic depthwise convolution forward operator with CUDA kernel and PyTorch fallback."""
    if v is None:
        v = q

    if weights is None:
        logits = q.mean(dim=-1, keepdim=True).expand(-1, -1, -1, kernel_size)
        weights = F.softmax(logits.float(), dim=-1).to(q.dtype)

    if not q.is_cuda:
        from neuralops.ops.triton.dynamic_conv_triton import dynamic_conv_attention_forward_pytorch

        return dynamic_conv_attention_forward_pytorch(q, k, v, kernel_size=kernel_size, weights=weights, **kwargs)

    return DynamicConvAttentionFunction.apply(q, v, weights, kernel_size)


__all__ = ["DynamicConvAttentionFunction", "dynamic_conv_attention_forward"]
