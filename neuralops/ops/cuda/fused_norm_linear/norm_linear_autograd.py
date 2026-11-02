#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused Normalization Linear Projection Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from neuralops.ops.cuda._bindings import load_cuda_module
from neuralops.ops.cuda._fallback import dispatch

logger = logging.getLogger(__name__)

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "norm_linear_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fused_norm_linear.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_fused_norm_linear",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fused_norm_linear_fwd", "fused_norm_linear_bwd_dx"],
        )
    return _MODULE


def fused_norm_linear_pytorch_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.ops.triton.fused_norm_linear import fused_norm_linear_pytorch
    return fused_norm_linear_pytorch(x, weight, bias=bias, eps=eps)


def _get_triton_impl():
    try:
        from neuralops.ops.triton.fused_norm_linear import fused_norm_linear
        return fused_norm_linear
    except ImportError:
        return None


class _FusedNormLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        orig_shape = x.shape
        D = orig_shape[-1]
        x_2d = x.reshape(-1, D).contiguous()
        weight_c = weight.contiguous()

        mod = _get_module()
        dummy_bias = torch.empty(0, device=x.device, dtype=x.dtype)
        b_arg = bias.contiguous() if bias is not None else dummy_bias

        out_2d, x_hat = mod.fused_norm_linear_fwd(x_2d, weight_c, b_arg, eps)
        ctx.save_for_backward(x_2d, weight_c, x_hat)
        ctx.eps = eps
        ctx.has_bias = bias is not None
        ctx.orig_shape = orig_shape
        return out_2d.reshape(*orig_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        x_2d, weight, x_hat = ctx.saved_tensors
        dout_2d = dout.reshape(-1, weight.shape[0]).contiguous()
        mod = _get_module()

        dx_2d = mod.fused_norm_linear_bwd_dx(dout_2d, weight, x_2d, ctx.eps)
        dx = dx_2d.reshape(ctx.orig_shape)

        dweight = None
        dbias = None
        if weight.requires_grad:
            dweight = torch.matmul(dout_2d.t(), x_hat)
        if ctx.has_bias:
            dbias = dout_2d.sum(dim=0)

        return dx, dweight, dbias, None


def fused_norm_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused LayerNorm + Linear.

    WARNING: The custom CUDA kernel is 240× SLOWER than PyTorch reference
    (measured 2026-10-18 across 9 configurations: 106-827 ms vs 0.44-3.39 ms).
    Root cause: serial `for(k)` loop with uncoalesced `weight[n*K+k]` access
    and redundant `x_norm` recomputation per (m, n) pair.

    Routing: TRITON (Level 2) → PYTORCH (Level 3). CUDA kernel is EXPERIMENTAL
    and MUST NOT be dispatched until rewritten with shared-memory tiling.
    """
    triton_fn = _get_triton_impl()
    if triton_fn is not None:
        try:
            return triton_fn(x, weight, bias=bias, eps=eps)
        except (RuntimeError, MemoryError, AssertionError, ValueError) as exc:
            logger.warning("Triton execution failed for fused_norm_linear, falling back to PyTorch", exc_info=True)
    return fused_norm_linear_pytorch_ref(x, weight, bias=bias, eps=eps)
