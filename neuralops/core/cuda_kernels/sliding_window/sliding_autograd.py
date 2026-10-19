#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Sliding Window Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

logger = logging.getLogger(__name__)

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "sliding_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "sliding_window.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_sliding_window",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["sliding_window_attn_fwd"],
        )
    return _MODULE


def sliding_window_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 128,
    num_sinks: int = 4,
    scale: float | None = None,
) -> torch.Tensor:
    """Pure PyTorch reference with FP32 stability."""
    B, H, N, D = q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale

    q_idx = torch.arange(N, device=q.device).unsqueeze(1)
    k_idx = torch.arange(N, device=q.device).unsqueeze(0)

    causal = k_idx <= q_idx
    in_window = k_idx >= (q_idx - window_size + 1)
    is_sink = k_idx < num_sinks
    valid_mask = causal & (in_window | is_sink)
    mask = valid_mask.unsqueeze(0).unsqueeze(0).expand(B, H, N, N)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return torch.nan_to_num(out, nan=0.0)


def _sliding_window_triton_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 128,
    num_sinks: int = 4,
    scale: float | None = None,
) -> torch.Tensor | None:
    if num_sinks > 0:
        return None

    try:
        from neuralops.core.triton_kernels.sliding_window import sliding_window_triton
        return sliding_window_triton(q, k, v, window_size=window_size)
    except (RuntimeError, MemoryError, AssertionError, ValueError, ImportError, AttributeError) as exc:
        logger.warning("Failed in Triton fallback for sliding_window: %s", exc, exc_info=True)
        return None


def _sliding_window_forward_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 128,
    num_sinks: int = 4,
    scale: float | None = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    mod = _get_module()
    cuda_fn = mod.sliding_window_attn_fwd if mod is not None else None

    return dispatch(
        cuda_fn,
        _sliding_window_triton_fallback,
        sliding_window_pytorch_ref,
        q,
        k,
        v,
        window_size,
        num_sinks,
        scale,
        kernel_name="sliding_window_attn",
    )


class _SlidingWindowAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window_size: int, num_sinks: int, scale: float):
        out = _sliding_window_forward_dispatch(q, k, v, window_size, num_sinks, scale)
        ctx.save_for_backward(q, k, v)
        ctx.window_size = window_size
        ctx.num_sinks = num_sinks
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        q, k, v = ctx.saved_tensors
        window_size = ctx.window_size
        num_sinks = ctx.num_sinks
        scale = ctx.scale

        B, H, N, D = q.shape
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale

        q_idx = torch.arange(N, device=q.device).unsqueeze(1)
        k_idx = torch.arange(N, device=q.device).unsqueeze(0)

        causal = k_idx <= q_idx
        in_window = k_idx >= (q_idx - window_size + 1)
        is_sink = k_idx < num_sinks

        valid_mask = causal & (in_window | is_sink)
        scores = scores.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        p = F.softmax(scores, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)

        grad_out_f = grad_output.float()
        grad_v = torch.matmul(p.transpose(-1, -2), grad_out_f).to(v.dtype)

        grad_p = torch.matmul(grad_out_f, v.float().transpose(-1, -2))
        sum_p_grad_p = (p * grad_p).sum(dim=-1, keepdim=True)
        grad_scores = p * (grad_p - sum_p_grad_p) * scale

        grad_q = torch.matmul(grad_scores, k.float()).to(q.dtype)
        grad_k = torch.matmul(grad_scores.transpose(-1, -2), q.float()).to(k.dtype)

        return grad_q, grad_k, grad_v, None, None, None


def sliding_window_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 128,
    num_sinks: int = 4,
    scale: float | None = None,
) -> torch.Tensor:
    """Sliding Window Attention with autograd backward pass and 3-level fallback."""
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    if any(t.requires_grad for t in (q, k, v) if isinstance(t, torch.Tensor)):
        return _SlidingWindowAttnFunction.apply(q, k, v, window_size, num_sinks, scale)

    return _sliding_window_forward_dispatch(q, k, v, window_size, num_sinks, scale)


sliding_window = sliding_window_attn

__all__ = ["sliding_window_attn", "sliding_window", "sliding_window_pytorch_ref"]
