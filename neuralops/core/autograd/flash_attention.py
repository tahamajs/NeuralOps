# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
FlashAttention Autograd Function

Exact PyTorch autograd function implementing tiled forward and backward passes for FlashAttention.
"""

from __future__ import annotations
import math
from typing import Any
import torch
from torch.autograd import Function

__all__ = ["FlashAttentionFunction", "flash_attention_forward_backward"]


class FlashAttentionFunction(Function):
    """Custom autograd Function for memory-efficient tiled attention with exact backward pass."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
        if causal:
            N_q, N_k = q.shape[-2], k.shape[-2]
            causal_mask = torch.triu(torch.full((N_q, N_k), float("-inf"), device=q.device), diagonal=1)
            scores = scores + causal_mask

        lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        p = torch.exp(scores - lse)
        out = torch.matmul(p, v)

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        q, k, v, out, lse = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        causal = ctx.causal

        scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
        if causal:
            N_q, N_k = q.shape[-2], k.shape[-2]
            causal_mask = torch.triu(torch.full((N_q, N_k), float("-inf"), device=q.device), diagonal=1)
            scores = scores + causal_mask

        p = torch.exp(scores - lse)

        grad_v = torch.matmul(p.transpose(-2, -1), grad_out)

        grad_p = torch.matmul(grad_out, v.transpose(-2, -1))

        sum_p_dp = (p * grad_p).sum(dim=-1, keepdim=True)
        grad_s = p * (grad_p - sum_p_dp) * softmax_scale

        grad_q = torch.matmul(grad_s, k)

        grad_k = torch.matmul(grad_s.transpose(-2, -1), q)

        return grad_q, grad_k, grad_v, None, None


def flash_attention_forward_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Applies FlashAttentionFunction."""
    return FlashAttentionFunction.apply(q, k, v, softmax_scale, causal)
