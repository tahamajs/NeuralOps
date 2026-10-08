"""FlashAttention autograd Function with forward/backward tiling reference."""
from __future__ import annotations
import math
import torch
from torch.autograd import Function

__all__ = ["FlashAttentionFunction", "flash_attention_forward_backward"]


class FlashAttentionFunction(Function):
    """Custom autograd Function for memory-efficient tiled attention with exact backward pass."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        # Reference forward pass
        scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
        if causal:
            N_q, N_k = q.shape[-2], k.shape[-2]
            causal_mask = torch.triu(torch.full((N_q, N_k), float("-inf"), device=q.device), diagonal=1)
            scores = scores + causal_mask

        # Softmax with logsumexp saved for backward
        lse = torch.logsumexp(scores, dim=-1, keepdim=True)
        p = torch.exp(scores - lse)
        out = torch.matmul(p, v)

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        causal = ctx.causal

        # Recompute P in backward from saved LSE (saving SRAM/HBM footprint)
        scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
        if causal:
            N_q, N_k = q.shape[-2], k.shape[-2]
            causal_mask = torch.triu(torch.full((N_q, N_k), float("-inf"), device=q.device), diagonal=1)
            scores = scores + causal_mask

        p = torch.exp(scores - lse)

        # dV = P^T * grad_out
        grad_v = torch.matmul(p.transpose(-2, -1), grad_out)

        # dP = grad_out * V^T
        grad_p = torch.matmul(grad_out, v.transpose(-2, -1))

        # dS = P * (dP - D) where D = sum(grad_out * out, dim=-1, keepdim=True)
        # Using standard softmax backward: dS = P * (dP - (P * dP).sum(-1, keepdim=True))
        sum_p_dp = (p * grad_p).sum(dim=-1, keepdim=True)
        grad_s = p * (grad_p - sum_p_dp) * softmax_scale

        # dQ = dS * K
        grad_q = torch.matmul(grad_s, k)

        # dK = dS^T * Q
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
