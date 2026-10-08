"""Fused attention backward pass reference implementation."""
from __future__ import annotations
import math
import torch

__all__ = ["fused_attention_backward"]


def fused_attention_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused backward implementation for multi-head attention.

    Computes gradients w.r.t Q, K, V without keeping full N x N attention matrix
    in memory by computing row-wise softmax intermediate reductions.

    Args:
        grad_out: Gradient of output (B, H, N_q, D).
        q: Query tensor (B, H, N_q, D).
        k: Key tensor (B, H, N_k, D).
        v: Value tensor (B, H, N_k, D).
        out: Forward output tensor (B, H, N_q, D).
        softmax_scale: Scaling factor for attention logits.
        causal: Whether causal mask is applied.

    Returns:
        (grad_q, grad_k, grad_v) tensors matching shapes of q, k, v.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    if causal:
        N_q, N_k = q.shape[-2], k.shape[-2]
        causal_mask = torch.triu(torch.full((N_q, N_k), float("-inf"), device=q.device), diagonal=1)
        scores = scores + causal_mask

    # P = softmax(scores)
    p = torch.softmax(scores, dim=-1)

    # dV = P^T @ dO
    grad_v = torch.matmul(p.transpose(-2, -1), grad_out)

    # dP = dO @ V^T
    grad_p = torch.matmul(grad_out, v.transpose(-2, -1))

    # dS = P * (dP - sum(dP * P))
    # Note: sum(dP * P) = sum(dO @ V^T * P) = sum(dO * (P @ V)) = sum(dO * out)
    di = (grad_out * out).sum(dim=-1, keepdim=True)
    grad_s = p * (grad_p - di) * softmax_scale

    # dQ = dS @ K
    grad_q = torch.matmul(grad_s, k)

    # dK = dS^T @ Q
    grad_k = torch.matmul(grad_s.transpose(-2, -1), q)

    return grad_q, grad_k, grad_v
