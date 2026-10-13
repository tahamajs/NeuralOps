"""Rotary Position Embedding (RoPE) shared utilities."""
from __future__ import annotations

import torch

__all__ = ["rotate_half", "_rotate_half", "apply_rotary_pos_emb", "broadcast_cos_sin"]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    d = x.shape[-1]
    if d % 2 == 1:
        half = d // 2
        x1 = x[..., :half]
        x2 = x[..., half : 2 * half]
        tail = x[..., 2 * half :]
        return torch.cat((-x2, x1, tail), dim=-1)
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


_rotate_half = rotate_half


def broadcast_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast cos and sin to match q's dimensions."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    past_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to query and key tensors."""
    cos, sin = broadcast_cos_sin(cos, sin, q)
    q_len = q.size(-2)
    k_len = k.size(-2)
    cos_q = cos[..., past_len : past_len + q_len, :]
    sin_q = sin[..., past_len : past_len + q_len, :]
    cos_k = cos[..., past_len : past_len + k_len, :]
    sin_k = sin[..., past_len : past_len + k_len, :]
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed
