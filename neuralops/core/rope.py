"""
Shared Rotary Position Embedding (RoPE) primitives.
Hardened for production across transformer architectures (DistilBERT, RoBERTa, GPT-2, Qwen, LLaMA).
Supports odd head dimensions, partial rotary dimensions, and broadcasted position embeddings.
"""

from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor safely handling odd dims:
    [-x2, x1, (x_odd)].
    """
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    rotated = torch.cat((-x2, x1), dim=-1)
    if 2 * half < d:
        return torch.cat((rotated, x[..., 2 * half :]), dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding (RoPE) with partial dimension and independent seq-len support.
    """
    if cos is None or sin is None:
        return q, k

    rotary_dim = cos.shape[-1]
    safe_rotary_dim = (rotary_dim // 2) * 2
    if safe_rotary_dim == 0:
        return q, k

    head_dim = q.shape[-1]
    if safe_rotary_dim > head_dim:
        safe_rotary_dim = (head_dim // 2) * 2

    q_rot, q_pass = q[..., :safe_rotary_dim], q[..., safe_rotary_dim:]
    k_rot, k_pass = k[..., :safe_rotary_dim], k[..., safe_rotary_dim:]

    cos_safe = cos[..., :safe_rotary_dim]
    sin_safe = sin[..., :safe_rotary_dim]

    if cos_safe.dim() == 2 and cos_safe.shape[0] == q_rot.shape[-2]:
        cos_safe = cos_safe.unsqueeze(0).unsqueeze(1)
        sin_safe = sin_safe.unsqueeze(0).unsqueeze(1)
    else:
        while cos_safe.dim() < q_rot.dim():
            cos_safe = cos_safe.unsqueeze(unsqueeze_dim)
            sin_safe = sin_safe.unsqueeze(unsqueeze_dim)

    cos_safe = cos_safe.to(device=q_rot.device, dtype=q_rot.dtype)
    sin_safe = sin_safe.to(device=q_rot.device, dtype=q_rot.dtype)

    q_rot = (q_rot * cos_safe) + (_rotate_half(q_rot) * sin_safe)
    k_rot = (k_rot * cos_safe) + (_rotate_half(k_rot) * sin_safe)

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot
    return q, k


__all__ = ["_rotate_half", "apply_rotary_pos_emb"]
