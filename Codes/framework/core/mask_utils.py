"""
core/mask_utils.py — Unified mask handling for all attention and compression modules.
Handles 2D/4D, bool/float, additive/multiplicative masks consistently.
"""

from __future__ import annotations

import torch


def convert_mask_to_additive(
    mask: torch.Tensor | None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """Convert any mask format to additive form (0 = keep, -inf = mask out).

    Handles: 2D (B,L), 4D (B,H,L,L), bool masks, float masks.
    """
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        additive = torch.zeros_like(mask, dtype=dtype)
        additive[~mask] = float("-inf")
        return additive
    if mask.dim() == 2:
        if mask.min() < -100:
            return mask.to(dtype)
        additive = (1.0 - mask.to(dtype)) * -10000.0
        return additive
    if mask.dim() == 4:
        if mask.min() < -100:
            return mask.to(dtype)
        return (1.0 - mask.to(dtype)) * -10000.0
    return mask.to(dtype)


def convert_mask_to_bool(
    attention_mask: torch.Tensor | None,
    B: int | None = None,
    L: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Convert any attention_mask (2D/4D, bool/float/additive) to a boolean [B, L] mask where True=valid token."""
    if attention_mask is None:
        if B is not None and L is not None and device is not None:
            return torch.ones((B, L), dtype=torch.bool, device=device)
        return None
    if attention_mask.dtype == torch.bool:
        if attention_mask.dim() == 2:
            return attention_mask
        if attention_mask.dim() == 4:
            return attention_mask.squeeze(1).squeeze(1).bool()
    if attention_mask.min() < -100:
        bool_m = attention_mask > -5000
        if bool_m.dim() == 4:
            bool_m = bool_m.squeeze(1).squeeze(1)
        return bool_m
    if attention_mask.dim() == 4:
        return attention_mask.squeeze(1).squeeze(1) > 0.5
    return attention_mask > 0.5


def expand_mask_to_4d(
    mask: torch.Tensor | None,
    n_heads: int = 1,
    seq_len: int | None = None,
) -> torch.Tensor | None:
    """Expand a 2D attention mask to 4D (B, H, L, L) format."""
    if mask is None:
        return None
    if mask.dim() == 4:
        return mask
    if mask.dim() == 2:
        B, L = mask.shape
        L = seq_len or L
        extended = mask[:, None, None, :].expand(B, n_heads, L, L)
        return extended.to(dtype=torch.float32)
    return mask


def safe_attention_mask(
    mask: torch.Tensor | None,
    hidden_states: torch.Tensor,
    n_heads: int = 1,
) -> torch.Tensor | None:
    """Universal mask adapter for attention computation. Returns additive mask in float32."""
    if mask is None:
        return None
    B, L, _ = hidden_states.shape
    m = convert_mask_to_additive(mask)
    if m.dim() == 2:
        m = m[:, None, None, :].expand(B, n_heads, L, L)
    return m
