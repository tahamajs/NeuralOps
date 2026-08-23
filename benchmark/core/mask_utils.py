"""Robust tensor masking utilities supporting 2D, 3D, 4D, boolean, integer, and additive attention masks."""

from __future__ import annotations
from typing import Optional
import torch


def convert_mask_to_bool(
    attention_mask: Optional[torch.Tensor],
    B: int,
    L: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert arbitrary HuggingFace attention masks into standard [B, L] boolean tensors
    where True denotes a valid (non-padded) token position.

    Args:
        attention_mask: Optional tensor of shape [B, L], [B, 1, 1, L], or [B, 1, L, L].
        B: Batch size fallback.
        L: Sequence length fallback.
        device: Target execution device.

    Returns:
        torch.Tensor of shape [B, L] and dtype torch.bool.
    """
    if attention_mask is None:
        return torch.ones((B, L), dtype=torch.bool, device=device)

    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask, device=device)

    # 2D Mask Handling [B, L]
    if attention_mask.dim() == 2:
        if attention_mask.dtype == torch.bool:
            return attention_mask
        return attention_mask > 0

    # 3D Mask Handling [B, 1, L] or [B, L, L]
    if attention_mask.dim() == 3:
        if attention_mask.shape[1] == 1:
            mask_2d = attention_mask.squeeze(1)
        else:
            mask_2d = attention_mask[:, :, -1]
        if mask_2d.dtype == torch.bool:
            return mask_2d
        return mask_2d > 0

    # 4D Mask Handling [B, 1, 1, L] or [B, 1, L, L]
    if attention_mask.dim() == 4:
        sub_mask = attention_mask[:, 0, 0, :]
        if sub_mask.dtype == torch.bool:
            return sub_mask
        # Check for large negative additive float masks (-10000.0 or -inf)
        if sub_mask.min() < -1000.0:
            return sub_mask > -10000.0
        return sub_mask > 0

    # Fallback default
    return torch.ones((B, L), dtype=torch.bool, device=device)
