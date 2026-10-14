"""
Mask and Position Processing Utilities for Left and Right Padding Invariance.

Provides unified position ID computation and key-value state zeroing for
linear attention, state-space models, and recurrent linear transformers.
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple
from neuralops.core.attention_mask import get_token_valid_mask


def prepare_mask_and_positions(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes cumulative mask-aware position IDs and zero-out masks.

    For left-padded sequences:
        tokens: [PAD, PAD, A, B]
        mask:   [  0,   0, 1, 1]
    The position IDs for valid tokens must be [0, 0, 0, 1], NOT [0, 1, 2, 3]!
    This ensures that positional embeddings (RoPE, sinusoidal, cos/sin) for the
    actual tokens A and B are identical regardless of whether they are left-padded,
    right-padded, or unpadded.

    Returns:
        valid_mask: [B, N] boolean tensor (True for real tokens, False for padding).
        pos_ids: [B, N] int64 tensor with 0-indexed positions for valid tokens.
        clean_hidden: [B, N, D] tensor with padded token embeddings zeroed out.
    """
    B, N, D = hidden_states.shape
    device = hidden_states.device

    valid_mask = get_token_valid_mask(attention_mask, B, N, device)  # [B, N] bool

    if position_ids is None:
        # Cumulative sum over valid tokens:
        # e.g., mask = [0, 0, 1, 1] -> cumsum = [0, 0, 1, 2] -> pos_ids = [0, 0, 0, 1]
        cumsum = torch.cumsum(valid_mask.long(), dim=-1)
        pos_ids = torch.clamp(cumsum - 1, min=0)
    else:
        pos_ids = position_ids

    # Zero out padded activations so they do not leak into cumulative sums or state updates
    mask_expanded = valid_mask.unsqueeze(-1).to(hidden_states.dtype)
    clean_hidden = hidden_states * mask_expanded

    return valid_mask, pos_ids, clean_hidden


def mask_kv_states(
    K: torch.Tensor,
    V: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zeroes out Key and Value states at padding positions before recurrent/cumulative accumulation.
    Handles K, V shapes of [B, H, N, d] or [B, N, H, d].
    """
    if valid_mask is None:
        return K, V

    if K.dim() == 4:
        if K.shape[1] == valid_mask.shape[1]:  # [B, N, H, d]
            m = valid_mask.unsqueeze(-1).unsqueeze(-1).to(K.dtype)
        else:  # [B, H, N, d]
            m = valid_mask.unsqueeze(1).unsqueeze(-1).to(K.dtype)
        return K * m, V * m

    elif K.dim() == 3:  # [B, N, D]
        m = valid_mask.unsqueeze(-1).to(K.dtype)
        return K * m, V * m

    return K, V


__all__ = [
    "prepare_mask_and_positions",
    "mask_kv_states",
]
