"""
Universal attention mask handler for standard and efficient attention modules.

Supports:
- 2D binary masks [Batch, SeqLen] (1 for valid tokens, 0 for padding)
- 3D masks [Batch, 1, SeqLen] or [Batch, SeqLen, SeqLen]
- 4D extended additive masks [Batch, 1, 1, SeqLen] or [Batch, 1, SeqLen, SeqLen] (where 0.0 is unmasked, -10000.0 is masked)
- Boolean masks [Batch, SeqLen]
"""

import torch


def apply_attention_mask(
    scores: torch.Tensor,
    attention_mask: torch.Tensor | None,
    mask_value: float = -10000.0,
) -> torch.Tensor:
    r"""Apply attention mask to attention logits in a numerically robust and universal manner.

    Args:
        scores (torch.Tensor): Attention logits of shape `[Batch, Heads, SeqLen_Q, SeqLen_K]`.
        attention_mask (Optional[torch.Tensor]): Mask tensor in arbitrary standard format:
            - None: No-op, returns `scores` directly.
            - 2D `[B, N_k]`: Binary (1=keep, 0=mask) or boolean.
            - 3D `[B, 1, N_k]` or `[B, N_q, N_k]`.
            - 4D `[B, 1, 1, N_k]` or `[B, 1, N_q, N_k]`: Pre-computed additive float mask or binary.
        mask_value (float): Large negative value to assign to masked positions.

    Returns:
        torch.Tensor: Masked attention logits of same shape and dtype as `scores`.
    """
    if attention_mask is None:
        return scores

    if not isinstance(attention_mask, torch.Tensor):
        return scores

    dtype = scores.dtype
    device = scores.device

    # Ensure mask is on the same device
    mask = attention_mask.to(device=device)

    # Case 1: 2D binary / boolean mask [B, N_k]
    if mask.dim() == 2:
        if mask.dtype == torch.bool:
            # True = keep, False = mask
            additive = torch.zeros_like(mask, dtype=dtype)
            additive = additive.masked_fill(~mask, mask_value)
        elif torch.is_floating_point(mask):
            # Check if it's already negative additive or 0/1 binary
            if (mask < 0).any():
                # Already additive
                additive = mask.to(dtype=dtype)
            else:
                # 1.0 = keep, 0.0 = mask
                additive = (1.0 - mask.to(dtype=dtype)) * mask_value
        else:
            # Integer binary mask (1=keep, 0=mask)
            additive = (1.0 - mask.to(dtype=dtype)) * mask_value
        # Broadcast to [B, 1, 1, N_k]
        return scores + additive.unsqueeze(1).unsqueeze(2)

    # Case 2: 3D mask [B, 1, N_k] or [B, N_q, N_k]
    elif mask.dim() == 3:
        if mask.dtype == torch.bool:
            additive = torch.zeros_like(mask, dtype=dtype)
            additive = additive.masked_fill(~mask, mask_value)
            return scores + additive.unsqueeze(1)
        elif torch.is_floating_point(mask) and (mask < 0).any():
            return scores + mask.unsqueeze(1).to(dtype=dtype)
        else:
            additive = (1.0 - mask.to(dtype=dtype)) * mask_value
            return scores + additive.unsqueeze(1)

    # Case 3: 4D mask [B, 1, 1, N_k] or [B, 1, N_q, N_k] or [B, H, N_q, N_k]
    elif mask.dim() == 4:
        if mask.dtype == torch.bool:
            additive = torch.zeros_like(mask, dtype=dtype)
            additive = additive.masked_fill(~mask, mask_value)
            return scores + additive
        elif torch.is_floating_point(mask):
            # HuggingFace extended attention mask already contains 0.0 for unmasked and -10000.0 for masked
            if (mask < 0).any():
                return scores + mask.to(dtype=dtype)
            else:
                # 4D binary mask
                return scores + (1.0 - mask.to(dtype=dtype)) * mask_value
        else:
            # 4D integer mask
            return scores + (1.0 - mask.to(dtype=dtype)) * mask_value

    return scores


def get_token_valid_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    r"""Extract a 1D boolean validity mask [Batch, SeqLen] where True = valid token, False = pad.

    Args:
        attention_mask (Optional[torch.Tensor]): Mask in any format.
        batch_size (int): Batch size B.
        seq_len (int): Sequence length N.
        device (torch.device): Device to place the mask on.

    Returns:
        torch.Tensor: Boolean tensor of shape `[Batch, SeqLen]`.
    """
    if attention_mask is None:
        return torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)

    mask = attention_mask.to(device=device)

    if mask.dim() == 2:
        if mask.dtype == torch.bool:
            return mask
        elif torch.is_floating_point(mask) and (mask < 0).any():
            return mask >= -1.0  # unmasked is ~0.0
        else:
            return mask > 0.5
    elif mask.dim() == 4:
        # [B, 1, 1, N] or [B, 1, N, N]
        if mask.size(2) == 1:
            last_dim_mask = mask[:, 0, 0, :]
            if last_dim_mask.dtype == torch.bool:
                return last_dim_mask
            elif torch.is_floating_point(last_dim_mask) and (last_dim_mask < 0).any():
                return last_dim_mask >= -1.0
            else:
                return last_dim_mask > 0.5
        elif mask.size(2) == seq_len:
            # 4D causal or 2D attention mask [B, 1, N, N]
            # Do NOT use mask[:, 0, 0, :] as row 0 only unmasks token 0 in causal masks!
            if mask.dtype == torch.bool:
                return mask.any(dim=2)[:, 0, :]
            elif torch.is_floating_point(mask) and (mask < 0).any():
                return (mask >= -1000.0).any(dim=2)[:, 0, :]
            else:
                return (mask > 0.5).any(dim=2)[:, 0, :]
    return torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    r"""Create a lower-triangular causal attention mask [SeqLen, SeqLen] where True = attend, False = mask."""
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    return mask


def create_band_mask(
    seq_len: int, window_size: int, device: torch.device | None = None
) -> torch.Tensor:
    r"""Create a sliding window band attention mask [SeqLen, SeqLen] where positions |i - j| <= window_size are True."""
    idx = torch.arange(seq_len, device=device)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return diff <= window_size
