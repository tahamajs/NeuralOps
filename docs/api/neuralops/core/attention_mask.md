Module neuralops.core.attention_mask
====================================
Universal attention mask handler for standard and efficient attention modules.

Supports:
- 2D binary masks [Batch, SeqLen] (1 for valid tokens, 0 for padding)
- 3D masks [Batch, 1, SeqLen] or [Batch, SeqLen, SeqLen]
- 4D extended additive masks [Batch, 1, 1, SeqLen] or [Batch, 1, SeqLen, SeqLen] (where 0.0 is unmasked, -10000.0 is masked)
- Boolean masks [Batch, SeqLen]

Functions
---------

`apply_attention_mask(scores: torch.Tensor, attention_mask: torch.Tensor | None, mask_value: float = -10000.0) ‑> torch.Tensor`
:   Apply attention mask to attention logits in a numerically robust and universal manner.
    
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

`create_band_mask(seq_len: int, window_size: int, device: torch.device | None = None) ‑> torch.Tensor`
:   Create a sliding window band attention mask [SeqLen, SeqLen] where positions |i - j| <= window_size are True.

`create_causal_mask(seq_len: int, device: torch.device | None = None) ‑> torch.Tensor`
:   Create a lower-triangular causal attention mask [SeqLen, SeqLen] where True = attend, False = mask.

`get_token_valid_mask(attention_mask: torch.Tensor | None, batch_size: int, seq_len: int, device: torch.device) ‑> torch.Tensor`
:   Extract a 1D boolean validity mask [Batch, SeqLen] where True = valid token, False = pad.
    
    Args:
        attention_mask (Optional[torch.Tensor]): Mask in any format.
        batch_size (int): Batch size B.
        seq_len (int): Sequence length N.
        device (torch.device): Device to place the mask on.
    
    Returns:
        torch.Tensor: Boolean tensor of shape `[Batch, SeqLen]`.