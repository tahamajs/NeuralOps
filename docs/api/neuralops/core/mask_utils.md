Module neuralops.core.mask_utils
================================
core/mask_utils.py — Unified mask handling for all attention and compression modules.
Handles 2D/4D, bool/float, additive/multiplicative masks consistently.

Functions
---------

`convert_mask_to_additive(mask: torch.Tensor | None, dtype: torch.dtype = torch.float32) ‑> torch.Tensor | None`
:   Convert any mask format to additive form (0 = keep, -inf = mask out).
    
    Handles: 2D (B,L), 4D (B,H,L,L), bool masks, float masks.

`convert_mask_to_bool(attention_mask: torch.Tensor | None, B: int | None = None, L: int | None = None, device: torch.device | None = None) ‑> torch.Tensor | None`
:   Convert any attention_mask (2D/4D, bool/float/additive) to a boolean [B, L] mask where True=valid token.

`expand_mask_to_4d(mask: torch.Tensor | None, n_heads: int = 1, seq_len: int | None = None) ‑> torch.Tensor | None`
:   Expand a 2D attention mask to 4D (B, H, L, L) format.

`safe_attention_mask(mask: torch.Tensor | None, hidden_states: torch.Tensor, n_heads: int = 1) ‑> torch.Tensor | None`
:   Universal mask adapter for attention computation. Returns additive mask in float32.