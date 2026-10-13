"""Grouped Query Attention (GQA) shared helpers."""
from __future__ import annotations

import torch

__all__ = ["repeat_kv"]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads for Grouped Query Attention (GQA).
    
    hidden_states: (batch, num_key_value_heads, seqlen, head_dim)
    n_rep: number of repetitions
    """
    if n_rep == 1:
        return hidden_states
    b, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(b, num_kv_heads * n_rep, slen, head_dim)
