"""
Shared Grouped-Query Attention (GQA) & Multi-Head Attention (MHA) Utilities.
Provides dynamic dimension extraction, head repetition, and reshaping
for Qwen3.5, LLaMA, Mistral, Gemma, and legacy MHA models.
"""

from __future__ import annotations

from typing import Any

import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands key/value heads from num_key_value_heads to num_attention_heads.
    Equivalent to torch.repeat_interleave(hidden_states, dim=1, repeats=n_rep)
    Shape: (batch, num_key_value_heads, seq_len, head_dim) -> (batch, num_attention_heads, seq_len, head_dim)
    For standard MHA where n_rep <= 1, returns hidden_states directly (zero-copy).
    """
    if n_rep <= 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def get_qkv_projection_dims(adapter: Any) -> tuple[int, int, int, int]:
    """
    Extracts (q_proj_out, kv_proj_out, head_dim, kv_repeat) dynamically from adapter or config.
    Supports dictionary adapters, HF PretrainedConfig, or object adapters.
    Returns:
        q_out: output dimension for query projection (e.g. 2048 for Qwen3.5-0.8B)
        kv_out: output dimension for key/value projection (e.g. 512 for Qwen3.5-0.8B)
        head_dim: dimension of each head (e.g. 256 for Qwen3.5-0.8B)
        n_rep: repetition factor kv_repeat = num_heads // num_key_value_heads (e.g. 4 for Qwen3.5-0.8B)
    """
    if isinstance(adapter, dict):
        hidden_size = int(adapter.get("hidden_size", 1024))
        num_heads = int(adapter.get("num_heads", 8))
        num_kv_heads = int(adapter.get("num_key_value_heads", num_heads))
        head_dim = int(adapter.get("head_dim", hidden_size // num_heads))
        q_out = int(adapter.get("q_proj_out", num_heads * head_dim))
        kv_out = int(adapter.get("kv_proj_out", num_kv_heads * head_dim))
        n_rep = int(adapter.get("kv_repeat", max(1, num_heads // max(1, num_kv_heads))))
    else:
        hidden_size = int(getattr(adapter, "hidden_size", 1024))
        num_heads = int(
            getattr(adapter, "num_attention_heads", getattr(adapter, "num_heads", 8))
        )
        num_kv_heads = int(getattr(adapter, "num_key_value_heads", num_heads))
        head_dim = int(getattr(adapter, "head_dim", hidden_size // num_heads))
        q_out = int(getattr(adapter, "q_proj_out", num_heads * head_dim))
        kv_out = int(getattr(adapter, "kv_proj_out", num_kv_heads * head_dim))
        n_rep = int(
            getattr(adapter, "kv_repeat", max(1, num_heads // max(1, num_kv_heads)))
        )
    return q_out, kv_out, head_dim, n_rep


def reshape_for_attention(
    x: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """
    Reshapes projected tensor (B, N, num_heads * head_dim) -> (B, num_heads, N, head_dim)
    Safe for both MHA and GQA query / key / value representations.
    """
    B, N, _ = x.shape
    return x.view(B, N, num_heads, head_dim).transpose(1, 2).contiguous()
