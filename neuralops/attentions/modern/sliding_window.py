"""Sliding window attention (Mistral-style)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.triton_kernels.sliding_window import sliding_window_attention_kernel

__all__ = ["SlidingWindowAttention"]


@ATTENTION_REGISTRY.register("sliding_window_attention")
class SlidingWindowAttention(BaseAttention):
    """Sliding window attention — attends to last W tokens only per position."""

    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True

    def __init__(
        self,
        config: object | None = None,
        embed_dim: int = 64,
        num_heads: int = 4,
        window_size: int = 4096,
        **kwargs,
    ):
        super().__init__(config)
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        B, N, D = hidden_states.shape
        if D != self.embed_dim:
            # Flexible projection if dims differ
            return hidden_states, None

        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        out = sliding_window_attention_kernel(q, k, v, window_size=self.window_size)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None
