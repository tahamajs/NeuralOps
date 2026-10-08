"""Multi-Query Attention (MQA) implementation."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

__all__ = ["MultiQueryAttention"]


@ATTENTION_REGISTRY.register("mqa_attention")
class MultiQueryAttention(BaseAttention):
    """Multi-Query Attention — multiple query heads share single key and value heads."""

    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True

    def __init__(
        self,
        config: object | None = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
        **kwargs,
    ):
        super().__init__(config)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_p = dropout
        self.causal = causal

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for MultiQueryAttention.

        Args:
            hidden_states: Input tensor of shape (B, N, D).
            attention_mask: Attention mask tensor.

        Returns:
            Tuple of (output tensor of shape (B, N, D), None).
        """
        B, N, D = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        k = self.k_proj(hidden_states).view(B, N, 1, self.head_dim).transpose(1, 2)              # (B, 1, N, d)
        v = self.v_proj(hidden_states).view(B, N, 1, self.head_dim).transpose(1, 2)              # (B, 1, N, d)

        # Broadcast k and v across query heads
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        if self.causal:
            causal_mask = torch.triu(torch.full((N, N), float("-inf"), device=hidden_states.device), diagonal=1)
            scores = scores + causal_mask

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout_p > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        out = torch.matmul(attn_weights, v)  # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out, None
