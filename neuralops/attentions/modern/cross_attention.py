"""Cross attention implementation."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

__all__ = ["CrossAttention"]


@ATTENTION_REGISTRY.register("cross_attention")
class CrossAttention(BaseAttention):
    """Multi-head cross attention between query sequence and encoder/context sequence."""

    supports_causal: bool = False
    supports_gqa: bool = True
    supports_rope: bool = False

    def __init__(
        self,
        config: object | None = None,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(config)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_p = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for cross attention.

        Args:
            hidden_states: Query sequence tensor of shape (B, N_q, D).
            context: Key/Value context sequence tensor of shape (B, N_kv, D).
                     If None, self-attention on hidden_states is used.
            attention_mask: Mask of shape (B, 1, N_q, N_kv) or broadcastable.

        Returns:
            Tuple of (output tensor of shape (B, N_q, D), None or attention weights).
        """
        B, N_q, D = hidden_states.shape
        if context is None:
            context = hidden_states
        B_kv, N_kv, _ = context.shape

        q = self.q_proj(hidden_states).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B_kv, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B_kv, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout_p > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        out = self.out_proj(out)
        return out, None
