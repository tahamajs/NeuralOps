"""Fused LayerNorm / RMSNorm + Attention block."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn

__all__ = ["FusedNormAttention"]


class FusedNormAttention(nn.Module):
    """Fuses pre-normalization (LayerNorm or RMSNorm) directly with attention computation and residual connection."""

    def __init__(
        self,
        attention_module: nn.Module,
        normalized_shape: int,
        norm_type: str = "rmsnorm",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attention = attention_module
        self.norm_type = norm_type.lower()
        if self.norm_type == "rmsnorm":
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.eps = eps
        else:
            self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "rmsnorm":
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + self.eps) * self.weight
        return self.norm(x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        normed = self._apply_norm(hidden_states)
        attn_out = self.attention(normed, *args, **kwargs)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        return hidden_states + attn_out
