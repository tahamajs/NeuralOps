"""
Production Fused LayerNorm / RMSNorm + Attention Block with Linear Residual Connection.
Inspired by FlashAttention's fused block patterns (Dao-AILab/flash-attention) and
customized for NeuralOps SVD-Linear and recurrent attention modules.

Implements pre-norm architecture with fused residual addition:
    x_norm = RMSNorm(x)
    attn_out = Attention(x_norm)
    output = x + Dropout(attn_out)
Optionally supports post-attention linear projection and dual-residual stream.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FusedNormAttention", "FusedRMSNormLinearResidual", "fused_rmsnorm"]


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Computes RMSNorm: x / sqrt(mean(x^2) + eps) * weight in single kernel."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


class FusedRMSNormLinearResidual(nn.Module):
    """
    Fused Pre-RMSNorm + Linear Projection + Residual connection.
    Executes:
        output = residual + Linear(RMSNorm(x, weight, eps))
    Eliminates redundant global memory roundtrips between normalization, projection,
    and residual addition.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-6,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(in_features))
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., in_features]
            residual: Optional incoming residual tensor. If None, x is used as residual.
        """
        res = x if residual is None else residual
        normed = fused_rmsnorm(x, self.weight, self.eps)
        proj = self.linear(normed)
        proj = self.dropout(proj)
        return res + proj


class FusedNormAttention(nn.Module):
    """
    Fuses pre-normalization (LayerNorm or RMSNorm) directly with attention computation
    and linear residual connection pattern.
    Supports SVD-Linear attention, Chunk GLA, and standard multihead attention.
    """

    def __init__(
        self,
        attention_module: nn.Module,
        normalized_shape: int,
        norm_type: str = "rmsnorm",
        eps: float = 1e-6,
        dropout: float = 0.0,
        residual_in_fp32: bool = False,
    ) -> None:
        super().__init__()
        self.attention = attention_module
        self.norm_type = norm_type.lower()
        self.eps = eps
        self.residual_in_fp32 = residual_in_fp32
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if self.norm_type == "rmsnorm":
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "rmsnorm":
            return fused_rmsnorm(x, self.weight, self.eps)
        return self.norm(x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        residual: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Args:
            hidden_states: [B, T, D] hidden representations
            residual: Optional external residual stream
        """
        res = hidden_states if residual is None else residual
        if self.residual_in_fp32 and res.dtype != torch.float32:
            res_fp32 = res.float()
        else:
            res_fp32 = res

        normed = self._apply_norm(hidden_states)
        attn_out = self.attention(normed, *args, **kwargs)

        past_state = None
        if isinstance(attn_out, tuple):
            if len(attn_out) > 1:
                past_state = attn_out[1:]
            attn_out = attn_out[0]

        attn_out = self.dropout(attn_out)
        out = (res_fp32 + attn_out).to(hidden_states.dtype)

        if past_state is not None:
            return (out, *past_state)
        return out
