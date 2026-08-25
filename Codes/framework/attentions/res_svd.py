r"""
Spectral Residual Low-Rank SVD Attention (Res-SVD Attention).

Combines base truncated SVD low-rank factorizations with trainable residual adapters:
$$W \approx U_{\text{base}} V_{\text{base}}^\top + \alpha \cdot B A$$
where $B A$ is initialized directly from the top-$k$ residual singular directions of $W - W_{\text{base}}$.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import apply_attention_mask
from framework.core.registry import ATTENTION_REGISTRY
from framework.compression_engine.res_svd import ResSVDLinear


@ATTENTION_REGISTRY.register("res_svd")
@ATTENTION_REGISTRY.register("spectral_residual_svd")
class ResSVDAttention(nn.Module):
    r"""
    Multi-Head Attention module using Spectral Residual SVD (Res-SVD) linear projections.
    """

    def __init__(
        self,
        config: Any,
        base_rank: Optional[int] = None,
        residual_rank: Optional[int] = None,
        rank: Optional[int] = 16,
        scaling: float = 1.0,
        factorize_base: bool = True,
        freeze_base: bool = False,
        dropout: Optional[float] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads: int = getattr(config, "num_attention_heads", getattr(config, "num_heads", 12))
        self.hidden_size: int = getattr(config, "hidden_size", getattr(config, "d_model", 768))
        self.head_dim: int = self.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim

        effective_rank = rank if rank is not None else 16
        self.base_rank: int = base_rank if base_rank is not None else effective_rank
        self.residual_rank: int = residual_rank if residual_rank is not None else max(4, effective_rank // 4)
        self.scaling = float(scaling)

        self.query = ResSVDLinear(
            in_features=self.hidden_size,
            out_features=self.all_head_size,
            base_rank=self.base_rank,
            residual_rank=self.residual_rank,
            scaling=self.scaling,
            factorize_base=factorize_base,
            bias=True,
        )
        self.key = ResSVDLinear(
            in_features=self.hidden_size,
            out_features=self.all_head_size,
            base_rank=self.base_rank,
            residual_rank=self.residual_rank,
            scaling=self.scaling,
            factorize_base=factorize_base,
            bias=True,
        )
        self.value = ResSVDLinear(
            in_features=self.hidden_size,
            out_features=self.all_head_size,
            base_rank=self.base_rank,
            residual_rank=self.residual_rank,
            scaling=self.scaling,
            factorize_base=factorize_base,
            bias=True,
        )
        self.out_proj = ResSVDLinear(
            in_features=self.all_head_size,
            out_features=self.hidden_size,
            base_rank=self.base_rank,
            residual_rank=self.residual_rank,
            scaling=self.scaling,
            factorize_base=factorize_base,
            bias=True,
        )

        dropout_p = (
            dropout
            if dropout is not None
            else getattr(
                config,
                "dropout",
                getattr(
                    config,
                    "hidden_dropout_prob",
                    getattr(config, "attention_probs_dropout_prob", 0.1),
                ),
            )
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        """Decompose and initialize weights from source module."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                temp_linear = nn.Linear(target.in_features, target.out_features, bias=b is not None)
                with torch.no_grad():
                    temp_linear.weight.copy_(w)
                    if b is not None and temp_linear.bias is not None:
                        temp_linear.bias.copy_(b)
                new_layer = ResSVDLinear.from_linear(
                    temp_linear,
                    base_rank=self.base_rank,
                    residual_rank=self.residual_rank,
                    scaling=self.scaling,
                    factorize_base=True,
                    freeze_base=False,
                )
                target.load_state_dict(new_layer.state_dict())
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.query(hidden_states).view(B, N, H, D).transpose(1, 2)
        K = self.key(hidden_states).view(B, N, H, D).transpose(1, 2)
        V = self.value(hidden_states).view(B, N, H, D).transpose(1, 2)

        scale = D**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        scores = apply_attention_mask(scores, attention_mask)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        output = self.out_proj(context)

        return output, attn_weights if output_attentions else None
