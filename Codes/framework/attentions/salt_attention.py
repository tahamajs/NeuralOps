r"""
SALT Attention: Singular Value Adaptation with Low-Rank Tuning.

Parameter-efficient adaptation mechanism for transformer attention projections combining
orthogonal singular vector preservation, singular value rescaling ($\gamma$), and additive low-rank adapters ($A B$).
"""

from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class SALTLinear(nn.Module):
    r"""
    Linear projection layer with SVD singular value modulation and low-rank residual branch.

    Formulation:
        $$W = U_{:, :r} \operatorname{diag}(\Sigma_{1:r} \odot \gamma) V_{1:r, :}^\top + A B$$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        original_weight: Optional[torch.Tensor] = None,
        original_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = min(rank, min(in_features, out_features))

        if original_weight is not None:
            W = original_weight.float()
        else:
            W = torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features))

        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        r = self.rank

        self.register_buffer("U", U[:, :r].contiguous())
        self.register_buffer("Vt", Vt[:r, :].contiguous())
        self.register_buffer("S", S[:r].contiguous())

        self.gamma = nn.Parameter(torch.ones(r))
        self.A = nn.Parameter(torch.randn(out_features, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, in_features))  # Zero init for exact initial preservation

        if original_bias is not None:
            self.bias = nn.Parameter(original_bias.clone())
        else:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute adapted linear projection."""
        adapted_S = self.S * self.gamma
        W_svd = (self.U * adapted_S.unsqueeze(0)) @ self.Vt
        W_lora = self.A @ self.B
        W = (W_svd + W_lora).to(dtype=x.dtype)
        return F.linear(x, W, self.bias.to(dtype=x.dtype) if self.bias is not None else None)


@ATTENTION_REGISTRY.register("salt_attention")
@ATTENTION_REGISTRY.register("salt")
class SALTAttention(BaseAttention):
    r"""
    SALT Attention module replacing standard linear projections with SALTLinear.

    Mathematical Formulation:
        1. SVD & LoRA Factorized Linear Projections:
           $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
        2. Scaled Dot-Product Attention:
           $$O = \operatorname{Softmax}\left(\frac{Q K^\top}{\sqrt{d}} + M\right) V W_O$$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.rank: int = rank if rank is not None else getattr(config, "salt_rank", getattr(config, "rank", 16))

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)

        # Projections
        if pretrained_attn is not None:
            self.init_from_pretrained(pretrained_attn)
        else:
            self.q_proj = SALTLinear(hidden_size, self.all_head_size, self.rank)
            self.k_proj = SALTLinear(hidden_size, self.all_head_size, self.rank)
            self.v_proj = SALTLinear(hidden_size, self.all_head_size, self.rank)
            self.out_proj = SALTLinear(self.all_head_size, hidden_size, self.rank)

    def init_from_pretrained(self, src: Any) -> None:
        r"""Initialize SALTLinear projections directly from pretrained module."""
        from framework.core.adapters import UniversalAdapter
        hidden_size = self.num_heads * self.head_dim

        for proj, attr in [("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "out_proj")]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                setattr(self, attr, SALTLinear(w.size(1), w.size(0), self.rank, original_weight=w, original_bias=b))
            except Exception:
                setattr(self, attr, SALTLinear(hidden_size, hidden_size, self.rank))

    def copy_weights(self, src: Any) -> None:
        r"""Universal adapter weight copying."""
        self.init_from_pretrained(src)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        scale = Dh ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        valid_mask_2d = get_token_valid_mask(attention_mask, B, L, device)
        key_valid_4d = valid_mask_2d.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~key_valid_4d, -10000.0)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        if head_mask is not None:
            attn = attn * head_mask

        context = torch.matmul(attn, V)
        context = context * valid_mask_2d.unsqueeze(1).unsqueeze(-1).to(dtype)
        context = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        out = self.out_proj(context)

        return out, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("salt_inject")
def inject_salt_attention(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> SALTAttention:
    rank = getattr(comp_config, "rank", getattr(comp_config, "salt_rank", 16))
    new_attn = SALTAttention(config=model_config, rank=rank, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn
