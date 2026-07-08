"""
SALT Attention: Singular Value Adaptation for attention projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class SALTLinear(nn.Module):
    """A linear layer with SVD‑based adaptation: W = U @ diag(S) @ Vt + A @ B."""

    def __init__(self, original_weight: torch.Tensor, rank: int = 16):
        super().__init__()
        out_features, in_features = original_weight.shape
        U, S, Vt = torch.linalg.svd(original_weight.float(), full_matrices=False)
        self.register_buffer("U", U)
        self.register_buffer("Vt", Vt)
        self.S = nn.Parameter(S)
        self.gamma = nn.Parameter(torch.ones(rank))
        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        adapted_S = self.S[: self.gamma.shape[0]] * self.gamma
        adapted_weight = (
            self.U[:, : self.gamma.shape[0]]
            @ torch.diag(adapted_S)
            @ self.Vt[: self.gamma.shape[0], :]
        )
        low_rank = self.A @ self.B
        W = adapted_weight + low_rank
        return F.linear(x, W)


@ATTENTION_REGISTRY.register("salt_attention")
class SALTAttention(BaseAttention):
    """
    SALT‑adapted attention.  Replaces Q/K/V/O projections with SALTLinear layers.
    """

    def __init__(self, config, rank: int = 16, pretrained_attn=None):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

        if pretrained_attn is None:
            raise ValueError("pretrained_attn required")

        self.q_proj = SALTLinear(pretrained_attn.q_lin.weight.data, rank)
        self.k_proj = SALTLinear(pretrained_attn.k_lin.weight.data, rank)
        self.v_proj = SALTLinear(pretrained_attn.v_lin.weight.data, rank)
        self.out_proj = SALTLinear(pretrained_attn.out_lin.weight.data, rank)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        scale = Dh**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(dtype=scores.dtype)
                scores = scores + (1.0 - mask) * -1e9
            else:
                scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        if head_mask is not None:
            attn = attn * head_mask

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(context)

        if output_attentions:
            return out, attn
        return out, None


def inject_salt_attention(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
):
    rank = getattr(comp_config, "rank", 16)
    new_attn = SALTAttention(config=model_config, rank=rank, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn
