"""
SVD Projection Attention – Each projection (Q, K, V, Out) is replaced by
a low‑rank SVDLinear, but we keep the standard attention mechanism.
Similar to svd_linear_attention but with a different factorisation style.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class SVDProjectionLinear(nn.Module):
    """
    Alternative SVDLinear: stores U and V, with σ absorbed into one of them.
    """
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.U @ self.V
        return F.linear(x, weight, self.bias)


class SVDProjectionAttention(nn.Module):
    def __init__(self, model_config, rank: int = 32, dropout: float = 0.1,
                 pretrained_attn=None):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot init")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.rank = rank

        self.q_proj = SVDProjectionLinear(self.hidden_size, self.all_head_size, rank,
                                          bias=pretrained_attn.q_lin.bias is not None)
        self.k_proj = SVDProjectionLinear(self.hidden_size, self.all_head_size, rank,
                                          bias=pretrained_attn.k_lin.bias is not None)
        self.v_proj = SVDProjectionLinear(self.hidden_size, self.all_head_size, rank,
                                          bias=pretrained_attn.v_lin.bias is not None)
        self.out_proj = SVDProjectionLinear(self.all_head_size, self.hidden_size, rank,
                                            bias=pretrained_attn.out_lin.bias is not None)

        self._init_from_linear(self.q_proj, pretrained_attn.q_lin)
        self._init_from_linear(self.k_proj, pretrained_attn.k_lin)
        self._init_from_linear(self.v_proj, pretrained_attn.v_lin)
        self._init_from_linear(self.out_proj, pretrained_attn.out_lin)

        self.dropout = nn.Dropout(dropout)

    def _init_from_linear(self, svd_linear, orig_linear):
        with torch.no_grad():
            W = orig_linear.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(self.rank, U.size(1), Vt.size(0))
            svd_linear.U.data[:, :r] = (U[:, :r] * S[:r]).to(svd_linear.U.dtype)
            svd_linear.V.data[:r, :] = Vt[:r, :].to(svd_linear.V.dtype)
            if svd_linear.bias is not None and orig_linear.bias is not None:
                svd_linear.bias.data = orig_linear.bias.data.clone()

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, H, D).transpose(1, 2)

        scale = D ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask) * -1e9

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_proj(context)

        return output, attn_weights if output_attentions else None


@ATTENTION_REGISTRY.register("svd_proj_attn")
def inject_svd_proj_attn(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    rank = getattr(comp_config, 'svd_rank', 32)
    dropout = getattr(comp_config, 'dropout', 0.1)
    new_attn = SVDProjectionAttention(model_config, rank, dropout, old_attn)
    layer.attention = new_attn
    return new_attn
