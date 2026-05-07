"""
SVD‑Linear Attention – Compress Q/K/V/Out projections with truncated SVD.
Zero‑shot initialisation from a pretrained DistilBERT attention layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from framework.attentions.core.registry import ATTENTION_REGISTRY


class SVDLinear(nn.Module):
    """
    Linear layer replaced by U @ W, where U (out_dim, rank) and W (rank, in_dim)
    are learned factors. The original weight is approximated by SVD truncation.
    """
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.W = nn.Parameter(torch.empty(rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.U @ self.W   # (out_dim, in_dim)
        return F.linear(x, weight, self.bias)


class SVDLinearAttention(nn.Module):
    """
    Standard full attention, but all linear projections (Q, K, V, Out)
    are replaced by SVDLinear with given rank.
    """
    def __init__(self, model_config, rank: int = 32, dropout: float = 0.1,
                 pretrained_attn: Optional[nn.Module] = None):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot SVD initialisation")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.rank = rank

        # Create SVDLinear layers with the same dimensions as the original
        self.q_proj = SVDLinear(self.hidden_size, self.all_head_size, rank,
                                bias=pretrained_attn.q_lin.bias is not None)
        self.k_proj = SVDLinear(self.hidden_size, self.all_head_size, rank,
                                bias=pretrained_attn.k_lin.bias is not None)
        self.v_proj = SVDLinear(self.hidden_size, self.all_head_size, rank,
                                bias=pretrained_attn.v_lin.bias is not None)
        self.out_proj = SVDLinear(self.all_head_size, self.hidden_size, rank,
                                  bias=pretrained_attn.out_lin.bias is not None)

        # Copy pretrained weights via truncated SVD
        with torch.no_grad():
            self._init_from_linear(self.q_proj, pretrained_attn.q_lin)
            self._init_from_linear(self.k_proj, pretrained_attn.k_lin)
            self._init_from_linear(self.v_proj, pretrained_attn.v_lin)
            self._init_from_linear(self.out_proj, pretrained_attn.out_lin)

        self.dropout = nn.Dropout(dropout)

    def _init_from_linear(self, svd_linear: SVDLinear, orig_linear: nn.Linear):
        """Approximate original weight by SVD and copy factors."""
        W = orig_linear.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        r = min(self.rank, U.size(1), Vt.size(0))
        svd_linear.U.data[:, :r] = (U[:, :r] * S[:r]).to(svd_linear.U.dtype)
        svd_linear.W.data[:r, :] = Vt[:r, :].to(svd_linear.W.dtype)
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


@ATTENTION_REGISTRY.register("svd_linear_attn")
def inject_svd_linear_attn(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    rank = getattr(comp_config, 'svd_rank', 32)
    dropout = getattr(comp_config, 'dropout', 0.1)
    new_attn = SVDLinearAttention(model_config, rank, dropout, old_attn)
    layer.attention = new_attn
    return new_attn