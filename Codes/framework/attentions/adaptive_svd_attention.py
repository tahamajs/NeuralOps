"""
Adaptive SVD Attention – each projection is decomposed as:
    W = U @ V + A @ B
where U,V come from truncated SVD of the pretrained weight, and A,B are
trainable residual adapters. This gives better approximation than pure SVD.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


class AdaptiveSVDLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        svd_rank: int,
        residual_rank: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.svd_rank = svd_rank
        self.residual_rank = residual_rank
        self.U = nn.Parameter(torch.empty(out_features, svd_rank))
        self.V = nn.Parameter(torch.empty(svd_rank, in_features))
        if residual_rank > 0:
            self.A = nn.Parameter(torch.empty(out_features, residual_rank))
            self.B = nn.Parameter(torch.empty(residual_rank, in_features))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        if self.A is not None and self.B is not None:
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_svd_factors(self, U: torch.Tensor, V: torch.Tensor):
        with torch.no_grad():
            self.U.copy_(U)
            self.V.copy_(V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Vx = F.linear(x, self.V)
        out = F.linear(Vx, self.U)
        if self.A is not None and self.B is not None:
            Bx = F.linear(x, self.B)
            out = out + F.linear(Bx, self.A)
        if self.bias is not None:
            out = out + self.bias
        return out


class AdaptiveSVDAttention(nn.Module):
    def __init__(
        self,
        model_config,
        svd_rank: int = 32,
        residual_rank: int = 8,
        dropout: float = 0.1,
        pretrained_attn=None,
    ):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot init")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size

        self.q_proj = AdaptiveSVDLinear(
            self.hidden_size,
            self.all_head_size,
            svd_rank,
            residual_rank,
            bias=pretrained_attn.q_lin.bias is not None,
        )
        self.k_proj = AdaptiveSVDLinear(
            self.hidden_size,
            self.all_head_size,
            svd_rank,
            residual_rank,
            bias=pretrained_attn.k_lin.bias is not None,
        )
        self.v_proj = AdaptiveSVDLinear(
            self.hidden_size,
            self.all_head_size,
            svd_rank,
            residual_rank,
            bias=pretrained_attn.v_lin.bias is not None,
        )
        self.out_proj = AdaptiveSVDLinear(
            self.all_head_size,
            self.hidden_size,
            svd_rank,
            residual_rank,
            bias=pretrained_attn.out_lin.bias is not None,
        )

        # Initialise SVD part from pretrained weights
        self._init_svd_part(self.q_proj, pretrained_attn.q_lin)
        self._init_svd_part(self.k_proj, pretrained_attn.k_lin)
        self._init_svd_part(self.v_proj, pretrained_attn.v_lin)
        self._init_svd_part(self.out_proj, pretrained_attn.out_lin)

        self.dropout = nn.Dropout(dropout)

    def _init_svd_part(self, adap_linear: AdaptiveSVDLinear, orig_linear: nn.Linear):
        with torch.no_grad():
            W = orig_linear.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(adap_linear.svd_rank, U.size(1), Vt.size(0))
            U_r = (U[:, :r] * S[:r]).to(adap_linear.U.dtype)
            V_r = Vt[:r, :].to(adap_linear.V.dtype)
            adap_linear.set_svd_factors(U_r, V_r)
            if adap_linear.bias is not None and orig_linear.bias is not None:
                adap_linear.bias.data = orig_linear.bias.data.clone()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, H, D).transpose(1, 2)

        scale = D**-0.5
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


@ATTENTION_REGISTRY.register("adaptive_svd_attn")
def inject_adaptive_svd_attn(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
):
    svd_rank = getattr(comp_config, "svd_rank", 32)
    res_rank = getattr(comp_config, "residual_rank", 8)
    dropout = getattr(comp_config, "dropout", 0.1)
    new_attn = AdaptiveSVDAttention(model_config, svd_rank, res_rank, dropout, old_attn)
    layer.attention = new_attn
    return new_attn
