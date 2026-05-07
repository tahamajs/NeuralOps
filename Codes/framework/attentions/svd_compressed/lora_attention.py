"""
LoRA Attention – each projection (Q, K, V, Out) is replaced by a LoRA adapter:
    output = original(x) + (alpha/rank) * B @ A @ x
The original weights are frozen; only the low‑rank adapters are trainable.
Zero‑shot initialisation (adapters start at zero).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from framework.attentions.core.registry import ATTENTION_REGISTRY


class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.original_layer(x)
        lora_out = F.linear(self.dropout(x), self.lora_A)   # (..., rank)
        lora_out = F.linear(lora_out, self.lora_B)          # (..., out_features)
        lora_out = lora_out * (self.alpha / self.rank)
        return orig_out + lora_out


class LoRAAttention(nn.Module):
    def __init__(self, model_config, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1,
                 pretrained_attn=None):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot init")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size

        # Keep original layers as they are, wrap them in LoRALinear
        self.q_proj = LoRALinear(pretrained_attn.q_lin, rank, alpha, dropout)
        self.k_proj = LoRALinear(pretrained_attn.k_lin, rank, alpha, dropout)
        self.v_proj = LoRALinear(pretrained_attn.v_lin, rank, alpha, dropout)
        self.out_proj = LoRALinear(pretrained_attn.out_lin, rank, alpha, dropout)

        self.dropout = nn.Dropout(dropout)

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


@ATTENTION_REGISTRY.register("lora_attn")
def inject_lora_attn(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    rank = getattr(comp_config, 'lora_rank', 8)
    alpha = getattr(comp_config, 'lora_alpha', 16.0)
    dropout = getattr(comp_config, 'dropout', 0.1)
    new_attn = LoRAAttention(model_config, rank, alpha, dropout, old_attn)
    layer.attention = new_attn
    return new_attn