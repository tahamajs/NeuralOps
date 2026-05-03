"""
LEKAv1 – Linear‑time Efficient Kernel Attention (basic version)
Pure kernel‑based linear attention (no landmarks).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class LEKAV1Attention(nn.Module):
    """
    Simple linear attention using random Fourier features + Softpick.
    """
    def __init__(self, config, comp_config, layer_idx, total_layers=6, bert_mode=False):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.head_dim = self.attention_head_size
        self.bert_mode = bert_mode
        self.layer_idx = layer_idx

        self.feature_dim = getattr(comp_config, 'feature_dim', 256)

        self.query  = nn.Linear(config.hidden_size, self.all_head_size)
        self.key    = nn.Linear(config.hidden_size, self.all_head_size)
        self.value  = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(getattr(comp_config, 'dropout', 0.1))

        # Random projection matrix (one per head, shared)
        self.register_buffer(
            'proj',
            torch.randn(self.num_attention_heads, self.head_dim, self.feature_dim)
        )

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def kernel_feature_map(self, x):
        # x: (B, H, N, D)
        proj = self.proj.to(x.device)   # (H, D, F)
        x_proj = torch.einsum('bhnd,hdf->bhnf', x, proj)  # (B, H, N, F)
        phi = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)  # (B, H, N, 2F)
        phi = phi / math.sqrt(self.feature_dim)
        return phi

    @staticmethod
    def softpick(x, dim=-1, temperature=1.0):
        m = x.max(dim=dim, keepdim=True).values
        e = torch.exp((x - m) / temperature)
        denom = e - torch.exp(-m / temperature)
        p = F.relu(denom)
        return p / (p.sum(dim=dim, keepdim=True) + 1e-9)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        B, N, D = hidden_states.shape
        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        Q_features = self.kernel_feature_map(Q)   # (B, H, N, 2F)
        K_features = self.kernel_feature_map(K)

        # Linear attention
        KV = torch.einsum('bhnf,bhnd->bhfd', K_features, V)   # (B, H, F, D)
        K_sum = K_features.sum(dim=2, keepdim=True)           # (B, H, 1, F)

        num = torch.einsum('bhnf,bhfd->bhnd', Q_features, KV)
        denom = torch.einsum('bhnf,bhf->bhn', Q_features, K_sum.squeeze(2)).unsqueeze(-1) + 1e-8
        context = num / denom

        if attention_mask is not None:
            # assume 2D mask (B,N) with 1 for keep
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(context.dtype)
            elif attention_mask.dim() == 4:
                mask = attention_mask[:, 0, 0, :].unsqueeze(1).unsqueeze(2)
            else:
                mask = attention_mask
            context = context * mask

        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(context)
        return out, None

    def get_ortho_loss(self):
        return 0.0


from framework.attentions.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("leka_v1")
def inject_leka_v1(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    new_attn = LEKAV1Attention(
        config=model_config,
        comp_config=comp_config,
        layer_idx=layer_idx,
        total_layers=total_layers,
        bert_mode=False,
    )
    # Copy pretrained weights
    new_attn.query.weight.data.copy_(old_attn.q_lin.weight.data)
    new_attn.query.bias.data.copy_(old_attn.q_lin.bias.data)
    new_attn.key.weight.data.copy_(old_attn.k_lin.weight.data)
    new_attn.key.bias.data.copy_(old_attn.k_lin.bias.data)
    new_attn.value.weight.data.copy_(old_attn.v_lin.weight.data)
    new_attn.value.bias.data.copy_(old_attn.v_lin.bias.data)
    new_attn.out_proj.weight.data.copy_(old_attn.out_lin.weight.data)
    new_attn.out_proj.bias.data.copy_(old_attn.out_lin.bias.data)
    layer.attention = new_attn
    return new_attn