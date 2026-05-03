"""
LEKAv3 – Linear‑time Efficient Kernel Attention (improved version)
Multi‑scale features, dynamic landmark count, and learnable gating.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class LEKAV3Attention(nn.Module):
    def __init__(self, config, comp_config, layer_idx, total_layers=6, bert_mode=False):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.head_dim = self.attention_head_size
        self.bert_mode = bert_mode
        self.layer_idx = layer_idx

        self.num_landmarks = getattr(comp_config, 'num_landmarks', 32)
        self.feature_dim   = getattr(comp_config, 'feature_dim', 128)

        self.query  = nn.Linear(config.hidden_size, self.all_head_size)
        self.key    = nn.Linear(config.hidden_size, self.all_head_size)
        self.value  = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(getattr(comp_config, 'dropout', 0.1))

        # Multi‑scale random projections
        self.register_buffer('proj1', torch.randn(self.head_dim, self.feature_dim))
        self.register_buffer('proj2', torch.randn(self.head_dim, self.feature_dim // 2))

        # Dynamic landmark scorer (based on V)
        self.landmark_scorer = nn.Sequential(
            nn.Linear(self.head_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        self.W_corr = nn.Parameter(
            torch.eye(self.head_dim).unsqueeze(0).repeat(self.num_attention_heads, 1, 1)
        )

        self.local_window_size = getattr(comp_config, 'local_window_size', 3)
        self.gate = nn.Parameter(torch.zeros(self.num_attention_heads))

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def kernel_feature_map(self, x):
        proj1 = self.proj1.to(x.device)
        proj2 = self.proj2.to(x.device)

        p1 = torch.matmul(x, proj1)  # (B, H, N, F)
        f1 = torch.cat([torch.cos(p1), torch.sin(p1)], dim=-1) / math.sqrt(self.feature_dim)

        p2 = torch.matmul(x, proj2)
        f2 = torch.cat([torch.cos(p2), torch.sin(p2)], dim=-1) / math.sqrt(self.feature_dim // 2)

        return torch.cat([f1, f2], dim=-1)

    def select_landmarks(self, K_features, V):
        B, H, N, feat_dim = K_features.shape
        L_max = self.num_landmarks
        if N <= L_max:
            return K_features, V

        # Use value features for scoring (more context‑aware)
        scores = self.landmark_scorer(V).squeeze(-1)  # (B, H, N)
        L = L_max
        _, idx = scores.topk(L, dim=-1)

        K_land = torch.gather(K_features, 2, idx.unsqueeze(-1).expand(-1, -1, -1, feat_dim))
        V_land = torch.gather(V, 2, idx.unsqueeze(-1).expand(-1, -1, -1, V.shape[-1]))
        return K_land, V_land

    @staticmethod
    def softpick(x, dim=-1, temperature=1.0):
        m = x.max(dim=dim, keepdim=True).values
        e = torch.exp((x - m) / temperature)
        denom = e - torch.exp(-m / temperature)
        p = F.relu(denom)
        return p / (p.sum(dim=dim, keepdim=True) + 1e-9)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        B, N, D = hidden_states.shape
        H = self.num_attention_heads
        L = self.num_landmarks

        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        Q_features = self.kernel_feature_map(Q)
        K_features = self.kernel_feature_map(K)

        K_land, V_land = self.select_landmarks(K_features, V)

        W = self.W_corr.unsqueeze(0).expand(B, -1, -1, -1)
        V_land_adj = torch.einsum('bhld,bhde->bhle', V_land, W)

        attn_weights = torch.matmul(Q_features, K_land.transpose(-2, -1))

        # Mask handling (same as v2)
        mask = attention_mask
        if mask is None:
            mask = kwargs.get('attn_mask', None)
        if mask is not None:
            if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
                token_valid = mask[:, 0, 0, :] > -1e9
            elif mask.dim() == 4:
                token_valid = mask.squeeze(1).squeeze(1) > -1e9
            elif mask.dim() == 3:
                token_valid = mask.squeeze(1).bool()
            elif mask.dim() == 2:
                token_valid = mask.bool()
            else:
                token_valid = None
            if token_valid is not None:
                attn_weights = attn_weights.masked_fill(~token_valid.unsqueeze(1).unsqueeze(-1), -1e9)

        attn_weights = self.softpick(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_land = torch.matmul(attn_weights, V_land_adj)

        # Local window attention
        gate_val = torch.sigmoid(self.gate).view(1, H, 1, 1)
        window_size = self.local_window_size
        idx = torch.arange(N, device=hidden_states.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        window_mask = (dist <= window_size // 2).unsqueeze(0).unsqueeze(0)

        scale = self.head_dim ** -0.5
        local_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        local_scores = local_scores.masked_fill(~window_mask, -1e9)
        local_attn = F.softmax(local_scores, dim=-1)
        context_local = torch.matmul(local_attn, V)

        context = gate_val * context_land + (1 - gate_val) * context_local

        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(context)
        return out, None

    def get_ortho_loss(self):
        return 0.0


from framework.attentions.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("leka_v3")
def inject_leka_v3(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    new_attn = LEKAV3Attention(
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