"""
Spectraformer: Random feature linearisation with flexible kernel approximation.
Paper: https://arxiv.org/abs/2103.04246 (placeholder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

class SpectraformerAttention(BaseAttention):
    def __init__(self, config, rank=256):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _reshape(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, N, _ = hidden_states.shape
        Q = self._reshape(self.query(hidden_states), B)   # (B, H, N, D)
        K = self._reshape(self.key(hidden_states), B)
        V = self._reshape(self.value(hidden_states), B)

        # Orthogonal random features
        R = torch.randn(self.head_dim, self.rank, device=hidden_states.device)
        Q_proj = torch.einsum('bhnd,dr->bhnr', Q, R) / (self.head_dim ** 0.5)
        K_proj = torch.einsum('bhnd,dr->bhnr', K, R) / (self.head_dim ** 0.5)

        # Positive feature map (ReLU)
        Q_proj = torch.relu(Q_proj)
        K_proj = torch.relu(K_proj)

        # Linear attention
        KV = torch.einsum('bhnr,bhnd->bhrd', K_proj, V)            # (B, H, R, D)
        Z = K_proj.sum(dim=2, keepdim=True)                         # (B, H, 1, R)
        num = torch.einsum('bhnr,bhrd->bhnd', Q_proj, KV)
        den = torch.einsum('bhnr,bhr->bhn', Q_proj, Z.squeeze(2)).unsqueeze(-1) + 1e-8
        context = num / den

        if attention_mask is not None:
            context = context * attention_mask.unsqueeze(-1).to(context.dtype)

        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        return (self.out_proj(context), None)
    