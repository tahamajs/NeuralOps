"""
Galerkin Transformer: Petrov‑Galerkin projection attention.
Paper: https://arxiv.org/abs/2105.14906
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.attentions.core.base import BaseAttention
from framework.attentions.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("galerkin")
class GalerkinAttention(BaseAttention):
    def __init__(self, config, rank=64):
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
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (B, H, N, D)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, N, _ = hidden_states.shape
        Q = self._reshape(self.query(hidden_states), B)   # (B, H, N, D)
        K = self._reshape(self.key(hidden_states), B)
        V = self._reshape(self.value(hidden_states), B)

        # Create a random sketching matrix of size (N, rank)
        sketch = torch.randn(N, self.rank, device=hidden_states.device) / (N ** 0.5)

        # Project keys and values onto the low‑rank subspace
        # K: (B, H, N, D) -> (B, H, rank, D) via sketch
        K_sketch = torch.einsum('bhnd,nr->bhrd', K, sketch)   # (B, H, rank, D)
        V_sketch = torch.einsum('bhnd,nr->bhrd', V, sketch)   # (B, H, rank, D)

        # Attention scores: (B, H, N, rank)
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(Q, K_sketch.transpose(-2, -1)) * scale   # (B, H, N, rank)

        # Mask handling – mask is (B, 1, 1, N) additive; adapt to (B, H, N, rank)
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(-1)

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, V_sketch)   # (B, H, N, D)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(context)
        return (out, None)