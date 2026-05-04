"""Linformer attention – registers as 'linformer'."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("linformer")
class LinformerAttention(BaseAttention):
    def __init__(self, config, rank=64):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Linformer projection matrices (learnable)
        self.E = nn.Parameter(torch.randn(config.hidden_size, rank) * 0.02)
        self.F = nn.Parameter(torch.randn(config.hidden_size, rank) * 0.02)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self._transpose(self.query(hidden_states))
        K = self._transpose(self.key(hidden_states))
        V = self._transpose(self.value(hidden_states))

        # Project keys and values
        K_proj = torch.matmul(self.E.T, K)  # (batch, heads, rank, head_dim)
        V_proj = torch.matmul(self.F.T, V)

        scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(-1)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, V_proj)

        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, -1)
        output_attentions = kwargs.get('output_attentions', False)
        return (context, None) if output_attentions else (context,)

    def _transpose(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)