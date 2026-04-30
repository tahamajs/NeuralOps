# framework/compression/padre_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_attention import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("padre")
class PADReAttention(BaseAttention):
    """
    PADRe: Polynomial Attention with dropout (no softmax).
    Paper: https://arxiv.org/abs/2402.12345 (placeholder)
    """
    def __init__(self, config, order=2):
        super().__init__(config, rank=order)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.order = order

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _reshape(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self._reshape(self.query(hidden_states), batch_size)
        K = self._reshape(self.key(hidden_states), batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)

        scale = 1.0 / (self.head_dim ** 0.5)
        S = torch.einsum('bqhd,bkhd->bqk', Q, K) * scale
        # Replace softmax with polynomial: (1 + S/order)^order
        S = S / self.order
        poly = (1 + S) ** self.order
        # Normalize
        Z = poly.sum(dim=-1, keepdim=True)
        probs = poly / Z
        probs = self.dropout(probs)

        context = torch.einsum('bqk,bkhd->bqhd', probs, V)
        context = context.permute(0,2,1,3).contiguous().view(batch_size, seq_len, self.all_head_size)
        return self.out_proj(context), None