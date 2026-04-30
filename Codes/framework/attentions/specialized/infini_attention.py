# framework/attentions/infini_attention.py
import torch
import torch.nn as nn
import math
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("infini")
class InfiniAttention(nn.Module):
    """
    Infini-Attention: Efficient Infinite Context Transformer (Google).
    Combines compressive memory with linear and local attention.
    """
    def __init__(self, config, compress_dim=128):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.compress_dim = compress_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.memory = nn.Parameter(torch.zeros(1, self.num_heads, compress_dim, self.head_dim))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Long‑term compressed memory attention
        mem_out = torch.einsum('bqhd,hrd->bqhr', Q, self.memory) / math.sqrt(self.head_dim)
        # Local attention (sliding window)
        # ...
        return out, None