# framework/attentions/routing_attention.py
import torch
import torch.nn as nn
import math
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("routing")
class RoutingAttention(nn.Module):
    """
    Routing Transformer: Attention via online k-means clustering.
    """
    def __init__(self, config, num_clusters=32):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_clusters = num_clusters
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Cluster keys using k-means
        # ...
        # Compute intra‑cluster attention
        # ...
        return out, None