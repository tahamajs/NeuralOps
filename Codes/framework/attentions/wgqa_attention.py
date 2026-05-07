import torch
import torch.nn as nn
import math
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("wgqa")
class WeightedGroupedQueryAttention(nn.Module):
    """
    Weighted Grouped-Query Attention (WGQA).
    Fixed grouping + learnable per‑channel importance weights on K/V heads.
    """
    def __init__(self, config, num_kv_heads=4):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = config.hidden_size // self.num_heads

        # Learnable weights for KV heads (broadcast across sequence & batch)
        self.k_weights = nn.Parameter(torch.ones(self.num_kv_heads, 1))   # (H_kv, 1)
        self.v_weights = nn.Parameter(torch.ones(self.num_kv_heads, 1))

        # Projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply per‑head importance: multiply each head by its scalar weight
        # weights have shape (H_kv, 1) -> (1,1,H_kv,1) after unsqueezing
        k = k * self.k_weights.view(1, 1, self.num_kv_heads, 1)
        v = v * self.v_weights.view(1, 1, self.num_kv_heads, 1)

        # Repeat to match query heads (fixed grouping)
        repeat_factor = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)

        q = q.transpose(1, 2)   # (B, num_heads, L, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)

        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(context)

        return out, attn_probs