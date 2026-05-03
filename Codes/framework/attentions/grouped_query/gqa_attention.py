import torch
import torch.nn as nn
import math
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("gqa")
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    Shares a smaller number of key/value heads across multiple query heads.
    """
    def __init__(self, config, num_kv_heads=4):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = config.hidden_size // self.num_heads
        # Projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q (full heads), K/V (fewer heads)
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Replicate K/V heads to match Q head count
        repeat_factor = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=2)   # (B, L, num_heads, D)
        v = v.repeat_interleave(repeat_factor, dim=2)

        # Standard scaled dot‑product attention
        q = q.transpose(1, 2)      # (B, num_heads, L, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)

        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)                     # (B, num_heads, L, D)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(context)

        return out, attn_probs   # returns attention weights for inspection