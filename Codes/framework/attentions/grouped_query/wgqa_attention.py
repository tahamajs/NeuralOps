# framework/attentions/wgqa_attention.py
import torch
import torch.nn as nn
import math
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("wgqa")
class WeightedGroupedQueryAttention(nn.Module):
    """
    Weighted Grouped-Query Attention (WGQA). Learnable per-head weighting.
    """
    def __init__(self, config, num_kv_heads=4):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = config.hidden_size // self.num_heads
        # Weighted grouped attention: additional learnable parameters per head
        self.k_weights = nn.Parameter(torch.ones(self.num_kv_heads))
        self.v_weights = nn.Parameter(torch.ones(self.num_kv_heads))
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        # Apply per-head weights to K and V
        k = k * self.k_weights.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        v = v * self.v_weights.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        # Repeat to match query heads
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.out_proj(context)
        return out, None