import torch
import torch.nn as nn
import math
from framework.core.registry import ATTENTION_REGISTRY
import torch.nn.functional as F

@ATTENTION_REGISTRY.register("gqa")
class GroupedQueryAttention(nn.Module):
    def __init__(self, config, num_kv_heads=4):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim)
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask) * -1e9
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out), None