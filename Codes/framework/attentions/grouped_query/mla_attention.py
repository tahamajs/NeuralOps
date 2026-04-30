# framework/attentions/mla_attention.py
import torch
import torch.nn as nn
import math
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("mla")
class MLAAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2.
    Compresses KV cache via low-rank projection.
    """
    def __init__(self, config, rank=128):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.rank = rank
        # Query projection (no compression for queries)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # Key-Value low-rank compression layers
        self.kv_down = nn.Linear(config.hidden_size, rank, bias=False)
        self.k_up = nn.Linear(rank, config.hidden_size, bias=False)
        self.v_up = nn.Linear(rank, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # Additional positional encoding (RoPE) would be applied to queries and keys.

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        # Queries: standard projection
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Compress KV into a low-rank latent vector
        compress = self.kv_down(hidden_states)   # (B, L, rank)
        k = self.k_up(compress).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_up(compress).view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Apply positional encoding if needed (e.g., RoPE shared among q, k)
        # Compute attention (same as standard multi-head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.out_proj(context)
        return out, None