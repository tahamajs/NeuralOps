import torch
import torch.nn as nn
import math
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("mla")
class MLAAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) – DeepSeek‑V2.
    Compresses the KV cache into a low‑rank latent space.
    """
    def __init__(self, config, rank=128):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.rank = rank

        # Query projection (standard)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Low‑rank KV bottleneck
        self.kv_down = nn.Linear(config.hidden_size, rank, bias=False)   # compress
        self.k_up = nn.Linear(rank, config.hidden_size, bias=False)      # expand to key
        self.v_up = nn.Linear(rank, config.hidden_size, bias=False)      # expand to value

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        # Queries – standard projection
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Compress into latent vector
        latent = self.kv_down(hidden_states)            # (B, L, rank)
        k = self.k_up(latent).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_up(latent).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Rearrange for attention
        q = q.transpose(1, 2)    # (B, num_heads, L, D)
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