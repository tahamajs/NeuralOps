import torch
import torch.nn as nn
import math
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("infini")
class InfiniAttention(nn.Module):
    """
    Infini-Attention: Efficient Infinite Context Transformer.
    Combines compressive memory (linear) with local sliding window attention.
    """
    def __init__(self, config, compress_dim=128, local_window=128):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.compress_dim = compress_dim
        self.local_window = local_window

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Compressive memory matrix: (1, num_heads, compress_dim, head_dim)
        self.memory = nn.Parameter(torch.randn(1, self.num_heads, compress_dim, self.head_dim) * 0.02)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, L, H, d).transpose(1, 2)

        # ---- Compressive memory attention ----
        # Q_memory: (B, H, L, compress_dim)
        mem = self.memory.expand(B, -1, -1, -1)  # (B, H, C, d)
        scores_mem = torch.matmul(Q, mem.transpose(-2, -1)) / math.sqrt(d)   # (B, H, L, C)
        attn_mem = torch.softmax(scores_mem, dim=-1)
        mem_out = torch.matmul(attn_mem, mem)   # (B, H, L, d)

        # ---- Local sliding window attention ----
        # Causal local window
        scale = d ** -0.5
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        # Build window mask
        idx = torch.arange(L, device=Q.device)
        window_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.local_window
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        attn_scores = attn_scores.masked_fill(~window_mask, float('-inf'))
        if attention_mask is not None:
            # assume attention_mask is (B, L) with 1 for keep, 0 for pad
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(dtype=attn_scores.dtype)
                attn_scores = attn_scores + (1.0 - mask) * -1e9
            else:
                attn_scores = attn_scores + attention_mask.unsqueeze(1)
        attn_local = torch.softmax(attn_scores, dim=-1)
        attn_local = self.dropout(attn_local)
        local_out = torch.matmul(attn_local, V)

        # ---- Combine ----
        out = local_out + mem_out
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out), None