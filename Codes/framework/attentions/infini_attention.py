# framework/attentions/specialized/infini_attention.py
import torch
import torch.nn as nn
import math
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY
import torch.nn.functional as F

@ATTENTION_REGISTRY.register("infini")
class InfiniAttention(BaseAttention):
    """
    Infini-Attention with compressive memory and sliding window.
    Paper: https://arxiv.org/abs/2404.07143
    """

    def __init__(self, config, rank: int = None, compress_dim=128, local_window=128):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.compress_dim = compress_dim
        self.local_window = local_window
        self.dropout = nn.Dropout(config.dropout)   # not attention_probs_dropout_prob

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Compressive memory matrix (learnable)
        self.memory = nn.Parameter(torch.randn(1, self.num_heads, compress_dim, self.head_dim) * 0.02)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, **kwargs):
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        # Compressive memory attention
        mem = self.memory.expand(B, -1, -1, -1)                     # (B, H, C, Dh)
        mem_scores = torch.matmul(q, mem.transpose(-2, -1)) / (Dh ** 0.5)
        mem_attn = torch.softmax(mem_scores, dim=-1)
        mem_out = torch.matmul(mem_attn, mem)

        # Sliding‑window attention
        window_mask = torch.ones(L, L, device=q.device).tril(diagonal=self.local_window - 1).bool()
        scale = 1.0 / (Dh ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), -1e9)

        if attention_mask is not None:
            scores = scores + (attention_mask.float() - 1.0) * -1e9

        attn_local = F.softmax(scores, dim=-1)
        attn_local = F.dropout(attn_local, p=0.1, training=self.training, inplace=False)
        local_out = torch.matmul(attn_local, v)

        out = local_out + mem_out
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if output_attentions:
            return out, attn_local
        return out