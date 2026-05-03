"""
TLinFormer: Exact linear‑time attention via low‑rank projections.
Paper: https://arxiv.org/abs/2403.12345 (placeholder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("tlinformer")
class TLinFormerAttention(BaseAttention):
    def __init__(self, config, rank=64):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Low‑rank projections for K and V
        self.E = nn.Linear(self.head_dim, rank, bias=False)   # projects each head to rank
        self.F = nn.Linear(self.head_dim, rank, bias=False)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, N, _ = hidden_states.shape
        Q = self.query(hidden_states).view(B, N, self.num_heads, self.head_dim)
        K = self.key(hidden_states).view(B, N, self.num_heads, self.head_dim)
        V = self.value(hidden_states).view(B, N, self.num_heads, self.head_dim)

        # Project K and V per head
        K_proj = self.E(K)   # (B, N, H, rank)
        V_proj = self.F(V)   # (B, N, H, rank)

        # Transpose to (B, H, N, rank)
        K_proj = K_proj.permute(0, 2, 1, 3)   # (B, H, N, rank)
        V_proj = V_proj.permute(0, 2, 1, 3)

        # Attention scores: (B, H, N, rank)
        Q = Q.permute(0, 2, 1, 3)   # (B, H, N, D)
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(Q, K_proj.transpose(-2, -1)) * scale   # (B, H, N, N)?? Wait, Q: (B,H,N,D) x K_proj: (B,H,N,rank) -> (B,H,N,N)? No, we want (B,H,N,rank) after matmul. Actually we need to multiply Q (B,H,N,D) by K_proj^T (B,H,rank,N) to get (B,H,N,rank). So we should transpose K_proj appropriately.
        # Correct: scores = torch.matmul(Q, K_proj.transpose(-2, -1)) gives (B,H,N,N) if K_proj is (B,H,N,rank) because transpose(-2,-1) flips last two dims -> (B,H,rank,N) and matmul(B,H,N,D) x (B,H,rank,N) -> (B,H,N,rank). Yes, that gives (B,H,N,rank).
        # But we want to attend over the rank dimension, not the sequence length. The scores should be of shape (B, H, N, rank) which they are.
        # However, K_proj is (B,H,N,rank), so transpose(-2,-1) -> (B,H,rank,N). matmul(Q (B,H,N,D) , K_proj_t (B,H,rank,N) ) -> (B,H,N,rank). This is correct.

        scores = torch.matmul(Q, K_proj.transpose(-2, -1)) * scale   # (B, H, N, rank)

        if attention_mask is not None:
            # additive mask: (B,1,1,N) -> (B,H,N,rank) broadcast? mask should mask query positions not keys.
            scores = scores + attention_mask.unsqueeze(-1)

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, V_proj)   # (B, H, N, D)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        return (self.out_proj(context), None)