"""
RPC Attention: Kernel PCA perspective on self‑attention.
"""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention


class RPCAttention(BaseAttention):
    def __init__(self, config, rank=64):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)

    def _reshape(self, x, batch_size) -> Any:
        """ reshape."""
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, **kwargs) -> Any:
        """Forward pass."""
        B, N, _ = hidden_states.shape
        Q = self._reshape(self.query(hidden_states), B)  # (B, H, N, D)
        K = self._reshape(self.key(hidden_states), B)
        V = self._reshape(self.value(hidden_states), B)

        # Flatten batch and heads for SVD (treat each head independently)
        (K.permute(0, 2, 1, 3).contiguous().view(B * N, self.num_heads, self.head_dim))
        # Actually SVD per head: we need (B*H, N, D)
        K_svd = (
            K.permute(0, 2, 1, 3)
            .contiguous()
            .view(B * self.num_heads, N, self.head_dim)
        )
        # SVD on each matrix? torch.svd works on 2D, so we loop over batch*heads
        U_list = []
        for i in range(B * self.num_heads):
            U_i, _, _ = torch.linalg.svd(K_svd[i], full_matrices=False)
            U_list.append(U_i[:, : self.rank])  # (N, rank)
        U_r = torch.stack(U_list, dim=0)  # (B*H, N, rank)
        U_r = U_r.view(B, self.num_heads, N, self.rank)  # (B, H, N, rank)

        # Project queries onto the principal components
        scores = torch.matmul(Q, U_r.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # (B, H, N, rank)
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(-1)

        probs = F.softmax(scores, dim=-1)
        # Aggregate values: map V back to original space? Actually we need to attend to the principal components.
        # The original RPC attention: context = probs @ (U_r^T V) (?) Let's use V directly as values in the reduced space.
        # We'll project V similarly: V_proj = U_r^T @ V (B,H,rank,D)
        V_proj = torch.matmul(U_r.transpose(-2, -1), V)  # (B, H, rank, D)
        context = torch.matmul(probs, V_proj)  # (B, H, N, D)

        context = (
            context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        )
        return (self.out_proj(context), None)
