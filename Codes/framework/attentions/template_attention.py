"""
Template for creating new attention mechanisms.
Customizable and ready to run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY
from ...core.base import BaseAttention


@ATTENTION_REGISTRY.register("your_attention_name")
class YourCustomAttention(BaseAttention):
    """
    Your custom attention mechanism.
    
    Currently implements a standard Transformer attention with an optional
    low‑rank modulation based on pooled hidden states.
    """

    def __init__(self, config, rank: Optional[int] = None, **kwargs):
        super().__init__(config, rank)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = config.attention_probs_dropout_prob

        self.custom_param = kwargs.get('custom_param', 0.1)

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        if rank is not None and rank > 0:
            self.rank_proj = nn.Linear(self.hidden_size, rank, bias=False)
        else:
            self.rank_proj = None

        self._init_weights()

    def _init_weights(self):
        for module in [self.query, self.key, self.value, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _reshape(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        Q = self._reshape(self.query(hidden_states), batch_size)
        K = self._reshape(self.key(hidden_states), batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # --- Your custom modification ---
        if self.rank_proj is not None:
            # Pool sequence dimension → global vector, then project to rank
            H_pool = hidden_states.mean(dim=1)          # (B, D)
            H_rank = self.rank_proj(H_pool)            # (B, rank)
            modulation = torch.sigmoid(H_rank).mean(dim=-1, keepdim=True)  # (B, 1)
            modulation = modulation.view(batch_size, 1, 1, 1)
            scores = scores * modulation

        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        context = torch.matmul(attn_probs, V)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(context)

        if output_attentions:
            return output, attn_probs
        return output, None

    @property
    def complexity(self) -> str:
        return "quadratic" if self.rank_proj is None else "quadratic (with rank modulation)"

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}, rank={self.rank}"