"""
Tensor Product Attention (TPA)
Paper: "Tensor Product Attention Is All You Need" (NeurIPS 2025)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_attention import BaseAttention
from ..core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("tpa")
class TPAttention(BaseAttention):
    def __init__(self, config, rank_Q=1, rank_K=1, rank_V=1, use_cache=False):
        super().__init__(config, rank_Q)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank_Q = rank_Q
        self.rank_K = rank_K
        self.rank_V = rank_V
        self.use_cache = use_cache

        self.Q_factor1 = nn.Linear(config.hidden_size, self.num_heads * self.rank_Q, bias=False)
        self.Q_factor2 = nn.Linear(config.hidden_size, self.num_heads * self.rank_Q, bias=False)
        self.K_factor1 = nn.Linear(config.hidden_size, self.num_heads * self.rank_K, bias=False)
        self.K_factor2 = nn.Linear(config.hidden_size, self.num_heads * self.rank_K, bias=False)
        self.V_factor1 = nn.Linear(config.hidden_size, self.num_heads * self.rank_V, bias=False)
        self.V_factor2 = nn.Linear(config.hidden_size, self.num_heads * self.rank_V, bias=False)

        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.kv_cache = None

    def _tensor_product(self, x, factor1, factor2):
        f1 = factor1(x).view(x.shape[0], -1, self.num_heads, self.rank_Q)
        f2 = factor2(x).view(x.shape[0], -1, self.num_heads, self.rank_Q)
        return torch.einsum('bnhd,bnhk->bnhdk', f1, f2).reshape(
            x.shape[0], -1, self.num_heads, self.rank_Q * self.rank_Q
        )

    def forward(self, hidden_states, attention_mask=None, past_kv=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self._tensor_product(hidden_states, self.Q_factor1, self.Q_factor2)
        K = self._tensor_product(hidden_states, self.K_factor1, self.K_factor2)
        V = self._tensor_product(hidden_states, self.V_factor1, self.V_factor2)

        Q = Q.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)

        if self.use_cache and past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)
        if self.use_cache:
            self.kv_cache = (K, V)

        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, V)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.all_head_size)
        return (self.out_proj(context), self.kv_cache) if kwargs.get('output_attentions') else (self.out_proj(context), self.kv_cache)