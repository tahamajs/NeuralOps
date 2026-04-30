# framework/compression/tlinformer_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_attention import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("tlinformer")
class TLinFormerAttention(BaseAttention):
    """
    TLinFormer: Exact linear-time attention by reconfiguring neuron connections.
    Paper: https://arxiv.org/abs/2403.12345 (placeholder)
    """
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
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Learnable low-rank projections for K and V
        self.E = nn.Linear(self.all_head_size, rank)
        self.F = nn.Linear(self.all_head_size, rank)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        Q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Flatten heads for projection
        Q_flat = Q.permute(0,2,1,3).contiguous().view(batch_size*self.num_heads, seq_len, self.head_dim)
        K_flat = K.permute(0,2,1,3).contiguous().view(batch_size*self.num_heads, seq_len, self.head_dim)
        V_flat = V.permute(0,2,1,3).contiguous().view(batch_size*self.num_heads, seq_len, self.head_dim)

        K_proj = self.E(K_flat).view(batch_size*self.num_heads, seq_len, self.rank)
        V_proj = self.F(V_flat).view(batch_size*self.num_heads, seq_len, self.rank)

        # Attention scores
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.bmm(Q_flat, K_proj.transpose(1,2)) * scale
        if attention_mask is not None:
            mask = attention_mask.view(batch_size, 1, seq_len, 1).expand(-1, self.num_heads, -1, -1)
            mask = mask.reshape(batch_size*self.num_heads, seq_len, 1)
            scores = scores + mask

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.bmm(probs, V_proj)
        context = context.view(batch_size, self.num_heads, seq_len, self.head_dim)
        context = context.permute(0,2,1,3).contiguous().view(batch_size, seq_len, self.all_head_size)
        return self.out_proj(context), None