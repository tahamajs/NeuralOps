# framework/compression/spectraformer_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_attention import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("spectra")
class SpectraformerAttention(BaseAttention):
    """
    Spectraformer: Random feature linearization with flexible kernel approximation.
    Paper: https://arxiv.org/abs/2103.04246 (placeholder)
    """
    def __init__(self, config, rank=256):
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

    def _reshape(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self._reshape(self.query(hidden_states), batch_size)
        K = self._reshape(self.key(hidden_states), batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)

        # Random feature map (using sin/cos with random orthogonal matrix)
        R = torch.randn(self.head_dim, self.rank, device=hidden_states.device)
        Q_proj = torch.einsum('bqhd,dr->bqhr', Q, R) / (self.head_dim**0.5)
        K_proj = torch.einsum('bkhd,dr->bkhr', K, R) / (self.head_dim**0.5)
        # Apply kernel approximation (e.g., positive random features)
        Q_proj = torch.exp(Q_proj)
        K_proj = torch.exp(K_proj)

        KV = torch.einsum('bkhr,bkhd->bhrd', K_proj, V)
        context = torch.einsum('bqhr,bhrd->bqhd', Q_proj, KV)
        Z = 1.0 / (torch.einsum('bqhr,bkhr->bqk', Q_proj, K_proj.sum(dim=2)) + 1e-6)
        context = context * Z.unsqueeze(-1)

        context = context.permute(0,2,1,3).contiguous().view(batch_size, seq_len, self.all_head_size)
        return self.out_proj(context), None