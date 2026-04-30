# framework/compression/galerkin_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_attention import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("galerkin")
class GalerkinAttention(BaseAttention):
    """
    Galerkin Transformer: Replace softmax attention with Petrov‑Galerkin projection.
    Paper: https://arxiv.org/abs/2105.14906
    Code: https://github.com/hsliuustc/galerkin-transformer
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

    def _reshape(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (B, H, N, D)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        Q = self._reshape(self.query(hidden_states), batch_size)
        K = self._reshape(self.key(hidden_states), batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)

        # Galerkin projection: K and V are projected onto a subspace
        # of dimension rank using a random sketching matrix.
        sketch = torch.randn(seq_len, self.rank, device=hidden_states.device) / seq_len**0.5
        K_sketch = torch.einsum('bqhd,sk->bqhd', K, sketch)   # (B, H, N, R) after einsum? Wait careful.
        # Better: use einsum 'bnhd,nr->bhrd'? Let's do:
        K_sketch = torch.einsum('bnhd,nr->bhrd', K, sketch)   # (B, H, R, D)
        V_sketch = torch.einsum('bnhd,nr->bhrd', V, sketch)   # (B, H, R, D)

        # Compute attention scores in the reduced space
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.einsum('bqhd,bhrd->bqhr', Q, K_sketch) * scale   # (B, H, N, R)

        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1).unsqueeze(-1)

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.einsum('bqhr,bhrd->bqhd', probs, V_sketch)   # (B, H, N, D)

        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.all_head_size)

        return self.out_proj(context), None
    