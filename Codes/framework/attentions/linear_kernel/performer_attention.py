"""
Performer Attention: Fast Attention Via Positive Orthogonal Random features (FAVOR+)
Paper: https://arxiv.org/abs/2009.14794
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


@ATTENTION_REGISTRY.register("performer")
class PerformerAttention(BaseAttention):
    def __init__(self, config, rank=256, redraw_projection=True):
        super().__init__(config, rank)   # rank is number of random features
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank
        self.redraw_projection = redraw_projection

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Random projection matrix (one per head) will be created on first forward
        self.register_buffer('projection', None)

    def _reshape(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)   # (B, H, N, D)

    def _get_projection_matrix(self, device):
        # Create orthogonal random matrix of shape (head_dim, rank) per head
        if self.projection is None or self.redraw_projection:
            # Generate a separate matrix per head (H, D, R)
            proj = []
            for _ in range(self.num_heads):
                ortho = torch.randn(self.head_dim, self.rank, device=device)
                q, _ = torch.linalg.qr(ortho)
                proj.append(q)
            self.projection = torch.stack(proj, dim=0)  # (H, D, R)
        return self.projection

    def _feature_map(self, x, proj):
        # x: (B, H, N, D), proj: (H, D, R)
        scale = 1.0 / (self.head_dim ** 0.25)
        x_proj = torch.einsum('bhnd,hdr->bhnr', x, proj) * scale
        phi = torch.relu(x_proj)  # using ReLU as positive feature map
        # normalization is handled by the linear attention denominator
        return phi

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self._reshape(self.query(hidden_states), batch_size)   # (B, H, N, D)
        K = self._reshape(self.key(hidden_states), batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)

        proj = self._get_projection_matrix(Q.device)  # (H, D, R)

        Q_prime = self._feature_map(Q, proj)   # (B, H, N, R)
        K_prime = self._feature_map(K, proj)   # (B, H, N, R)

        # Linear attention
        KV = torch.einsum('bhnr,bhnd->bhrd', K_prime, V)    # (B, H, R, D)
        K_sum = K_prime.sum(dim=2, keepdim=True)            # (B, H, 1, R)
        num = torch.einsum('bhnr,bhrd->bhnd', Q_prime, KV)
        den = torch.einsum('bhnr,bhr->bhn', Q_prime, K_sum.squeeze(2)).unsqueeze(-1) + 1e-8
        context = num / den

        if attention_mask is not None:
            mask = attention_mask.view(batch_size, 1, 1, seq_len)
            # mask padding positions by zeroing their output (optional)
            context = context * mask.unsqueeze(-1).to(context.dtype)

        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.all_head_size)
        return (self.out_proj(context), None) if kwargs.get('output_attentions') else (self.out_proj(context),)