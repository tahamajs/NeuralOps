import torch
import torch.nn as nn
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("zeros")
class ZeroSAttention(BaseAttention):
    """
    Zero‑mean linear attention: removes mean of keys and uses linear attention with proper normalisation.
    """
    def __init__(self, config, rank=64):
        super().__init__(config, rank)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.W_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.W_k(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)

        # Zero center keys along sequence dimension
        k_centered = k - k.mean(dim=2, keepdim=True)

        # Linear attention: O = Q (K^T V) / (Q sum(K))
        q_nonneg = torch.relu(q) + 1e-8
        k_nonneg = torch.relu(k_centered) + 1e-8

        kv = torch.einsum('bhnk,bhnd->bhkd', k_nonneg, v)   # (B, H, k, d) where k=d
        k_sum = k_nonneg.sum(dim=2, keepdim=True)            # (B, H, 1, d)

        num = torch.matmul(q_nonneg, kv)                     # (B, H, L, d)
        den = torch.matmul(q_nonneg, k_sum.transpose(-2,-1)) + 1e-8
        out = num / den
        out = out.permute(0,2,1,3).contiguous().view(B, L, D)
        return self.out(out), None