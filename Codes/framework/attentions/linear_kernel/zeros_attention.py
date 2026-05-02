import torch
import torch.nn as nn
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("zeros")
class ZeroSAttention(BaseAttention):
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
        # Zero‑sum: remove mean of K before computing attention
        k_centered = k - k.mean(dim=2, keepdim=True)
        # linear attention with zero‑sum property
        scale = self.head_dim ** -0.5
        kv = torch.matmul(k_centered.transpose(-2,-1) * scale, v)
        out = torch.matmul(q * scale, kv) / L
        out = out.permute(0,2,1,3).contiguous().view(B, L, D)
        return self.out(out), None