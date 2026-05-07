import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("regla")
class ReGLAAttention(BaseAttention):
    """
    Refined Gated Linear Attention (ReGLA) from NAACL 2025.
    Gated linear attention with proper normalization.
    """
    def __init__(self, config, rank=64, kernel_size=3):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.heads = config.num_attention_heads
        self.head_dim = self.dim // self.heads

        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.gate_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.conv = nn.Conv1d(self.dim, self.dim, kernel_size, padding=kernel_size//2, groups=self.dim)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.heads, self.head_dim).transpose(1,2)  # (B, H, L, d)
        k = self.k_proj(x).view(B, L, self.heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, L, self.heads, self.head_dim).transpose(1,2)
        g = self.gate_proj(x).view(B, L, self.heads, self.head_dim).transpose(1,2)

        # element‑wise gate on keys
        k_gated = k * torch.sigmoid(g)

        # linear attention with proper normalisation
        kv = torch.matmul(k_gated.transpose(-2, -1), v)  # (B, H, d, d)
        z = k_gated.sum(dim=2, keepdim=True)             # (B, H, 1, d)

        num = torch.matmul(q, kv)                        # (B, H, L, d)
        den = torch.matmul(q, z.transpose(-2, -1)) + 1e-8  # (B, H, L, 1)
        out = num / den

        out = out.transpose(1,2).contiguous().view(B, L, D)

        # Local depthwise conv enhancement
        loc = self.conv(x.permute(0,2,1)).permute(0,2,1)
        return self.out_proj(out + loc), None