# framework/attentions/regla_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("regla")
class ReGLAAttention(BaseAttention):
    """
    Refined Gated Linear Attention (ReGLA) from NAACL 2025.
    Replaces quadratic attention.
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
        q = self.q_proj(x).view(B, L, self.heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B, L, self.heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, L, self.heads, self.head_dim).transpose(1,2)
        g = self.gate_proj(x).view(B, L, self.heads, self.head_dim).transpose(1,2)

        # Causal linear attention with gating
        kv = torch.matmul(k.transpose(-2, -1) * g.sigmoid(), v)   # (B, H, D, D)
        out = torch.matmul(q, kv) / L
        out = out.transpose(1,2).contiguous().view(B, L, D)

        # Local enhancement
        loc = self.conv(x.permute(0,2,1)).permute(0,2,1)
        return self.out_proj(out + loc), None