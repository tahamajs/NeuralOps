import torch
import torch.nn as nn
import torch.nn.functional as F
from .samba_m3 import SAMBAM3Block   # or from ..attentions.samba_m3

class REAPMoE(nn.Module):
    """Token‑choice MoE with top‑2 gating (REAP compression)."""
    def __init__(self, d_model, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model*4, d_model)
        ) for _ in range(n_experts)])
        self.router = nn.Linear(d_model, n_experts)

    def forward(self, x):
        B, L, D = x.shape
        scores = self.router(x)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            mask = (indices == e).any(dim=-1)
            if mask.any():
                selected_weights = (weights * (indices == e).float()).sum(dim=-1, keepdim=True)
                out += selected_weights * self.experts[e](x * mask.unsqueeze(-1))
        return out

class SambaMoEBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_experts=8, window=128):
        super().__init__()
        self.swa_mamba = SAMBAM3Block(d_model, n_heads, window=window)
        self.moe = REAPMoE(d_model, n_experts)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.swa_mamba(x)
        x = self.norm(x + self.moe(x))
        return x