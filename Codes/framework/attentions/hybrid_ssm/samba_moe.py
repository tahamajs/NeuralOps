import torch
import torch.nn as nn
from samba_m3 import SAMBAM3Block  # reuse the block

class REAPMoE(nn.Module):
    """Router‑weighted Expert Activation Pruning for MoE compression."""
    def __init__(self, d_model, n_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_experts)])
        self.router = nn.Linear(d_model, n_experts)

    def forward(self, x):
        B, L, D = x.shape
        scores = self.router(x)  # (B, L, E)
        top_scores, top_idx = torch.topk(scores, k=2, dim=-1)
        out = torch.zeros_like(x)
        for i in range(2):
            mask = (top_idx[:, :, i] == i).unsqueeze(-1)
            out = out + mask * self.experts[i](x)
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