# framework/attentions/granite_hybrid.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba2

class GraniteHybridBlock(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2):
        super().__init__()
        self.mamba = Mamba2(d_model=dim)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.moe = MixtureOfExperts(dim, dim, num_experts, top_k)  # see helper below
        self.ratio = 9  # 9 mamba : 1 attn
        self.counter = 0

    def forward(self, x):
        if self.counter % (self.ratio+1) < self.ratio:
            x = self.mamba(x)
        else:
            x, _ = self.attn(x, x, x)
        x = self.moe(x)
        return x

class MixtureOfExperts(nn.Module):
    def __init__(self, dim, ff_dim, num_experts, top_k):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(dim, ff_dim) for _ in range(num_experts)])
        self.router = nn.Linear(dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        B, L, D = x.shape
        scores = self.router(x)  # (B, L, E)
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for e in range(len(self.experts)):
            mask = (topk_indices == e).any(dim=-1)
            if mask.sum() > 0:
                out[mask] += self.experts[e](x[mask])
        return out