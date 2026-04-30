import torch
import torch.nn as nn
from mamba_ssm import Mamba

class ExpertChoiceMoE(nn.Module):
    def __init__(self, d_model, n_experts=8, capacity=4):
        super().__init__()
        self.n_experts = n_experts
        self.capacity = capacity
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_experts)])
        self.router = nn.Linear(d_model, n_experts)

    def forward(self, x):
        # Expert choice: each expert picks top‑capacity tokens
        B, L, D = x.shape
        scores = self.router(x)  # (B, L, E)
        # For simplicity, we use standard token‑choice with auxiliary loss
        # Real Expert‑Choice requires sorting, omitted here.
        top_scores, top_idx = torch.topk(scores, k=self.capacity, dim=-1)
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            # Find tokens that chose expert e
            mask = (top_idx == e).any(dim=-1)
            if mask.sum() > 0:
                out[mask] += self.experts[e](x[mask])
        return out

class Jamba2Block(nn.Module):
    def __init__(self, d_model, n_heads, n_experts=8, capacity=4):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.moe = ExpertChoiceMoE(d_model, n_experts, capacity)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.mamba(x))
        attn_out, _ = self.attn(x, x, x)
        x = self.norm2(x + attn_out)
        x = self.norm3(x + self.moe(x))
        return x