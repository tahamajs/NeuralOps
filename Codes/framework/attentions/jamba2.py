import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("mamba")
class Mamba(nn.Module):
    """Trivial Mamba placeholder – replace with official implementation."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, d_model*2)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.out_proj(F.silu(self.in_proj(x)))

@ATTENTION_REGISTRY.register("expert_choice_moe")
class ExpertChoiceMoE(nn.Module):
    """Top‑2 token‑choice MoE."""
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

@ATTENTION_REGISTRY.register("jamba2")
class Jamba2Block(nn.Module):
    def __init__(self, d_model, n_heads, n_experts=8, top_k=2):
        super().__init__()
        self.mamba = Mamba(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.moe = ExpertChoiceMoE(d_model, n_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x,**kwargs):
        x = self.norm1(x + self.mamba(x))
        attn_out, _ = self.attn(x, x, x)
        x = self.norm2(x + attn_out)
        x = self.norm3(x + self.moe(x))
        return x