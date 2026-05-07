import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

class Mamba2(nn.Module):
    """Simplified Mamba‑2 stub (replace with official mamba_ssm)."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        inner_dim = d_model * expand
        self.in_proj = nn.Linear(d_model, inner_dim * 2)
        self.conv1d = nn.Conv1d(inner_dim, inner_dim, d_conv, groups=inner_dim, padding=d_conv-1)
        self.dt_proj = nn.Linear(inner_dim, d_model)
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) * 0.1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(inner_dim, d_model)

    def forward(self, x):
        B, L, D = x.shape
        x_proj = self.in_proj(x)            # (B, L, 2*inner)
        x_ssm, z = x_proj.chunk(2, dim=-1)  # each (B, L, inner)
        x_ssm = self.conv1d(x_ssm.transpose(1,2))[..., :L].transpose(1,2)
        dt = F.softplus(self.dt_proj(x_ssm))   # (B, L, D)
        A = -torch.exp(self.A_log)             # (D, d_state)
        # Recurrent scan (slow but functional)
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            xt = x_ssm[:, t, :self.d_model]   # take first d_model channels as input
            h = h * torch.exp(dt[:, t].unsqueeze(-1) * A) + (xt.unsqueeze(-1) * dt[:, t].unsqueeze(-1))
            out_t = (h * self.D.unsqueeze(-1)).sum(dim=-1)
            outputs.append(out_t)
        out = torch.stack(outputs, dim=1) * F.silu(z)
        return self.out_proj(out)

class MixtureOfExperts(nn.Module):
    """Top‑k token‑choice MoE."""
    def __init__(self, dim, ff_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim)
        ) for _ in range(num_experts)])
        self.router = nn.Linear(dim, num_experts)

    def forward(self, x):
        B, L, D = x.shape
        scores = self.router(x)            # (B, L, E)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = (indices == e).any(dim=-1)   # (B, L)
            if mask.any():
                selected_weights = (weights * (indices == e).float()).sum(dim=-1, keepdim=True)
                out += mask.unsqueeze(-1) * selected_weights * self.experts[e](x * mask.unsqueeze(-1))
        return out

@ATTENTION_REGISTRY.register("granite_hybrid")
class GraniteHybridBlock(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2, ratio=9):
        super().__init__()
        self.mamba = Mamba2(d_model=dim)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.moe = MixtureOfExperts(dim, dim * 4, num_experts, top_k)
        self.ratio = ratio
        self.register_buffer('counter', torch.tensor(0, dtype=torch.long))   # use buffer for state

    def forward(self, x):
        self.counter += 1
        if self.counter % (self.ratio + 1) < self.ratio:
            x = self.mamba(x)
        else:
            x, _ = self.attn(x, x, x)
        x = x + self.moe(x)
        return x