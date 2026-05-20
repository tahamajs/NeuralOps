import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("kda")
class KDAAttention(nn.Module):
    def __init__(self, config, rank=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.dim = config.hidden_size
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.beta = nn.Parameter(torch.ones(self.num_heads, 1))

    def forward(self, x, **kwargs):
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim
        q = self.q_proj(x).view(B, L, H, d).transpose(1,2)
        k = F.normalize(self.k_proj(x).view(B, L, H, d).transpose(1,2), dim=-1)
        v = self.v_proj(x).view(B, L, H, d).transpose(1,2)
        h = torch.zeros(B, H, d, d, device=x.device)
        outputs = []
        for t in range(L):
            qt = q[:, :, t, :]
            kt = k[:, :, t, :]
            vt = v[:, :, t, :]
            decay = torch.sigmoid(self.beta).view(1, H, 1, 1)
            h = decay * h + (kt.unsqueeze(-1) @ vt.unsqueeze(1))
            out_t = (h @ qt.unsqueeze(-1)).squeeze(-1)
            outputs.append(out_t)
        out = torch.stack(outputs, dim=2)   # (B, H, L, d)
        out = out.transpose(1,2).contiguous().view(B, L, D)
        return self.out_proj(out), None
    