# framework/attentions/kda_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class KDAAttention(nn.Module):
    def __init__(self, dim, num_heads, chunk_size=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.beta = nn.Parameter(torch.ones(num_heads, 1))  # per-channel decay
        self.chunk_size = chunk_size

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = F.normalize(self.k_proj(x).view(B, L, self.num_heads, self.head_dim), dim=-1)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        # Simplified recurrent scan (real KDA uses chunkwise parallel)
        h = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)
        out = []
        for t in range(L):
            qt = q[:, t]      # (B, H, D)
            kt = k[:, t]
            vt = v[:, t]
            decay = torch.sigmoid(self.beta).unsqueeze(0)   # (1, H, 1)
            h = decay * h + (kt.unsqueeze(-1) @ vt.unsqueeze(2))
            out_t = (h @ qt.unsqueeze(-1)).squeeze(-1)
            out.append(out_t)
        out = torch.stack(out, dim=1).view(B, L, D)
        return self.out_proj(out)