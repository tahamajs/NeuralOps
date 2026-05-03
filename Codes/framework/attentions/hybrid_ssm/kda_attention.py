import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("kda")
class KDAAttention(nn.Module):
    """
    Kernelised‑Dynamic‑Decay Attention (KDA).
    Uses per‑channel decay rates in a linear recurrent scan.
    """
    def __init__(self, dim, num_heads, chunk_size=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.beta = nn.Parameter(torch.ones(num_heads, 1))   # per‑head decay
        self.chunk_size = chunk_size  # (not used in this simple loop)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = F.normalize(self.k_proj(x).view(B, L, self.num_heads, self.head_dim), dim=-1)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        # Recurrent scan
        h = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)
        outputs = []
        for t in range(L):
            qt = q[:, t]      # (B, H, d)
            kt = k[:, t]
            vt = v[:, t]
            decay = torch.sigmoid(self.beta).unsqueeze(0)   # (1, H, 1)
            # state update: H = decay * H + k.t() @ v
            h = decay.unsqueeze(-1) * h + kt.unsqueeze(-1) @ vt.unsqueeze(2)
            # output = H @ q
            out_t = (h @ qt.unsqueeze(-1)).squeeze(-1)   # (B, H, d) -> (B, H)
            outputs.append(out_t)
        out = torch.stack(outputs, dim=2)   # (B, H, L) -> need to reshape
        out = out.permute(0, 2, 1).contiguous().view(B, L, D)
        return self.out_proj(out)