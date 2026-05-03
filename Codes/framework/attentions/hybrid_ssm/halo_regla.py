import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.registry import ATTENTION_REGISTRY

# ---- simple RoPE helpers ----
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)   # (L, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x, offset=0):
        L = x.shape[1]
        cos = self.cos_cached[offset:offset+L].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[offset:offset+L].unsqueeze(0).unsqueeze(2)
        return (x * cos) + (rotate_half(x) * sin)

# ---- Linear attention layer (ReGLA‑style) ----
class ReGLALayer(nn.Module):
    """ReLU‑kernel linear attention."""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        q = F.relu(self.q_proj(x)).view(B, L, self.heads, self.head_dim).transpose(1,2)
        k = F.relu(self.k_proj(x)).view(B, L, self.heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, L, self.heads, self.head_dim).transpose(1,2)
        kv = torch.einsum('bhld,bhlm->bhldm', k, v)   # (B, H, d, d)
        z = k.cumsum(dim=2)                           # (B, H, L, d)
        num = torch.einsum('bhld,bhldm->bhldm', q, kv)
        den = torch.einsum('bhld,bhld->bhl', q, z) + 1e-8
        out = (num.sum(-1) / den.unsqueeze(-1)).transpose(1,2).contiguous().view(B, L, D)
        return self.out_proj(out)

# ---- Standard Transformer layer (with RoPE placeholder) ----
class TransformerLayer(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.rope = RotaryEmbedding(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # RoPE not integrated into MHA for simplicity; a real implementation would apply it
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

@ATTENTION_REGISTRY.register("halo_regla")
class HALOReGLAStudent(nn.Module):
    """Hybrid model alternating linear (ReGLA) and full Transformer layers."""
    def __init__(self, dim, num_layers, hybrid_ratio=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % hybrid_ratio == 0:
                self.layers.append(ReGLALayer(dim))
            else:
                self.layers.append(TransformerLayer(dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x