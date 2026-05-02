import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("chela_slay")
class ChelaSlayAttention(BaseAttention):
    """
    CHELA‑SLAY: Spherical linearized attention with Yat‑kernel.
    Constrains Q,K to unit sphere, uses random Fourier features.
    """
    def __init__(self, config, rank=64, feature_dim=256):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.feature_dim = feature_dim

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim)

        # Yat kernel random features (fixed)
        self.register_buffer("random_proj", torch.randn(self.head_dim, feature_dim))

    def _spherical_normalize(self, x):
        return F.normalize(x, p=2, dim=-1)

    def _yat_feature_map(self, x):
        # x: (..., head_dim) -> (..., feature_dim)
        proj = torch.matmul(x, self.random_proj)   # (..., feature_dim)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) * (1.0 / self.feature_dim**0.25)

    def forward(self, x, mask=None, **kwargs):
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.W_k(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)

        # Spherical normalization
        q_norm = self._spherical_normalize(q)
        k_norm = self._spherical_normalize(k)

        # Apply Yat‑kernel feature map
        q_feat = self._yat_feature_map(q_norm)   # (B, H, L, 2*feature_dim)
        k_feat = self._yat_feature_map(k_norm)

        # Linear attention: O = φ(Q) ( φ(K)^T V ) / (φ(Q) sum(φ(K)))
        kv = torch.matmul(k_feat.transpose(-2, -1), v)   # (B, H, 2F, D_h)
        Z = 1.0 / torch.matmul(q_feat, k_feat.sum(dim=-2, keepdim=True).unsqueeze(-1))
        out = torch.matmul(q_feat, kv) * Z   # (B, H, L, D_h)
        out = out.permute(0,2,1,3).contiguous().view(B, L, D)
        return self.out_proj(out), None