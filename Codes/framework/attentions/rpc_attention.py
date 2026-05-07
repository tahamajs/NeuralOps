import torch
import torch.nn as nn
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("rpc")
class RPCAttention(BaseAttention):
    """
    Attention derived from Kernel PCA: projects queries onto principal components of keys
    via a random Gaussian projection matrix.
    """
    def __init__(self, config, rank=64):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.dim = config.hidden_size

        self.W_q = nn.Linear(self.dim, self.dim)
        self.W_k = nn.Linear(self.dim, self.dim)
        self.W_v = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)

        # Fixed random projection matrix (one per head)
        proj = torch.randn(self.num_heads, self.head_dim, self.head_dim)
        self.register_buffer('proj', proj)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.W_k(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)

        # Project keys and queries using the same per‑head random matrix
        proj = self.proj  # (H, D, D)
        k_feat = torch.einsum('bhld,hde->bhle', k, proj)
        q_feat = torch.einsum('bhld,hde->bhle', q, proj)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q_feat, k_feat.transpose(-2,-1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0,2,1,3).contiguous().view(B, L, D)
        return self.out_proj(out), None