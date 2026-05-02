import torch
import torch.nn as nn
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("hla")
class HLAAttention(BaseAttention):
    def __init__(self, config, rank=64):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.dim = config.hidden_size
        self.W_q = nn.Linear(self.dim, self.dim)
        self.W_k = nn.Linear(self.dim, self.dim)
        self.W_v = nn.Linear(self.dim, self.dim)
        self.out = nn.Linear(self.dim, self.dim)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim)
        k = self.W_k(x).view(B, L, self.num_heads, self.head_dim)
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim)
        # Hadamard product based attention
        # Simplified: use element‑wise product and softmax
        scores = torch.einsum('bhid,bhjd->bhij', q, k) / (self.head_dim**0.5)
        # Apply Hadamard gate?
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.reshape(B, L, D)
        return self.out(out), None