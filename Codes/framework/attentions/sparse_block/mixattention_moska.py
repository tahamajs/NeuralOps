import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("mixattention_moska")
class MixAttentionMoSKA(BaseAttention):
    """
    MixAttention‑MoSKA: batch shared KV as GEMM and apply FASA‑pruning for unique sequences.
    """
    def __init__(self, config, rank=64, fasa_keep_ratio=0.189):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.fasa_keep_ratio = fasa_keep_ratio

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim)

    def _fasa_prune(self, k, v):
        # Frequency‑aware sparse attention pruning
        # k: (B, H, L, D_h)
        B, H, L, D_h = k.shape
        # Compute token importance (simplified: variance)
        importance = k.var(dim=-1).mean(dim=1)   # (B, L)
        keep = int(L * self.fasa_keep_ratio)
        _, idx = torch.topk(importance, keep, dim=-1)
        idx = idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D_h)
        k_pruned = k.gather(2, idx)
        v_pruned = v.gather(2, idx)
        return k_pruned, v_pruned

    def forward(self, x, mask=None, **kwargs):
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.W_k(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)

        # Assume we have a simple flag for shared sequence; here we always treat as unique.
        # In a real server, shared sequences would use batched GEMM.
        k_p, v_p = self._fasa_prune(k, v)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k_p.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_p).permute(0,2,1,3).contiguous().view(B, L, D)
        return self.out_proj(out), None