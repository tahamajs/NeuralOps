import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("mixattention_moska")
class MixAttentionMoSKA(BaseAttention):
    """
    MixAttention‑MoSKA: batch‑shared KV as GEMM (simplified) and FASA‑pruning
    for unique sequences.
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
        # k: (B, H, L, D_h)
        B, H, L, D_h = k.shape
        # Token importance = variance across time dimension (mean over heads)
        importance = k.var(dim=-1).mean(dim=1)   # (B, L)
        keep = max(1, int(L * self.fasa_keep_ratio))
        _, idx = torch.topk(importance, keep, dim=-1)
        idx = idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D_h)
        k_pruned = k.gather(2, idx)
        v_pruned = v.gather(2, idx)
        return k_pruned, v_pruned

    def forward(self, x, mask=None, **kwargs):
        B, L, D = x.shape
        H = self.num_heads
        q = self.W_q(x).view(B, L, H, self.head_dim).permute(0,2,1,3)
        k = self.W_k(x).view(B, L, H, self.head_dim).permute(0,2,1,3)
        v = self.W_v(x).view(B, L, H, self.head_dim).permute(0,2,1,3)

        k_p, v_p = self._fasa_prune(k, v)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k_p.transpose(-2, -1)) * scale
        if mask is not None:
            # adapt mask to pruned length: simple approach – ignore, or gather mask similarly.
            raise NotImplementedError("mask handling with pruned length not implemented yet")
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_p).permute(0,2,1,3).contiguous().view(B, L, D)
        return self.out_proj(out), None
    