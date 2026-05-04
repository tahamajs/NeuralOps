# framework/attentions/ssm/unified_rope.py
"""
Rotary Position Embedding (RoPE) – pre‑computes cos/sin tables.
Based on RoFormer (https://arxiv.org/abs/2104.09864)
"""

import torch
import torch.nn as nn
from ..core.registry import AUX_REGISTRY


@AUX_REGISTRY.register("unified_rope")
class UnifiedRoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(2))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(2))

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {dim}")
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        cos = self.cos_cached[:, offset:offset+seq_len]
        sin = self.sin_cached[:, offset:offset+seq_len]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_odd * cos + x_even * sin
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)