import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY
from ..ssm.gated_delta_net import GatedDeltaNetLayer

@ATTENTION_REGISTRY.register("goldfinch_ahn")
class GoldFinchAHNBlock(BaseAttention):
    """
    GoldFinch‑AHN block: short‑term sliding window + long‑term GatedDeltaNet memory.
    """
    def __init__(self, config, rank=64, window_size=256):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.window_size = window_size

        # Short‑term memory: sliding window attention
        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)

        # Long‑term memory: GatedDeltaNet (assume it returns (B,L,D))
        self.delta_mem = GatedDeltaNetLayer(self.dim, use_spectral_norm=True)

        # Fusion gate
        self.gate = nn.Linear(self.dim, 1)
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x, mask=None, **kwargs):
        B, L, D = x.shape

        # Short‑term sliding window attention (causal, fixed window)
        q = self.W_q(x)  # (B, L, D)
        k = self.W_k(x)
        v = self.W_v(x)
        scores = torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5)   # (B, L, L)

        # Causal sliding window mask: allow j <= i and j > i - window_size
        idx = torch.arange(L, device=x.device)
        causal_mask = idx.unsqueeze(1) >= idx.unsqueeze(0)                  # (L, L)
        window_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)) < self.window_size
        mask = ~(causal_mask & window_mask)                                 # True where to mask
        scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
        short_out = torch.bmm(F.softmax(scores, dim=-1), v)

        # Long‑term memory via GatedDeltaNet
        long_out = self.delta_mem(x)   # (B, L, D)

        # Fusion gate
        gate = torch.sigmoid(self.gate(x))   # (B, L, 1)
        out = self.out_proj(gate * short_out + (1 - gate) * long_out)
        return (out, None)