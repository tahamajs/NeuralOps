import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

# Placeholder for GatedDeltaNetLayer in case the real module is missing
class GatedDeltaNetLayer(nn.Module):
    """Stub for GatedDeltaNetLayer – replace with actual implementation."""
    def __init__(self, d_model, use_spectral_norm=False):
        super().__init__()
        self.net = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.net(x)

@ATTENTION_REGISTRY.register("goldfinch_ahn")
class GoldFinchAHNBlock(BaseAttention):
    """
    GoldFinch‑AHN block: short‑term sliding window + long‑term GatedDeltaNet memory.
    """
    def __init__(self, config, rank=64, window_size=256):
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.window_size = window_size

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.delta_mem = GatedDeltaNetLayer(self.dim, use_spectral_norm=True)
        self.gate = nn.Linear(self.dim, 1)
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x, mask=None, **kwargs):
        B, L, D = x.shape

        # Short‑term sliding window (causal)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        scores = torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5)

        idx = torch.arange(L, device=x.device)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        window = (idx.unsqueeze(1) - idx.unsqueeze(0)) < self.window_size
        mask_val = ~(causal & window)
        scores = scores.masked_fill(mask_val.unsqueeze(0), float('-inf'))
        short_out = torch.bmm(F.softmax(scores, dim=-1), v)

        long_out = self.delta_mem(x)          # (B, L, D)
        gate = torch.sigmoid(self.gate(x))    # (B, L, 1)
        out = self.out_proj(gate * short_out + (1 - gate) * long_out)
        return (out, None)