import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_attention import BaseAttention
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

        # Long‑term memory: GatedDeltaNet
        self.delta_mem = GatedDeltaNetLayer(self.dim, use_spectral_norm=True)

        # Fusion gate
        self.gate = nn.Linear(self.dim, 1)
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x, mask=None, **kwargs):
        B, L, D = x.shape

        # Short‑term sliding window attention
        q = self.W_q(x).view(B, L, D)
        k = self.W_k(x).view(B, L, D)
        v = self.W_v(x).view(B, L, D)
        # Apply causal sliding window (simplified: full attention masked to window)
        attn = torch.bmm(q, k.transpose(1,2)) / (D ** 0.5)
        window_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=-self.window_size).bool()
        attn = attn.masked_fill(window_mask.unsqueeze(0), float('-inf'))
        short_out = torch.bmm(F.softmax(attn, dim=-1), v)

        # Long‑term memory via GatedDeltaNet
        long_out = self.delta_mem(x)   # (B, L, D)

        # Fusion
        gate = torch.sigmoid(self.gate(x))   # (B, L, 1)
        out = self.out_proj(gate * short_out + (1 - gate) * long_out)
        return (out, None)