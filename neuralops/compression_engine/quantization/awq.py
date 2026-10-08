"""Activation-aware Weight Quantization (AWQ)."""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["AWQQuantizer"]


class AWQQuantizer:
    """Protects salient weights by observing activation magnitudes."""

    def __init__(self, layer: nn.Linear, bits: int = 4) -> None:
        self.layer = layer
        self.bits = bits
        self.act_scales = torch.zeros(layer.weight.shape[1], device=layer.weight.device, dtype=torch.float32)

    def observe(self, x: torch.Tensor) -> None:
        """Observes channel-wise maximum absolute activation values."""
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        s = torch.mean(torch.abs(x.to(torch.float32)), dim=0)
        self.act_scales = torch.maximum(self.act_scales, s)

    def quantize(self, salient_ratio: float = 0.01) -> torch.Tensor:
        """Applies activation-aware channel re-scaling and quantizes."""
        W = self.layer.weight.data.clone().to(torch.float32)
        scales = (self.act_scales.clamp(min=1e-5)).pow(0.5)

        # Scale weights to protect salient channels
        W_scaled = W * scales.unsqueeze(0)
        q_levels = 2 ** self.bits
        max_val = torch.amax(torch.abs(W_scaled), dim=-1, keepdim=True)
        step = (2 * max_val / (q_levels - 1)).clamp(min=1e-8)

        Q_scaled = torch.round(W_scaled / step) * step
        Q = Q_scaled / scales.unsqueeze(0)

        self.layer.weight.data.copy_(Q.to(self.layer.weight.dtype))
        return Q
