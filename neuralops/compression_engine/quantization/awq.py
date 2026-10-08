"""Production-grade Activation-aware Weight Quantization (AWQ)."""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["AWQQuantizer", "quantize_model_awq"]


class AWQQuantizer:
    """Protects salient weights by observing activation magnitudes and re-scaling channels."""

    def __init__(self, layer: nn.Linear, bits: int = 4, alpha: float = 0.5) -> None:
        self.layer = layer
        self.bits = bits
        self.alpha = alpha
        self.act_scales = torch.zeros(layer.weight.shape[1], device=layer.weight.device, dtype=torch.float32)

    def observe(self, x: torch.Tensor) -> None:
        """Observes per-channel activation scales from calibration batches."""
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        s = torch.mean(torch.abs(x.to(torch.float32)), dim=0)
        self.act_scales = torch.maximum(self.act_scales, s)

    def _quantize_int4(self, W: torch.Tensor) -> torch.Tensor:
        """Symmetric per-channel INT4 quantization grid."""
        q_levels = 2 ** self.bits
        q_max = 2 ** (self.bits - 1) - 1
        q_min = -2 ** (self.bits - 1)
        max_val = torch.amax(torch.abs(W), dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_val / q_max
        q = torch.clamp(torch.round(W / scale), q_min, q_max)
        return q * scale

    def quantize(self) -> torch.Tensor:
        """Applies activation-aware channel re-scaling and quantizes."""
        W = self.layer.weight.data.clone().to(torch.float32)
        # Protect salient channels using activation magnitude: s = act^alpha
        scales = self.act_scales.clamp(min=1e-5).pow(self.alpha)

        # Re-scale weights: W' = W * diag(s)
        W_scaled = W * scales.unsqueeze(0)
        Q_scaled = self._quantize_int4(W_scaled)
        # Restore scaling: Q = Q' / diag(s)
        Q = Q_scaled / scales.unsqueeze(0)

        self.layer.weight.data.copy_(Q.to(self.layer.weight.dtype))
        return Q


def quantize_model_awq(
    model: nn.Module,
    calibration_data: list[torch.Tensor],
    bits: int = 4,
    alpha: float = 0.5,
) -> nn.Module:
    """Applies AWQ quantization across all linear layers in model."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            awq = AWQQuantizer(module, bits=bits, alpha=alpha)
            for batch in calibration_data:
                awq.observe(batch)
            awq.quantize()
    return model
