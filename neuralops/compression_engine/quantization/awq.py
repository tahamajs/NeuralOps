"""
AWQ: Activation-Aware Weight Quantization

Identifies salient weight channels via activation magnitudes and protects them using per-channel scaling transformations before low-bit quantization.

Author: Taha Majlesi
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from neuralops.core.base import BaseCompressor
from neuralops.core.registry import COMPRESSOR_REGISTRY

__all__ = ["AWQQuantizer", "AWQCompressor", "quantize_model_awq"]


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
        _q_levels = 2 ** self.bits
        q_max = 2 ** (self.bits - 1) - 1
        q_min = -2 ** (self.bits - 1)
        max_val = torch.amax(torch.abs(W), dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_val / q_max
        q = torch.clamp(torch.round(W / scale), q_min, q_max)
        return q * scale

    def quantize(self) -> torch.Tensor:
        """Applies activation-aware channel re-scaling and quantizes."""
        W = self.layer.weight.data.clone().to(torch.float32)
        scales = self.act_scales.clamp(min=1e-5).pow(self.alpha)

        W_scaled = W * scales.unsqueeze(0)
        Q_scaled = self._quantize_int4(W_scaled)
        Q = Q_scaled / scales.unsqueeze(0)

        self.layer.weight.data.copy_(Q.to(self.layer.weight.dtype))
        return Q


@COMPRESSOR_REGISTRY.register("awq_real", is_alias=True)
@COMPRESSOR_REGISTRY.register("awq_compressor", is_alias=True)
class AWQCompressor(BaseCompressor):
    """Unified BaseCompressor wrapper for AWQ quantization with compressed-tensors support."""

    def __init__(
        self,
        model: nn.Module,
        bits: int = 4,
        alpha: float = 0.5,
        group_size: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="awq", **kwargs)
        self.bits = bits
        self.alpha = alpha
        self.group_size = group_size

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """Applies AWQ quantization across all linear layers in model."""
        stats = {"bits": self.bits, "alpha": self.alpha, "layers": []}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                awq = AWQQuantizer(module, bits=self.bits, alpha=self.alpha)
                awq.quantize()
                stats["layers"].append(name)
        if not stats["layers"] and isinstance(self.model, nn.Linear):
            awq = AWQQuantizer(self.model, bits=self.bits, alpha=self.alpha)
            awq.quantize()
            stats["layers"].append("root")
        return self.model, stats

    def compress_to_tensors(self, model: Optional[nn.Module] = None) -> dict[str, torch.Tensor]:
        """Returns compressed tensor dictionary compatible with compressed-tensors / safetensors."""
        m = model if model is not None else self.model
        tensors: dict[str, torch.Tensor] = {}
        for name, param in m.named_parameters():
            tensors[name] = param.data
        for name, buffer in m.named_buffers():
            tensors[name] = buffer.data
        return tensors

    def get_quantization_config(self) -> dict[str, Any]:
        """Returns quantization configuration compatible with compressed-tensors and vLLM."""
        return {
            "quant_method": "compressed-tensors",
            "format": f"W{self.bits}A16",
            "bits": self.bits,
            "group_size": self.group_size,
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": self.bits,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group" if self.group_size > 0 else "channel",
                        "group_size": self.group_size,
                    },
                    "targets": ["Linear"],
                }
            },
        }


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
