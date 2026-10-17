"""QuaRot (Quantization with Rotation) implementation for neuralops."""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

from neuralops.core.base import BaseCompressor
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.core.triton_kernels.elementwise import fast_hadamard_transform

__all__ = ["QuaRotCompressor", "quarot_factory"]


@COMPRESSOR_REGISTRY.register("quarot", is_alias=True)
@COMPRESSOR_REGISTRY.register("quarot_compressor", is_alias=True)
class QuaRotCompressor(BaseCompressor):
    """QuaRot: 4-bit / FP8 Quantization with randomized Walsh-Hadamard pre-rotation."""

    def __init__(
        self,
        model: nn.Module,
        bits: int = 4,
        hadamard_block_size: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="quarot", **kwargs)
        self.bits = bits
        self.hadamard_block_size = hadamard_block_size
        self.scales: dict[str, torch.Tensor] = {}

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """Applies Walsh-Hadamard orthogonal rotation followed by uniform quantization."""
        stats = {"bits": self.bits, "block_size": self.hadamard_block_size, "layers": []}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self._quantize_layer(module, name or "root")
                stats["layers"].append(name or "root")
        if not stats["layers"] and isinstance(self.model, nn.Linear):
            self._quantize_layer(self.model, "root")
            stats["layers"].append("root")
        return self.model, stats

    def _quantize_layer(self, layer: nn.Linear, name: str) -> None:
        w = layer.weight.data.float()
        # Apply orthogonal Hadamard rotation along input channels
        w_rot = fast_hadamard_transform(w, block_size=self.hadamard_block_size)
        q_max = 2 ** (self.bits - 1) - 1
        q_min = -2 ** (self.bits - 1)
        scale = (w_rot.abs().amax(dim=1, keepdim=True) / q_max).clamp(min=1e-8)
        w_q = torch.clamp(torch.round(w_rot / scale), q_min, q_max)
        # Dequantize rotated weight and rotate back
        w_dequant_rot = w_q * scale
        w_rec = fast_hadamard_transform(w_dequant_rot, block_size=self.hadamard_block_size)
        layer.weight.data.copy_(w_rec.to(layer.weight.dtype))
        self.scales[name] = scale

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
            "hadamard_block_size": self.hadamard_block_size,
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": self.bits,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                    },
                    "targets": ["Linear"],
                }
            },
        }


def quarot_factory(model: nn.Module, bits: int = 4, **kwargs: Any) -> nn.Module:
    comp = QuaRotCompressor(model, bits=bits, **kwargs)
    compressed, _ = comp.compress()
    return compressed
