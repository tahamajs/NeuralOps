"""
FP8 Compressor: Native Hardware Float8 Precision Compressor

Quantizes model parameters to FP8 formats (E4M3 and E5M2) with dynamic scaling factors optimized for modern GPU tensor cores.

Author: Taha Majlesi
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from neuralops.core.base import BaseCompressor
from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["FP8Compressor", "fp8_compressor_factory"]


@COMPRESSOR_REGISTRY.register("fp8_compressor", is_alias=True)
@COMPRESSOR_REGISTRY.register("fp8", is_alias=True)
class FP8Compressor(BaseCompressor):
    """
    FP8 Post-Training Quantization Compressor.
    Supports:
      - scale_type: 'per_tensor' | 'per_channel'
      - fp8_format: 'e4m3' | 'e5m2'
      - Uses native `torch.float8_e4m3fn` (or `torch.float8_e5m2`) on SM >= 8.9 (e.g. RTX 4070)
        with graceful emulation on CPU / older GPUs.
    """

    def __init__(
        self,
        model: nn.Module,
        scale_type: str = "per_tensor",
        fp8_format: str = "e4m3",
        use_native_fp8: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="fp8_compressor", **kwargs)
        if scale_type not in ("per_tensor", "per_channel"):
            raise ValueError(f"scale_type must be 'per_tensor' or 'per_channel', got '{scale_type}'")
        self.scale_type = scale_type
        self.fp8_format = fp8_format.lower()
        self.scales: dict[str, torch.Tensor] = {}
        self.quantized_tensors: dict[str, torch.Tensor] = {}

        self.has_sm89 = False
        if torch.cuda.is_available():
            try:
                major, minor = torch.cuda.get_device_capability()
                if (major > 8) or (major == 8 and minor >= 9):
                    self.has_sm89 = True
            except Exception:
                self.has_sm89 = False

        self.use_native = self.has_sm89 if use_native_fp8 is None else use_native_fp8
        self.max_val = 448.0 if self.fp8_format == "e4m3" else 57344.0
        self.target_dtype = (
            torch.float8_e4m3fn if self.fp8_format == "e4m3" else torch.float8_e5m2
        ) if hasattr(torch, "float8_e4m3fn") else torch.float32

    def _quantize_tensor(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantizes weight tensor to FP8 and returns (dequantized_w, fp8_w, scale)."""
        w_float = w.float()
        if self.scale_type == "per_channel":
            max_abs = w_float.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            max_abs = w_float.abs().amax().clamp(min=1e-8)

        scale = max_abs / self.max_val
        w_scaled = (w_float / scale).clamp(-self.max_val, self.max_val)

        if self.use_native and hasattr(torch, "float8_e4m3fn") and w.is_cuda:
            w_fp8 = w_scaled.to(self.target_dtype)
            w_dequant = w_fp8.to(w.dtype) * scale.to(w.dtype)
        else:
            mantissa_bits = 3 if self.fp8_format == "e4m3" else 2
            exp = torch.floor(torch.log2(w_scaled.abs().clamp(min=1e-8)))
            base = torch.pow(2.0, exp)
            step = base * (2.0 ** (-mantissa_bits))
            w_fp8 = torch.round(w_scaled / step) * step
            w_dequant = (w_fp8 * scale).to(w.dtype)
            w_fp8 = w_fp8.to(torch.float32)

        return w_dequant, w_fp8, scale

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """Quantizes all linear layer weights in the model to FP8."""
        stats = {
            "scale_type": self.scale_type,
            "fp8_format": self.fp8_format,
            "use_native": self.use_native,
            "layers": [],
        }
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layer_key = name or "root"
                w_dequant, w_fp8, scale = self._quantize_tensor(module.weight.data)
                module.weight.data.copy_(w_dequant)
                self.scales[layer_key] = scale
                self.quantized_tensors[f"{layer_key}.weight"] = w_fp8
                if module.bias is not None:
                    self.quantized_tensors[f"{layer_key}.bias"] = module.bias.data
                stats["layers"].append(layer_key)

        if not stats["layers"] and isinstance(self.model, nn.Linear):
            w_dequant, w_fp8, scale = self._quantize_tensor(self.model.weight.data)
            self.model.weight.data.copy_(w_dequant)
            self.scales["root"] = scale
            self.quantized_tensors["weight"] = w_fp8
            if self.model.bias is not None:
                self.quantized_tensors["bias"] = self.model.bias.data
            stats["layers"].append("root")

        return self.model, stats

    def compress_to_tensors(self, model: Optional[nn.Module] = None) -> dict[str, torch.Tensor]:
        """Returns compressed tensor dictionary compatible with compressed-tensors / safetensors."""
        m = model if model is not None else self.model
        tensors: dict[str, torch.Tensor] = {}
        for name, param in m.named_parameters():
            if name in self.quantized_tensors:
                tensors[name] = self.quantized_tensors[name]
            else:
                tensors[name] = param.data
        for name, buffer in m.named_buffers():
            tensors[name] = buffer.data
        for k, scale in self.scales.items():
            tensors[f"{k}.weight_scale"] = scale
        return tensors

    def get_quantization_config(self) -> dict[str, Any]:
        """Returns quantization configuration compatible with compressed-tensors and vLLM."""
        return {
            "quant_method": "compressed-tensors",
            "format": "FP8",
            "fp8_format": self.fp8_format,
            "scale_type": self.scale_type,
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "format": self.fp8_format,
                        "strategy": "channel" if self.scale_type == "per_channel" else "tensor",
                        "symmetric": True,
                    },
                    "targets": ["Linear"],
                }
            },
        }


def fp8_compressor_factory(
    model: nn.Module,
    scale_type: str = "per_tensor",
    fp8_format: str = "e4m3",
    **kwargs: Any,
) -> nn.Module:
    comp = FP8Compressor(model, scale_type=scale_type, fp8_format=fp8_format, **kwargs)
    compressed, _ = comp.compress()
    return compressed
