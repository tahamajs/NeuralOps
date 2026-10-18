"""
SmoothQuant: Accurate 8-Bit Weight and Activation Quantization

Smooths activation outliers by mathematically migrating scale difficulty from activations to weights via per-channel scaling factors.

Author: Taha Majlesi
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from neuralops.core.base import BaseCompressor
from neuralops.core.registry import COMPRESSOR_REGISTRY

__all__ = ["smoothquant_layer", "smoothquant_model", "SmoothQuantLinear", "SmoothQuantCompressor"]


class SmoothQuantLinear(nn.Module):
    """Linear layer applying in-SRAM SmoothQuant activation scaling and GEMM."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: torch.Tensor,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.register_buffer("s_inv", (1.0 / s).to(device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from neuralops.core.triton_kernels.fused_norm_linear import triton_fused_act_scale_quant_gemm
        return triton_fused_act_scale_quant_gemm(x, self.s_inv, self.weight, bias=self.bias)


def smoothquant_layer(
    module: nn.Linear,
    alpha: float = 0.5,
    calib_data: torch.Tensor | list[torch.Tensor] | None = None,
    act_scales: torch.Tensor | None = None,
) -> tuple[nn.Linear, torch.Tensor]:
    """Applies SmoothQuant to a linear layer.

    Migrates activation outliers into weights:
      s = (a_max^alpha) / (w_max^(1-alpha))
      W' = W * diag(s)
      X' = X * diag(s)^-1

    Returns:
        Tuple of (modified module, scale factor s).
    """
    if calib_data is None and act_scales is None:
        s_id = torch.ones(module.weight.shape[1], device=module.weight.device)
        return module, s_id

    if act_scales is None:
        if isinstance(calib_data, list):
            stacked = torch.cat([x.reshape(-1, x.shape[-1]) for x in calib_data], dim=0)
        else:
            stacked = calib_data.reshape(-1, calib_data.shape[-1])
        if stacked.shape[-1] != module.weight.shape[1]:
            a_max = module.weight.abs().amax(dim=0).clamp(min=1e-5)
        else:
            a_max = stacked.abs().amax(dim=0).clamp(min=1e-5)
    else:
        a_max = act_scales.clamp(min=1e-5)

    w_max = module.weight.abs().amax(dim=0).clamp(min=1e-5)

    s = (a_max.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=1e-5)

    module.weight.data = (module.weight.data * s.unsqueeze(0)).to(module.weight.dtype)
    module.register_buffer("s_inv", (1.0 / s).to(module.weight.dtype))

    orig_forward = module.forward
    def _smoothed_forward(x: torch.Tensor) -> torch.Tensor:
        from neuralops.core.triton_kernels.fused_norm_linear import triton_fused_act_scale_quant_gemm
        if hasattr(module, "s_inv"):
            return triton_fused_act_scale_quant_gemm(x, module.s_inv, module.weight, bias=module.bias)
        return orig_forward(x)

    module.forward = _smoothed_forward
    return module, s


def smoothquant_model(
    model: nn.Module,
    calib_data: list[torch.Tensor] | None = None,
    alpha: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Applies SmoothQuant to eligible nn.Linear layers in model, protecting task heads."""
    from neuralops.utils.compat.architectures import should_compress_linear
    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not should_compress_linear(model, name):
                continue
            _, s = smoothquant_layer(module, alpha=alpha, calib_data=calib_data)
            scales[name] = s
    return scales


@COMPRESSOR_REGISTRY.register("smoothquant_real", is_alias=True)
@COMPRESSOR_REGISTRY.register("smoothquant_compressor", is_alias=True)
class SmoothQuantCompressor(BaseCompressor):
    """Unified BaseCompressor wrapper for SmoothQuant W8A8 with compressed-tensors support."""

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.5,
        bits: int = 8,
        calib_data: Optional[list[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="smoothquant", **kwargs)
        self.alpha = alpha
        self.bits = bits
        self.calib_data = calib_data
        self.scales: dict[str, torch.Tensor] = {}

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """Applies SmoothQuant across all linear layers in the model."""
        stats = {"alpha": self.alpha, "bits": self.bits, "layers": []}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                _, s = smoothquant_layer(module, alpha=self.alpha, calib_data=self.calib_data)
                self.scales[name or "root"] = s
                stats["layers"].append(name or "root")
        if not stats["layers"] and isinstance(self.model, nn.Linear):
            _, s = smoothquant_layer(self.model, alpha=self.alpha, calib_data=self.calib_data)
            self.scales["root"] = s
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
            "format": "W8A8",
            "bits": self.bits,
            "alpha": self.alpha,
            "config_groups": {
                "group_0": {
                    "weights": {
                        "num_bits": self.bits,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                    },
                    "input_activations": {
                        "num_bits": self.bits,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "token",
                    },
                    "targets": ["Linear"],
                }
            },
        }
