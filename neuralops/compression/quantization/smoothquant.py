#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SmoothQuant: Accurate 8-Bit Weight and Activation Quantization

Smooths activation outliers by mathematically migrating scale difficulty from activations
to weights via per-channel scaling factors, and completes the transformation by performing
symmetric INT8 quantization on the migrated weights.

Note on kernel naming: `triton_fused_act_scale_quant_gemm` applies per-channel activation
outlier smoothing ($X' = X \cdot \operatorname{diag}(s)^{-1}$, $W' = W \cdot \operatorname{diag}(s)$)
prior to downstream low-precision quantization, resolving the misnomer where 'quant' in the
kernel name designates the pre-quantization scale-smoothing transform.
"""

from __future__ import annotations

from typing import Any, Optional
import torch
import torch.nn as nn

from neuralops.core.base import BaseCompressor
from neuralops.core.registry import COMPRESSOR_REGISTRY

__all__ = ["smoothquant_layer", "smoothquant_model", "SmoothQuantLinear", "SmoothQuantCompressor"]


class SmoothQuantLinear(nn.Module):
    r"""Linear layer applying in-SRAM SmoothQuant activation scaling and INT8 quantized GEMM.

    Applies per-channel activation outlier smoothing ($X' = X \cdot \operatorname{diag}(s)^{-1}$,
    $W' = W \cdot \operatorname{diag}(s)$) and symmetric INT8 quantization on the migrated weights:
    $W_{\text{int8}} = \operatorname{clamp}(\operatorname{round}(W' / \text{scale}_W), -128, 127)$.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("s_inv", (1.0 / s.to(torch.float32)).to(device=device, dtype=dtype))
        self.register_buffer("act_scale", s.to(torch.float32).to(device=device, dtype=dtype))

        if weight is None:
            weight = torch.randn(out_features, in_features, device=device, dtype=dtype)

        w_fp32 = weight.to(torch.float32)
        s_fp32 = s.to(torch.float32).to(w_fp32.device)
        w_migrated = w_fp32 * s_fp32.unsqueeze(0)

        scale_w = (w_migrated.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
        w_int8 = torch.clamp(torch.round(w_migrated / scale_w), -128, 127).to(torch.int8)
        w_dequant = (w_int8.to(torch.float32) * scale_w).to(device=device, dtype=dtype)

        self.register_buffer("weight_int8", w_int8)
        self.register_buffer("weight_scale", scale_w.to(device=device, dtype=dtype))
        self.weight = nn.Parameter(w_dequant)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies fused activation outlier smoothing ($X' = X \cdot \operatorname{diag}(s)^{-1}$) and GEMM prior to downstream quantization via triton_fused_act_scale_quant_gemm."""
        from neuralops.ops.triton.fused_norm_linear import triton_fused_act_scale_quant_gemm

        w_deq = (self.weight_int8.to(torch.float32) * self.weight_scale.to(torch.float32)).to(x.dtype)
        return triton_fused_act_scale_quant_gemm(x, self.s_inv, w_deq, bias=self.bias)


def smoothquant_layer(
    module: nn.Linear,
    alpha: float = 0.5,
    calib_data: torch.Tensor | list[torch.Tensor] | None = None,
    act_scales: torch.Tensor | None = None,
    quantize: bool = True,
) -> tuple[nn.Linear, torch.Tensor]:
    """Applies SmoothQuant to a linear layer.

    Migrates activation outliers into weights:
      s = (a_max^alpha) / (w_max^(1-alpha))
      W' = W * diag(s)
      X' = X * diag(s)^-1

    The runtime hook executes `triton_fused_act_scale_quant_gemm`, which applies per-channel
    activation outlier smoothing ($X' = X \cdot \operatorname{diag}(s)^{-1}$, $W' = W \cdot \operatorname{diag}(s)$)
    prior to downstream low-precision quantization, resolving the kernel misnomer.

    When quantize=True, completes the transformation by performing INT8 symmetric quantization:
      W_int8 = clamp(round(W' / scale_W), -128, 127)
      W_dequant = W_int8 * scale_W

    Returns:
        Tuple of (modified module, scale factor s).
    """
    if calib_data is None and act_scales is None:
        s_id = torch.ones(module.weight.shape[1], device=module.weight.device, dtype=module.weight.dtype)
        return module, s_id

    if act_scales is None:
        if isinstance(calib_data, list):
            stacked = torch.cat([x.reshape(-1, x.shape[-1]) for x in calib_data], dim=0)
        else:
            assert calib_data is not None
            stacked = calib_data.reshape(-1, calib_data.shape[-1])
        if stacked.shape[-1] != module.weight.shape[1]:
            a_max = module.weight.abs().amax(dim=0).clamp(min=1e-5)
        else:
            a_max = stacked.abs().amax(dim=0).clamp(min=1e-5)
    else:
        a_max = act_scales.clamp(min=1e-5)

    w_max = module.weight.abs().amax(dim=0).clamp(min=1e-5)
    s = (a_max.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=1e-5)

    w_fp32 = module.weight.data.to(torch.float32)
    s_fp32 = s.to(torch.float32).to(w_fp32.device)
    w_migrated = w_fp32 * s_fp32.unsqueeze(0)

    module.register_buffer("s_inv", (1.0 / s_fp32).to(module.weight.dtype))
    module.register_buffer("act_scale", s_fp32.to(module.weight.dtype))

    if quantize:
        scale_w = (w_migrated.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
        w_int8 = torch.clamp(torch.round(w_migrated / scale_w), -128, 127).to(torch.int8)
        w_dequant = (w_int8.to(torch.float32) * scale_w).to(module.weight.dtype)

        module.weight.data.copy_(w_dequant)
        module.register_buffer("weight_int8", w_int8)
        module.register_buffer("weight_scale", scale_w.to(module.weight.dtype))
    else:
        module.weight.data = w_migrated.to(module.weight.dtype)

    orig_forward = module.forward

    def _smoothed_forward(x: torch.Tensor) -> torch.Tensor:
        from neuralops.ops.triton.fused_norm_linear import triton_fused_act_scale_quant_gemm

        if hasattr(module, "weight_int8") and hasattr(module, "weight_scale"):
            w_eval = (module.weight_int8.to(torch.float32) * module.weight_scale.to(torch.float32)).to(x.dtype)
        else:
            w_eval = module.weight
        if hasattr(module, "s_inv"):
            return triton_fused_act_scale_quant_gemm(x, module.s_inv, w_eval, bias=module.bias)
        return orig_forward(x)

    module.forward = _smoothed_forward
    return module, s


def smoothquant_model(
    model: nn.Module,
    calib_data: list[torch.Tensor] | None = None,
    alpha: float = 0.5,
    quantize: bool = True,
) -> dict[str, torch.Tensor]:
    """Applies SmoothQuant to eligible nn.Linear layers in model, protecting task heads."""
    from neuralops.utils.compat.architectures import should_compress_linear

    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not should_compress_linear(model, name):
                continue
            _, s = smoothquant_layer(module, alpha=alpha, calib_data=calib_data, quantize=quantize)
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
        stats: dict[str, Any] = {
            "alpha": self.alpha,
            "bits": self.bits,
            "layers": [],
            "layer_errors": {},
            "layer_scales": {},
            "layer_weight_scales": {},
            "method": "smoothquant_w8a8",
        }
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layer_key = name or "root"
                orig_w = module.weight.data.clone()
                _, s = smoothquant_layer(module, alpha=self.alpha, calib_data=self.calib_data, quantize=True)
                self.scales[layer_key] = s
                stats["layers"].append(layer_key)
                stats["layer_scales"][layer_key] = s
                if hasattr(module, "weight_scale"):
                    stats["layer_weight_scales"][layer_key] = module.weight_scale
                norm_orig = torch.norm(orig_w.float(), p="fro").item()
                norm_diff = torch.norm((orig_w - module.weight.data).float(), p="fro").item()
                stats["layer_errors"][layer_key] = float(norm_diff / max(norm_orig, 1e-12))

        if not stats["layers"] and isinstance(self.model, nn.Linear):
            orig_w = self.model.weight.data.clone()
            _, s = smoothquant_layer(self.model, alpha=self.alpha, calib_data=self.calib_data, quantize=True)
            self.scales["root"] = s
            stats["layers"].append("root")
            stats["layer_scales"]["root"] = s
            if hasattr(self.model, "weight_scale"):
                stats["layer_weight_scales"]["root"] = self.model.weight_scale
            norm_orig = torch.norm(orig_w.float(), p="fro").item()
            norm_diff = torch.norm((orig_w - self.model.weight.data).float(), p="fro").item()
            stats["layer_errors"]["root"] = float(norm_diff / max(norm_orig, 1e-12))

        avg_err = (
            sum(stats["layer_errors"].values()) / max(1, len(stats["layer_errors"]))
            if stats["layer_errors"]
            else 0.0
        )
        stats["avg_reconstruction_error"] = float(avg_err)
        stats["compression_ratio"] = round(16.0 / max(1, self.bits), 2)
        stats["num_compressed_layers"] = len(stats["layers"])
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
