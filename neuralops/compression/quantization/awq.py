#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
AWQ: Activation-Aware Weight Quantization

Identifies salient weight channels via activation magnitudes, protects them using
per-channel scaling transformations, and quantizes weights onto a strict integer grid
retaining explicit per-channel/per-group scaling factors.
"""

from __future__ import annotations

from typing import Any, Optional
import torch
import torch.nn as nn

from neuralops.core.base import BaseCompressor
from neuralops.core.registry import COMPRESSOR_REGISTRY

__all__ = ["AWQQuantizer", "AWQCompressor", "quantize_model_awq"]


class AWQQuantizer:
    """Protects salient weights by observing activation magnitudes and quantizing onto an integer grid."""

    def __init__(
        self,
        layer: nn.Linear,
        bits: int = 4,
        alpha: float = 0.5,
        group_size: int = -1,
    ) -> None:
        self.layer = layer
        self.bits = bits
        self.alpha = alpha
        self.group_size = group_size
        self.act_scales = torch.zeros(layer.weight.shape[1], device=layer.weight.device, dtype=torch.float32)
        self.qweight: Optional[torch.Tensor] = None
        self.scales: Optional[torch.Tensor] = None
        self.error: float = 0.0

    def observe(self, x: torch.Tensor) -> None:
        """Observes per-channel activation scales from calibration batches."""
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        s = torch.mean(torch.abs(x.to(torch.float32)), dim=0)
        self.act_scales = torch.maximum(self.act_scales, s)

    def _quantize_to_grid(
        self, W_scaled: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantizes scaled weights to an exact integer grid and returns (q_int, weight_scale)."""
        q_max = 2 ** (self.bits - 1) - 1
        q_min = -(2 ** (self.bits - 1))
        out_features, in_features = W_scaled.shape

        if self.group_size > 0 and in_features % self.group_size == 0:
            num_groups = in_features // self.group_size
            W_grouped = W_scaled.reshape(out_features, num_groups, self.group_size)
            max_val = torch.amax(torch.abs(W_grouped), dim=-1, keepdim=True).clamp(min=1e-8)
            group_scales = max_val / float(q_max)
            q_grouped = torch.clamp(torch.round(W_grouped / group_scales), q_min, q_max)
            q_int = q_grouped.reshape(out_features, in_features).to(torch.int8)
            weight_scale = group_scales.expand(out_features, num_groups, self.group_size).reshape(out_features, in_features)
        else:
            max_val = torch.amax(torch.abs(W_scaled), dim=-1, keepdim=True).clamp(min=1e-8)
            scale = max_val / float(q_max)
            q_int = torch.clamp(torch.round(W_scaled / scale), q_min, q_max).to(torch.int8)
            weight_scale = scale.expand(out_features, in_features)

        return q_int, weight_scale

    def quantize(self) -> torch.Tensor:
        """Applies activation-aware channel re-scaling, stores integer grid and explicit scale factors."""
        orig_w = self.layer.weight.data.clone().to(torch.float32)
        act_s = self.act_scales.clamp(min=1e-5).pow(self.alpha).to(orig_w.device)

        W_scaled = orig_w * act_s.unsqueeze(0)
        q_int, weight_scale = self._quantize_to_grid(W_scaled)

        # Combined per-channel/per-group scale factor: S_eff = weight_scale / act_s
        effective_scale = weight_scale / act_s.unsqueeze(0)
        W_dequant = q_int.to(torch.float32) * effective_scale

        self.qweight = q_int
        self.scales = effective_scale

        self.layer.register_buffer("qweight", q_int)
        self.layer.register_buffer("scales", effective_scale.to(self.layer.weight.dtype))
        self.layer.register_buffer("weight_scale", weight_scale.to(self.layer.weight.dtype))
        self.layer.register_buffer("act_scale", act_s.to(self.layer.weight.dtype))
        self.layer.weight.data.copy_(W_dequant.to(self.layer.weight.dtype))

        norm_orig = torch.norm(orig_w, p="fro").item()
        norm_diff = torch.norm(orig_w - W_dequant, p="fro").item()
        self.error = float(norm_diff / max(norm_orig, 1e-12))
        return W_dequant


@COMPRESSOR_REGISTRY.register("awq", is_alias=True)
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
        calib_data: Optional[list[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="awq", **kwargs)
        self.bits = bits
        self.alpha = alpha
        self.group_size = group_size
        self.calib_data = calib_data
        self.quantized_weights: dict[str, torch.Tensor] = {}
        self.scales: dict[str, torch.Tensor] = {}

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """Applies AWQ quantization across all linear layers in model."""
        stats: dict[str, Any] = {
            "bits": self.bits,
            "alpha": self.alpha,
            "group_size": self.group_size,
            "layers": [],
            "layer_errors": {},
            "method": "awq",
        }
        total_params = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layer_key = name or "root"
                awq = AWQQuantizer(
                    module, bits=self.bits, alpha=self.alpha, group_size=self.group_size
                )
                if self.calib_data is not None:
                    for batch in self.calib_data:
                        awq.observe(batch)
                awq.quantize()
                if awq.qweight is not None:
                    self.quantized_weights[f"{layer_key}.qweight"] = awq.qweight
                if awq.scales is not None:
                    self.scales[f"{layer_key}.scales"] = awq.scales
                stats["layers"].append(layer_key)
                stats["layer_errors"][layer_key] = awq.error
                total_params += module.weight.numel()

        if not stats["layers"] and isinstance(self.model, nn.Linear):
            awq = AWQQuantizer(
                self.model, bits=self.bits, alpha=self.alpha, group_size=self.group_size
            )
            if self.calib_data is not None:
                for batch in self.calib_data:
                    awq.observe(batch)
            awq.quantize()
            if awq.qweight is not None:
                self.quantized_weights["qweight"] = awq.qweight
            if awq.scales is not None:
                self.scales["scales"] = awq.scales
            stats["layers"].append("root")
            stats["layer_errors"]["root"] = awq.error
            total_params += self.model.weight.numel()

        avg_err = (
            sum(stats["layer_errors"].values()) / max(1, len(stats["layer_errors"]))
            if stats["layer_errors"]
            else 0.0
        )
        stats["avg_reconstruction_error"] = float(avg_err)
        stats["total_parameters"] = total_params
        stats["num_compressed_layers"] = len(stats["layers"])
        stats["compression_ratio"] = round(16.0 / max(1, self.bits), 2)
        return self.model, stats

    def compress_to_tensors(self, model: Optional[nn.Module] = None) -> dict[str, torch.Tensor]:
        """Returns compressed tensor dictionary compatible with compressed-tensors / safetensors."""
        m = model if model is not None else self.model
        tensors: dict[str, torch.Tensor] = {}
        for name, param in m.named_parameters():
            tensors[name] = param.data
        for name, buffer in m.named_buffers():
            tensors[name] = buffer.data
        for k, v in self.quantized_weights.items():
            if v is not None:
                tensors[k] = v
        for k, v in self.scales.items():
            if v is not None:
                tensors[k] = v
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
    group_size: int = -1,
) -> nn.Module:
    """Applies AWQ quantization across all linear layers in model."""
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            awq = AWQQuantizer(module, bits=bits, alpha=alpha, group_size=group_size)
            for batch in calibration_data:
                awq.observe(batch)
            awq.quantize()
    return model
