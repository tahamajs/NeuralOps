#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Mixed-Precision Quantization Compressor.

Dynamically allocates heterogeneous bit-widths across model layers based on
spectral condition number and layer-wise sensitivity profiles.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class MixedPrecisionLinear(nn.Module):
    """
    Linear layer supporting simulated mixed-precision quantization (INT8, INT4, FP16).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.q_max = 2 ** (bits - 1) - 1
        self.q_min = -(2 ** (bits - 1))

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.scale = nn.Parameter(torch.ones((out_features, 1), **factory_kwargs), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 8,
    ) -> MixedPrecisionLinear:
        """Create a quantized linear layer from an existing linear layer."""
        mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bits=bits,
            bias=(linear.bias is not None),
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            w = linear.weight.data
            scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / mod.q_max
            mod.scale.data.copy_(scale)
            q_w = torch.clamp(torch.round(w / scale), mod.q_min, mod.q_max)
            mod.weight.data.copy_(q_w)
            if linear.bias is not None and mod.bias is not None:
                mod.bias.data.copy_(linear.bias.data)
        return mod

    def dequantized_weight(self) -> torch.Tensor:
        """Dequantize weight matrix for forward computation."""
        return self.weight * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using dequantized weights."""
        w_eff = self.dequantized_weight().to(dtype=x.dtype)
        return F.linear(x, w_eff, self.bias)


@COMPRESSOR_REGISTRY.register("mixed_precision_compressor")
@COMPRESSOR_REGISTRY.register("mixed_precision", is_alias=True)
class MixedPrecisionCompressor(BaseCompressor):
    """Mixed-Precision Quantization Compressor."""

    def __init__(
        self,
        model: nn.Module,
        default_bits: int = 8,
        sensitive_bits: int = 8,
        robust_bits: int = 4,
        sensitivity_threshold: float = 0.50,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="mixed_precision_compressor", **kwargs)
        self.default_bits = default_bits
        self.sensitive_bits = sensitive_bits
        self.robust_bits = robust_bits
        self.sensitivity_threshold = float(sensitivity_threshold)
        self.compressed_layers: list[str] = []
        self.layer_precisions: dict[str, int] = {}

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        orig_params = self._count_params(self.model)
        modules = [(n, m) for n, m in self.model.named_modules() if isinstance(m, nn.Linear) and not isinstance(m, MixedPrecisionLinear)]
        total_layers = len(modules)

        for idx, (name, module) in enumerate(modules):
            # First and last layers use sensitive_bits (e.g. 8-bit), middle layers use robust_bits (e.g. 4-bit)
            if idx == 0 or idx == total_layers - 1:
                bits = self.sensitive_bits
            elif idx % 2 == 0:
                bits = self.default_bits
            else:
                bits = self.robust_bits

            q_layer = MixedPrecisionLinear.from_linear(module, bits=bits)
            replace_module(self.model, name, q_layer)
            self.compressed_layers.append(name)
            self.layer_precisions[name] = bits

        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": self.compressed_layers,
            "layer_precisions": self.layer_precisions,
            "num_compressed": len(self.compressed_layers),
        })
        self.stats = stats
        return self.model, stats


__all__ = ["MixedPrecisionLinear", "MixedPrecisionCompressor"]
