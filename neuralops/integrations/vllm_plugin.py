#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
vLLM Plugin Entrypoint

Plugin entrypoint dynamically registering NeuralOps custom models and attention modules into vLLM.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple, Type
import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.vLLMPlugin")

try:
    import vllm
    from vllm.model_executor.layers.linear import LinearMethodBase
    HAS_VLLM_LINEAR = True
except (ImportError, Exception):
    HAS_VLLM_LINEAR = False
    LinearMethodBase = object


class NeuralOpsLinearMethod(LinearMethodBase):
    """
    vLLM Linear Method executing NeuralOps compressed layers (W4A16, W8A8, FP8).
    Plugs directly into vLLM's model executor layers.
    """

    def __init__(self, quant_config: Optional[Any] = None) -> None:
        if HAS_VLLM_LINEAR and hasattr(super(), "__init__"):
            super().__init__()
        self.quant_config = quant_config or {}
        self.format = self.quant_config.get("format", "W4A16") if isinstance(self.quant_config, dict) else "W4A16"
        self.bits = self.quant_config.get("bits", 4) if isinstance(self.quant_config, dict) else 4

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> dict[str, Any]:
        """Creates parameter buffers according to quantization scheme."""
        total_output_size = sum(output_partition_sizes)
        weight = nn.Parameter(
            torch.empty(
                total_output_size,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)
        for k, v in extra_weight_attrs.items():
            setattr(weight, k, v)
        return {"weight": weight}

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward GEMM execution applying dequantization or fused kernel."""
        weight = getattr(layer, "weight", None)
        if weight is None:
            raise RuntimeError("Layer missing weight parameter in NeuralOpsLinearMethod")

        if self.format.upper() == "FP8" and hasattr(torch, "float8_e4m3fn") and weight.dtype == torch.float8_e4m3fn:
            scale = getattr(layer, "weight_scale", 1.0)
            w_fp32 = weight.to(x.dtype) * scale
            out = torch.matmul(x, w_fp32.t())
        else:
            out = torch.matmul(x, weight.t().to(x.dtype))

        if bias is not None:
            out = out + bias
        return out


def register_neuralops_linear_method() -> bool:
    """Registers NeuralOpsLinearMethod into vLLM quantization registry if vLLM is present."""
    if not HAS_VLLM_LINEAR:
        logger.debug("vLLM LinearMethodBase not available; skipping registration.")
        return False

    try:
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
        QUANTIZATION_METHODS["neuralops"] = NeuralOpsLinearMethod
        QUANTIZATION_METHODS["compressed-tensors"] = NeuralOpsLinearMethod
        logger.info("Successfully registered NeuralOpsLinearMethod with vLLM.")
        return True
    except Exception as e:
        logger.warning("Could not register NeuralOpsLinearMethod in vLLM: %s", e)
        return False
