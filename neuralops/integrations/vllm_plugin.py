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
    """Reserved adapter type; packed quantized weight layouts are not implemented."""

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
        """Reject initialization until packed quantization layouts are defined."""
        raise NotImplementedError(
            "NeuralOps vLLM quantized weight creation is not implemented; dense tensors "
            "must not be registered as W4A16/FP8 weights."
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reject execution until vLLM quantization metadata and kernels are integrated."""
        raise NotImplementedError(
            "NeuralOps vLLM quantized linear execution is not implemented; the adapter "
            "is intentionally not registered."
        )


def register_neuralops_linear_method() -> bool:
    """Refuse registration until packed-weight layouts have a verified implementation."""
    if not HAS_VLLM_LINEAR:
        logger.debug("vLLM LinearMethodBase not available; skipping registration.")
        return False
    logger.warning(
        "NeuralOps linear quantization was not registered: packed W4A16/FP8 layouts "
        "and vLLM quantization lifecycle are not implemented and verified."
    )
    return False
