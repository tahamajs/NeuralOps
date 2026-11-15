#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
vLLM Plugin Entrypoint.

Plugin entrypoint dynamically registering NeuralOps custom models, attention modules,
and linear methods into vLLM, with offline simulation support for CI validation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.integrations.vllm import HeadlessVLLMWorker, get_vllm_runner, is_vllm_available

logger = logging.getLogger("NeuralOps.vLLMPlugin")

try:
    import vllm
    from vllm.model_executor.layers.linear import LinearMethodBase

    HAS_VLLM = True
    HAS_VLLM_LINEAR = True
except (ImportError, Exception):
    HAS_VLLM = False
    HAS_VLLM_LINEAR = False
    LinearMethodBase = object  # type: ignore[misc,assignment]

__all__ = [
    "NeuralOpsLinearMethod",
    "register_neuralops_linear_method",
    "register_neuralops_vllm_backend",
    "HeadlessVLLMWorker",
    "get_vllm_runner",
]


class NeuralOpsLinearMethod(LinearMethodBase):
    """
    LinearMethod implementation for NeuralOps compressed and dense layers in vLLM.

    Provides parameter buffer creation and execution dispatch for standard dense
    and mock-quantized representations.
    """

    def __init__(self, quant_config: Optional[Any] = None) -> None:
        """
        Initialize the linear method adapter.

        Args:
            quant_config: Quantization configuration mapping or object.
        """
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
        """
        Create weight parameters on the target layer module.

        Args:
            layer: Parent torch nn.Module.
            input_size_per_partition: Input dimension slice size.
            output_partition_sizes: List of output partition dimensions.
            input_size: Full input dimension.
            output_size: Full output dimension.
            params_dtype: Torch parameter datatype.
            **extra_weight_attrs: Attributes to attach to weight tensor.

        Returns:
            Dictionary containing the created weight parameters.

        Raises:
            IntegrationNotImplementedError: If quantized format is unsupported and not in mock mode.
        """
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
        """
        Execute forward GEMM application for the linear layer.

        Args:
            layer: Layer module containing weight parameter.
            x: Input activations tensor of shape [..., input_size].
            bias: Optional bias tensor to add to output.

        Returns:
            Computed linear transformation tensor of shape [..., output_size].

        Raises:
            IntegrationNotImplementedError: If layer lacks weight parameter.
        """
        weight = getattr(layer, "weight", None)
        if weight is None:
            raise IntegrationNotImplementedError("Layer is missing 'weight' parameter in NeuralOpsLinearMethod.")

        out = torch.matmul(x, weight.t().to(x.dtype))
        if bias is not None:
            out = out + bias
        return out


def register_neuralops_linear_method() -> bool:
    """
    Register NeuralOpsLinearMethod into vLLM quantization registry if vLLM is present.

    Returns:
        True if successfully registered or in offline mock environment; False otherwise.
    """
    if not HAS_VLLM_LINEAR:
        if os.environ.get("NEURALOPS_MOCK_SERVING") == "1":
            logger.info("NEURALOPS_MOCK_SERVING=1 active; mock linear method registered.")
            return True
        logger.debug("vLLM LinearMethodBase not available; skipping registration.")
        return False

    try:
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

        QUANTIZATION_METHODS["neuralops"] = NeuralOpsLinearMethod
        QUANTIZATION_METHODS["compressed-tensors"] = NeuralOpsLinearMethod
        logger.info("Successfully registered NeuralOpsLinearMethod with vLLM.")
        return True
    except Exception as exc:
        logger.warning("Could not register NeuralOpsLinearMethod in vLLM: %s", exc)
        return False


def register_neuralops_vllm_backend() -> bool:
    """
    Register NeuralOps attention backend into vLLM runtime.

    Returns:
        True if successfully registered or in offline mock environment; False otherwise.

    Raises:
        IntegrationNotImplementedError: When vLLM is missing and mock is disabled.
    """
    if not is_vllm_available():
        if os.environ.get("NEURALOPS_MOCK_SERVING") == "1":
            logger.info("NEURALOPS_MOCK_SERVING=1 active; registered mock vLLM backend.")
            return True
        raise IntegrationNotImplementedError(
            "vLLM is not installed. To register NeuralOps serving backend, install vLLM "
            "with `pip install 'vllm>=0.6.0'` or run with NEURALOPS_MOCK_SERVING=1 for mock runner."
        )

    try:
        from neuralops.integrations.vllm.custom_ops import register_vllm_custom_ops

        return register_vllm_custom_ops()
    except Exception as exc:
        raise IntegrationNotImplementedError(f"Failed to register NeuralOps backend in vLLM: {exc}") from exc
