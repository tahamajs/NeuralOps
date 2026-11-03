#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
vLLM Runtime Integration for NeuralOps.
"""

from typing import Any, Optional
from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.integrations.vllm.backend import (
    NeuralOpsAttentionBackend,
    NeuralOpsAttentionImpl,
)
from neuralops.integrations.vllm.state_table import PagedStateTable

try:
    import vllm
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


class VLLMAdapter:
    """vLLM runtime adapter for NeuralOps models."""

    def __init__(self, model: Optional[Any] = None, **kwargs: Any) -> None:
        if not HAS_VLLM:
            raise IntegrationNotImplementedError(
                "vLLM is not installed. NeuralOps vLLM serving adapter fails closed when vLLM is unavailable. "
                "See docs/roadmap/VLLM_PLAN.md for details."
            )
        self.model = model
        self.kwargs = kwargs

    def generate(self, prompts: Any, **kwargs: Any) -> Any:
        if not HAS_VLLM:
            raise IntegrationNotImplementedError(
                "vLLM is not installed. Cannot run generation."
            )
        return self.model.generate(prompts, **kwargs)


__all__ = [
    "NeuralOpsAttentionBackend",
    "NeuralOpsAttentionImpl",
    "PagedStateTable",
    "VLLMAdapter",
    "HAS_VLLM",
]
