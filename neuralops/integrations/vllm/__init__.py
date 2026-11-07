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

import sys
import types
from typing import Any, Optional

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.integrations.vllm.backend import NeuralOpsAttentionImpl
from neuralops.integrations.vllm.state_table import PagedStateTable
from neuralops.integrations.vllm_backend.backend import NeuralOpsAttentionBackend
from neuralops.integrations.vllm_backend.kv_cache_spec import NeuralOpsRecurrentSpec

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
            raise IntegrationNotImplementedError("vLLM is not installed. Cannot run generation.")
        return self.model.generate(prompts, **kwargs)


def _load_flat_vllm_symbols():
    import importlib.util
    import sys
    from pathlib import Path

    _flat_file = Path(__file__).parent.parent / "vllm.py"
    spec = importlib.util.spec_from_file_location("neuralops.integrations.vllm_module", _flat_file)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules["neuralops.integrations.vllm_module"] = mod
        spec.loader.exec_module(mod)
        return mod
    return None


_flat_vllm = _load_flat_vllm_symbols()
if _flat_vllm is not None:
    NeuralOpsvLLMMetadata = getattr(_flat_vllm, "NeuralOpsvLLMMetadata", None)
    NeuralOpsvLLMImpl = getattr(_flat_vllm, "NeuralOpsvLLMImpl", None)
    NeuralOpsvLLMAttentionBackend = getattr(_flat_vllm, "NeuralOpsvLLMAttentionBackend", None)
    NeuralOpsForCausalLMvLLM = getattr(_flat_vllm, "NeuralOpsForCausalLMvLLM", None)
    register_neuralops_vllm_plugin = getattr(_flat_vllm, "register_neuralops_vllm_plugin", None)
    vllm_is_available = getattr(_flat_vllm, "vllm_is_available", None)


class _VLLMModule(types.ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        if name == "HAS_VLLM" and _flat_vllm is not None:
            _flat_vllm.HAS_VLLM = value
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _VLLMModule


__all__ = [
    "NeuralOpsAttentionBackend",
    "NeuralOpsAttentionImpl",
    "NeuralOpsForCausalLMvLLM",
    "NeuralOpsRecurrentSpec",
    "NeuralOpsvLLMAttentionBackend",
    "NeuralOpsvLLMImpl",
    "NeuralOpsvLLMMetadata",
    "PagedStateTable",
    "VLLMAdapter",
    "register_neuralops_vllm_plugin",
    "vllm_is_available",
    "HAS_VLLM",
]
