#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
vLLM Custom Attention Backend Scaffold for NeuralOps.
"""

from __future__ import annotations

import logging
from typing import Optional

from neuralops.integrations.vllm_backend.backend import (
    NeuralOpsAttentionBackend,
    NeuralOpsImpl,
)
from neuralops.integrations.vllm_backend.kv_cache_spec import (
    NeuralOpsRecurrentSpec,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NeuralOpsAttentionBackend",
    "NeuralOpsImpl",
    "NeuralOpsRecurrentSpec",
    "register",
]


def register(backend_name: Optional[str] = None) -> bool:
    """
    Register NeuralOpsAttentionBackend with vLLM attention backend selector.

    Follows vLLM PR #30869 explicit custom backend registration pattern to avoid
    the AttentionBackendEnum.CUSTOM aliasing bug with TORCH_SDPA.
    """
    target_name = backend_name or NeuralOpsAttentionBackend.get_name()
    qualname = "neuralops.integrations.vllm_backend.backend.NeuralOpsAttentionBackend"
    registered = False

    # 1. Try modern vLLM v1 registry API (PR #30869 / vLLM v0.6+)
    try:
        from vllm.v1.attention.backends.registry import (  # type: ignore[import-not-found]
            AttentionBackendEnum,
            register_backend,
        )

        register_backend(AttentionBackendEnum.CUSTOM, qualname)
        register_backend(target_name, qualname)
        registered = True
        logger.info("Registered NeuralOpsAttentionBackend via v1.attention.backends.registry")
    except (ImportError, AttributeError, Exception) as e:
        logger.debug("v1 registry failed: %s", e)

    # 2. Try legacy selector registry API with AttentionBackendEnum.CUSTOM
    try:
        from vllm.attention.selector import (  # type: ignore[import-not-found]
            AttentionBackendEnum,
            _ATTENTION_BACKEND_REGISTRY,
        )

        custom_enum = getattr(AttentionBackendEnum, "CUSTOM", None)
        if custom_enum is not None:
            _ATTENTION_BACKEND_REGISTRY[custom_enum] = NeuralOpsAttentionBackend
            registered = True
        _ATTENTION_BACKEND_REGISTRY[target_name] = NeuralOpsAttentionBackend
        registered = True
        logger.info("Registered NeuralOpsAttentionBackend in _ATTENTION_BACKEND_REGISTRY")
    except (ImportError, AttributeError, Exception) as e:
        logger.debug("Legacy selector registry failed: %s", e)

    return registered
