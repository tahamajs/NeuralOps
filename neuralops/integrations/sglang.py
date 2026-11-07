#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SGLang Serving Integration Bridge

Provides runtime integration and Radix prefix caching adapters for SGLang.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Optional

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.integrations.radix_prefix_cache import (
    RadixPrefixNode,
    SGLangRadixAdapter,
)
from neuralops.integrations.sglang.radix_recurrent import (
    RadixRecurrentNode,
    RadixRecurrentTree,
)

logger = logging.getLogger("NeuralOps.SGLang")


def sglang_is_available() -> bool:
    """Return True if SGLang or sgl_kernel is importable in current environment."""
    return (
        importlib.util.find_spec("sglang") is not None
        or importlib.util.find_spec("sgl_kernel") is not None
    )


def register_neuralops_sglang_plugin(
    config_class: Optional[type] = None,
    arch_names: Optional[list[str]] = None,
    **kwargs: Any,
) -> bool:
    """
    Register NeuralOps Linear Attention Backend within SGLang runtime.

    If sglang or sgl_kernel is installed, registers NeuralOpsLinearAttnBackend with SGLang
    linear attention model registry with uses_mamba_radix_cache=True.
    If sglang is not installed, raises an actionable IntegrationNotImplementedError specifying
    how to install sglang (`pip install sglang>=0.4.0`).
    """
    if not sglang_is_available():
        raise IntegrationNotImplementedError(
            "SGLang is not installed. To register NeuralOps linear attention plugins into SGLang, "
            "install SGLang with:\n"
            "    pip install 'sglang>=0.4.0'\n"
            "Or see https://github.com/sgl-project/sglang for platform-specific instructions."
        )

    try:
        from neuralops.integrations.sglang_backend.registry import register

        return register(config_class=config_class, arch_names=arch_names, **kwargs)
    except Exception as exc:
        raise IntegrationNotImplementedError(
            f"Failed to register NeuralOps plugin in SGLang runtime: {exc}"
        ) from exc


__all__ = [
    "register_neuralops_sglang_plugin",
    "sglang_is_available",
    "RadixPrefixNode",
    "RadixRecurrentNode",
    "RadixRecurrentTree",
    "SGLangRadixAdapter",
]
