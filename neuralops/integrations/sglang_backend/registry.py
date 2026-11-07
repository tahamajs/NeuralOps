#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SGLang Linear Attention Registry for NeuralOps.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "register",
]


def register(
    config_class: Optional[type] = None,
    arch_names: Optional[list[str]] = None,
    **kwargs: Any,
) -> bool:
    """
    Register NeuralOps Linear Attention Backend with SGLang linear attention registry.

    Registers LinearAttnModelSpec with uses_mamba_radix_cache=True for subquadratic
    and recurrent attention mechanisms (GLA, GDN, DeltaNet, etc.).
    """
    try:
        from sglang.srt.configs.linear_attn_model_registry import LinearAttnModelSpec, register_linear_attn_model
    except ImportError as e:
        logger.warning("SGLang is not installed; linear attention registration skipped: %s", e)
        return False

    if config_class is None:
        try:
            from transformers import GPT2Config

            target_config: type = GPT2Config
        except ImportError:

            class DummyConfig:
                pass

            target_config = DummyConfig
    else:
        target_config = config_class

    target_archs = arch_names or [
        "GPT2LMHeadModel",
        "NeuralOpsModel",
        "LlamaForCausalLM",
    ]

    backend_qualname = "neuralops.integrations.sglang_backend.backend.NeuralOpsLinearAttnBackend"

    spec = LinearAttnModelSpec(
        config_class=target_config,
        backend_class_name=backend_qualname,
        arch_names=target_archs,
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
        **kwargs,
    )

    register_linear_attn_model(spec)
    logger.info(
        "Successfully registered NeuralOpsLinearAttnBackend in SGLang (uses_mamba_radix_cache=True, archs=%s)",
        target_archs,
    )
    return True
