#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Post-Initialization Registration Hooks

Registers deferred aliases and late-bound integrations without eager imports during module load.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _register_core_aliases() -> None:
    """Register backward-compatible aliases in COMPRESSOR_REGISTRY."""
    from neuralops.core.registry import COMPRESSOR_REGISTRY

    try:
        from neuralops.compression.svd.direct_svd import DirectSVDCompressor
        COMPRESSOR_REGISTRY.register("svd", is_alias=True)(DirectSVDCompressor)
        COMPRESSOR_REGISTRY.register("hnc_svd", is_alias=True)(DirectSVDCompressor)
        COMPRESSOR_REGISTRY.register("hnc_svd_compressor", is_alias=True)(DirectSVDCompressor)
    except Exception as exc:
        try:
            from neuralops.compression_engine.svd.direct_svd import DirectSVDCompressor
            COMPRESSOR_REGISTRY.register("svd", is_alias=True)(DirectSVDCompressor)
            COMPRESSOR_REGISTRY.register("hnc_svd", is_alias=True)(DirectSVDCompressor)
            COMPRESSOR_REGISTRY.register("hnc_svd_compressor", is_alias=True)(DirectSVDCompressor)
        except Exception as inner_exc:
            logger.warning("Could not register direct SVD compressor aliases: %s (fallback: %s)", exc, inner_exc)

    try:
        from neuralops.compression.low_rank.svd_llm_compressor import SVDLLMCompressor
        COMPRESSOR_REGISTRY.register("svd_llm", is_alias=True)(SVDLLMCompressor)
    except Exception as exc:
        try:
            from neuralops.compression_engine.low_rank.svd_llm_compressor import SVDLLMCompressor
            COMPRESSOR_REGISTRY.register("svd_llm", is_alias=True)(SVDLLMCompressor)
        except Exception as inner_exc:
            logger.warning("Could not register SVD-LLM compressor alias: %s (fallback: %s)", exc, inner_exc)
