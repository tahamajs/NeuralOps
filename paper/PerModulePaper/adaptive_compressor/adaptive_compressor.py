"""
Adaptive Transformer Compressor: backward-compatible wrapper delegating to
the unified and production-hardened AdaptiveRanksCompressor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.adaptive_ranks import (
    AdaptiveRanksCompressor,
    adaptive_ranks_factory,
)

@COMPRESSOR_REGISTRY.register("adaptive_compressor")
class AdaptiveTransformerCompressor(AdaptiveRanksCompressor):
    """
    Backward-compatibility subclass delegating to the unified AdaptiveRanksCompressor.
    Resolves registry collisions while maintaining the legacy class interface.
    """

    def __init__(
        self,
        model: nn.Module,
        energy_threshold: float = 0.92,
        min_rank: int = 16,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            energy_threshold=energy_threshold,
            min_rank=min_rank,
            max_error=max_error,
            fallback_to_original=fallback_to_original,
            target_modules=target_modules,
            **kwargs,
        )


adaptive_svd_factory = adaptive_ranks_factory
COMPRESSOR_REGISTRY.register("adaptive_svd_factory")(adaptive_svd_factory)

adaptive_compressor_factory = adaptive_ranks_factory
COMPRESSOR_REGISTRY.register("adaptive_compressor_factory")(adaptive_compressor_factory)

__all__ = [
    "AdaptiveTransformerCompressor",
    "adaptive_svd_factory",
    "adaptive_compressor_factory",
]

