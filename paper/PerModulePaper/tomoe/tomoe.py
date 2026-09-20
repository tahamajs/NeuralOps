"""
ToMoE: Mixture of Experts via weight clustering.
Registered as 'tomoe' and 'tomoe_compressor'.
Alias/re-export module for tomoe_compressor.
"""
from __future__ import annotations

from framework.compression_engine.tomoe_compressor import (
    ToMoELinear,
    ToMoECompressor,
    tomoe_compressor_factory,
    tomoe_factory,
    inject_tomoe,
)
from framework.core.registry import COMPRESSOR_REGISTRY

__all__ = [
    "ToMoELinear",
    "ToMoECompressor",
    "tomoe_compressor_factory",
    "tomoe_factory",
    "inject_tomoe",
]
