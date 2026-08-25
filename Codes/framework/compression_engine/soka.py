"""SoKA: Second-Order Knowledge-Aware Low-Rank Linear Compressor. Registered as 'soka'."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from framework.compression_engine.soka_compressor import (
    LowRankLinear,
    LowRankCompressor,
    SoKACompressor,
    soka_factory,
)
from framework.core.registry import COMPRESSOR_REGISTRY

__all__ = [
    "LowRankLinear",
    "LowRankCompressor",
    "SoKACompressor",
    "soka_factory",
]
