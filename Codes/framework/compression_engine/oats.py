"""OATS: Outlier-Aware Training Sparsification. Registered as 'oats'."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from framework.compression_engine.oats_compressor import (
    OATSCompressor,
    oats_compressor_factory,
)
from framework.core.registry import COMPRESSOR_REGISTRY

__all__ = ["OATSCompressor", "oats_compressor_factory"]
