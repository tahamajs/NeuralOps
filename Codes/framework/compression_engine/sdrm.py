"""SDRM: Scaled Dimension Rank Modulation Compressor. Registered as 'sdrm'."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from framework.compression_engine.sdrm_compressor import (
    SDRMCompressor,
    sdrm_compressor_factory,
    sdrm_factory,
    inject_sdrm,
)
from framework.core.registry import COMPRESSOR_REGISTRY

__all__ = [
    "SDRMCompressor",
    "sdrm_compressor_factory",
    "sdrm_factory",
    "inject_sdrm",
]
