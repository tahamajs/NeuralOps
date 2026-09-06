"""ProxSparse: regularized learning of semi-structured sparsity. Registered as 'proxsparse'."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from framework.compression_engine.proxsparse import (
    ProxSparseCompressor,
    ProxSparsePruner,
    ProxSparsePrunedLoRALinear,
    proxsparse_factory,
)
from framework.core.registry import COMPRESSOR_REGISTRY

__all__ = [
    "ProxSparsePrunedLoRALinear",
    "ProxSparsePruner",
    "ProxSparseCompressor",
    "proxsparse_factory",
]

