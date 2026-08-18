"""
Compression Engine modules for Efficient Transformer Toolkit.

Provides low-rank factorizations (SVD, AASVD, Dynamic Entropy SVD, Res-SVD),
quantization, pruning, tensor train, and hybrid compression strategies.
"""

import importlib
import logging

from framework.compression_engine.dynamic_rank import (
    DynamicEntropySVDCompressor,
    DynamicSVDLinear,
    allocate_entropy_ranks,
    compute_singular_entropy,
)
from framework.compression_engine.res_svd import (
    ResSVDCompressor,
    ResSVDLinear,
    SpectralResidualLoRA,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DynamicEntropySVDCompressor",
    "DynamicSVDLinear",
    "compute_singular_entropy",
    "allocate_entropy_ranks",
    "ResSVDCompressor",
    "ResSVDLinear",
    "SpectralResidualLoRA",
]


def __getattr__(name: str):
    """Lazy module loader for other compression engines."""
    try:
        module = importlib.import_module(f"framework.compression_engine.{name}")
        globals()[name] = module
        return module
    except ImportError as e:
        raise AttributeError(
            f"module 'framework.compression_engine' has no attribute '{name}' ({e})"
        )
