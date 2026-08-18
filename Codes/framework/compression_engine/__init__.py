"""
Compression Engine modules for Efficient Transformer Toolkit.

Provides low-rank factorizations (SVD, AASVD, Dynamic Entropy SVD, Res-SVD),
quantization, pruning, tensor train, and hybrid compression strategies.
"""

import os
import glob
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

# Auto-discover and register all compressor modules in this directory
_pkg_dir = os.path.dirname(__file__)
for _file in glob.glob(os.path.join(_pkg_dir, "*.py")):
    _mod_name = os.path.splitext(os.path.basename(_file))[0]
    if not _mod_name.startswith("_"):
        try:
            importlib.import_module(f"framework.compression_engine.{_mod_name}")
        except Exception as _e:
            logger.debug("Failed to auto-load compressor module %s: %s", _mod_name, _e)

__all__ = [
    "DynamicEntropySVDCompressor",
    "DynamicSVDLinear",
    "compute_singular_entropy",
    "allocate_entropy_ranks",
    "ResSVDCompressor",
    "ResSVDLinear",
    "SpectralResidualLoRA",
]
