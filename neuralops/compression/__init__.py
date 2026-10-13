"""
Canonical `neuralops.compression` subpackage.
Re-exports the Compressor registry, BaseCompressor, and compression categories.
"""

from __future__ import annotations

from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.core.base import BaseCompressor
from neuralops.core.compression_pipeline import compress_model
import neuralops.compression_engine as _comp_engine

__all__ = ["COMPRESSOR_REGISTRY", "BaseCompressor", "compress_model"]
