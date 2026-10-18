"""
NeuralOps Unified Compression Engines.

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.core.base import BaseCompressor
from neuralops.core.compression_pipeline import compress_model
import neuralops.compression_engine as _comp_engine

__all__ = ["COMPRESSOR_REGISTRY", "BaseCompressor", "compress_model"]
