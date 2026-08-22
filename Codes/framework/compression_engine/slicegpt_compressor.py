"""SliceGPT PCA-based slicing compressor. Registered as 'slicegpt_pca'."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from framework.compression_engine.slicegpt_pca import (
    SliceGPTPCACompressor,
    slicegpt_pca_factory,
)
from framework.core.registry import COMPRESSOR_REGISTRY

__all__ = ["SliceGPTPCACompressor", "slicegpt_pca_factory"]
