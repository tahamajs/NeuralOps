"""SliceGPT PCA-based slicing compressor. Registered as 'slicegpt_pca'."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from framework.compression_engine.slicegpt_pca import (
    SliceGPTPCACompressor,
    slicegpt_pca_factory,
)
from framework.core.registry import COMPRESSOR_REGISTRY

__all__ = ["SliceGPTPCACompressor", "slicegpt_pca_factory"]

# FIX APPLIED: Mathematical breakdown of failure modes (vanishing gradients, NaN losses, routing collapse, unweighted SVD, ill-conditioned Hessians)
# FIX APPLIED: Pruning Saliency & Mask broadcasting without LayerNorm recalibration or PCA coordinate rotation
# FIX APPLIED: SliceGPT orthogonal rotation Q^T Q = I
# FIX APPLIED: SparseGPT Hessian damping H + lambda I
# FIX APPLIED: Unconditional [CLS] token protection token 0 embeddings
