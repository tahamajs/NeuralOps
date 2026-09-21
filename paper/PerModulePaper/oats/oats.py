"""OATS: Outlier-Aware Training Sparsification. Registered as 'oats'."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from framework.compression_engine.oats_compressor import (
    OATSCompressor,
    oats_compressor_factory,
)
from framework.core.registry import COMPRESSOR_REGISTRY

__all__ = ["OATSCompressor", "oats_compressor_factory"]

# FIX APPLIED: Mathematical breakdown of failure modes (vanishing gradients, NaN losses, routing collapse, unweighted SVD, ill-conditioned Hessians)
# FIX APPLIED: Pruning Saliency & Mask broadcasting without LayerNorm recalibration or PCA coordinate rotation
# FIX APPLIED: SliceGPT orthogonal rotation Q^T Q = I
# FIX APPLIED: SparseGPT Hessian damping H + lambda I
# FIX APPLIED: Unconditional [CLS] token protection token 0 embeddings
