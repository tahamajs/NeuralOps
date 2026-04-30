# framework/compression/svd_based/__init__.py
"""
SVD-based Compression Methods

This module provides core SVD-based compression techniques for linear layers,
including the main SVDLinearLayer and adaptive rank selection utilities.

Examples:
    >>> from framework.compression.svd_based import SVDLinearLayer, get_adaptive_ranks
    >>> 
    >>> # Create compressed layer
    >>> compressed = SVDLinearLayer(original_layer, rank=32)
    >>> 
    >>> # Get adaptive ranks for BERT
    >>> rank_map = get_adaptive_ranks("bert-base-uncased", ["query", "key", "value"])
"""

from .svd_linear import SVDLinearLayer
from .adaptive_ranks import get_adaptive_ranks, apply_adaptive_svd

__all__ = [
    "SVDLinearLayer",
    "get_adaptive_ranks",
    "apply_adaptive_svd",
]

__version__ = "1.0.0"