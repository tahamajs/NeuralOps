# framework/compression/hybrid/__init__.py
"""
Hybrid SVD Encoder Layers

This module provides hybrid encoder layers that replace standard attention
with SVD-based single-head attention followed by standard FFN.

Includes:
    - SVDSingleHead: Single-head SVD approximation with positional encoding
    - DirectSVDEncoderLayer: Complete encoder layer using SVD attention

Examples:
    >>> from framework.compression.hybrid import SVDSingleHead, DirectSVDEncoderLayer
    >>> 
    >>> # Use SVD single head
    >>> svd_attn = SVDSingleHead(hidden_size=768, svd_rank=20)
    >>> 
    >>> # Replace entire encoder layer
    >>> encoder_layer = DirectSVDEncoderLayer(config)
"""

from .hybrid_svd_layer import SVDSingleHead, DirectSVDEncoderLayer

__all__ = [
    "SVDSingleHead",
    "DirectSVDEncoderLayer",
]

__version__ = "1.0.0"