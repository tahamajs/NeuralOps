"""Randomized SVD module alias."""
from neuralops.compression.svd.randomized_compression import (
    RandomizedSVDCompressor,
    RandomizedSVDLinear,
)
from neuralops.core.svd_utils import randomized_svd

__all__ = [
    "RandomizedSVDCompressor",
    "RandomizedSVDLinear",
    "randomized_svd",
]
