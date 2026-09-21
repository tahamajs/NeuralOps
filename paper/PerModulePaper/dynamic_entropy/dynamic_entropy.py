"""
Dynamic Entropy SVD alias module.
Re-exports DynamicEntropySVDCompressor, dynamic_entropy_svd_factory, and injection hooks from dynamic_rank.
"""

from framework.compression_engine.dynamic_rank import (
    compute_singular_entropy,
    allocate_entropy_ranks,
    DynamicSVDLinear,
    DynamicEntropySVDCompressor,
    dynamic_entropy_svd_factory,
    dynamic_rank_factory,
    entropy_rank_allocator_factory,
    inject_dynamic_entropy,
    dynamic_entropy_inject,
    inject_dynamic_rank,
    dynamic_rank_inject,
)

__all__ = [
    "compute_singular_entropy",
    "allocate_entropy_ranks",
    "DynamicSVDLinear",
    "DynamicEntropySVDCompressor",
    "dynamic_entropy_svd_factory",
    "dynamic_rank_factory",
    "entropy_rank_allocator_factory",
    "inject_dynamic_entropy",
    "dynamic_entropy_inject",
    "inject_dynamic_rank",
    "dynamic_rank_inject",
]
