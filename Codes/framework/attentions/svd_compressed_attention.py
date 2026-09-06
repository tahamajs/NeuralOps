"""
SVD-Compressed Attention Re-export Module.
Enables backward compatibility for imports from `framework.attentions.svd_compressed_attention`.
"""

from framework.attentions.svd_compressed import (
    SVDCompressedAttention,
    inject_svd_compressed,
    inject_svd_compressed as inject_svd_compressed_attention,
    lagrangian_rank_allocation,
    calculate_spectral_entropy,
    compute_cumulative_entropy,
)

__all__ = [
    "SVDCompressedAttention",
    "inject_svd_compressed",
    "inject_svd_compressed_attention",
    "lagrangian_rank_allocation",
    "calculate_spectral_entropy",
    "compute_cumulative_entropy",
]
