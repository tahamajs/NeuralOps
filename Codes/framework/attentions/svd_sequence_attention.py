"""
Low-Rank SVD Sequence Attention Re-export Module.
Enables backward compatibility for imports from `framework.attentions.svd_sequence_attention`.
"""

from framework.attentions.svd_attention import (
    SVDLinearAttention,
    LowRankSVDSequenceAttention,
    LoRALinear,
    inject_svd_linear_attention,
    inject_svd_linear_attention as inject_svd_sequence_attention,
)

__all__ = [
    "SVDLinearAttention",
    "LowRankSVDSequenceAttention",
    "LoRALinear",
    "inject_svd_linear_attention",
    "inject_svd_sequence_attention",
]
