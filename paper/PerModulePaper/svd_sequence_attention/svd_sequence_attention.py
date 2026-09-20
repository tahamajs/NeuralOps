"""
Low-Rank SVD Sequence Attention Re-export Module.
Enables backward compatibility for imports from `framework.attentions.svd_sequence_attention`.
"""

from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.svd_attention import (
    SVDLinearAttention,
    LowRankSVDSequenceAttention,
    LoRALinear,
    inject_svd_linear_attention,
    inject_svd_linear_attention as inject_svd_sequence_attention,
)

SVDSequenceAttention = SVDLinearAttention
inject_svd_sequence = inject_svd_sequence_attention

# Ensure registered under all canonical keys
ATTENTION_REGISTRY.register("svd_sequence", supports_causal=True, supports_gqa=True)(SVDSequenceAttention)
ATTENTION_REGISTRY.register("svd_sequence_attention", supports_causal=True, supports_gqa=True)(SVDSequenceAttention)
ATTENTION_REGISTRY.register("svd_sequence_inject")(inject_svd_sequence)
ATTENTION_REGISTRY.register("svd_sequence_attention_inject")(inject_svd_sequence_attention)
ATTENTION_REGISTRY.register("inject_svd_sequence")(inject_svd_sequence)
ATTENTION_REGISTRY.register("inject_svd_sequence_attention")(inject_svd_sequence_attention)

__all__ = [
    "SVDLinearAttention",
    "SVDSequenceAttention",
    "LowRankSVDSequenceAttention",
    "LoRALinear",
    "inject_svd_linear_attention",
    "inject_svd_sequence_attention",
    "inject_svd_sequence",
]
