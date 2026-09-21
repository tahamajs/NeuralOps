"""
MoBA: Mixture of Block Attention for Long-Context Modeling.
Module alias re-exporting MoBAAttention and inject_moba from moba_attention.
"""

from framework.attentions.moba_attention import (
    MoBAAttention,
    inject_moba,
)

__all__ = [
    "MoBAAttention",
    "inject_moba",
]
