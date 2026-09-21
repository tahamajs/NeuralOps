r"""
SALT Attention: Singular Value Adaptation with Low-Rank Tuning.
Re-exports from framework.attentions.salt_attention.
"""

from framework.attentions.salt_attention import SALTAttention, SALTLinear, inject_salt_attention

__all__ = ["SALTAttention", "SALTLinear", "inject_salt_attention"]
