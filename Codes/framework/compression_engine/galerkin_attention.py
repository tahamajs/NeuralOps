r"""
Galerkin Transformer: Petrov-Galerkin projection attention.
Re-exports repaired GalerkinAttention from framework.attentions.galerkin_attention.
"""

from framework.attentions.galerkin_attention import GalerkinAttention, inject_galerkin

__all__ = ["GalerkinAttention", "inject_galerkin"]