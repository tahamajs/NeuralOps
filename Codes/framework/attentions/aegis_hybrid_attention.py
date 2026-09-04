r"""Compatibility shim for aegis_hybrid_attention."""
from framework.attentions.aegis_hybrid import (
    AegisHybridConfig,
    FactorizedLinear,
    AegisHybridAttention,
    inject_aegis_hybrid,
)

__all__ = [
    "AegisHybridConfig",
    "FactorizedLinear",
    "AegisHybridAttention",
    "inject_aegis_hybrid",
]
