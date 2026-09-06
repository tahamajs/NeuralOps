"""
Spectron Attention Re-export Module.
Enables backward compatibility for imports from `framework.attentions.spectron_attention`.
"""

from framework.attentions.spectron import (
    SpectronAttention,
    LowRankSparseLinear,
    inject_spectron_layer,
    inject_spectron,
    inject_spectron_layer as inject_spectron_attention,
    allocate_ranks_spectron,
    spectral_entropy,
    cumulative_entropy_fraction,
)

__all__ = [
    "SpectronAttention",
    "LowRankSparseLinear",
    "inject_spectron_layer",
    "inject_spectron",
    "inject_spectron_attention",
    "allocate_ranks_spectron",
    "spectral_entropy",
    "cumulative_entropy_fraction",
]
