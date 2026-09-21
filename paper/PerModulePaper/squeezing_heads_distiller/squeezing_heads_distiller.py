"""Shim for squeezing_heads_distiller re-exporting from squeezing_heads_distill."""
from framework.attentions.squeezing_heads_distill import (
    SqueezingHeadsDistiller,
    SqueezingHeads,
    SqueezingHeadsAttention,
    LoRALinear,
    inject_squeezing_heads,
    inject_squeezing_heads_distill,
    inject_squeezing_heads_distiller,
    inject_squeezing_heads_attention,
    inject_squeezing_heads_distill_attention,
)

__all__ = [
    "SqueezingHeadsDistiller",
    "SqueezingHeads",
    "SqueezingHeadsAttention",
    "LoRALinear",
    "inject_squeezing_heads",
    "inject_squeezing_heads_distill",
    "inject_squeezing_heads_distiller",
    "inject_squeezing_heads_attention",
    "inject_squeezing_heads_distill_attention",
]
