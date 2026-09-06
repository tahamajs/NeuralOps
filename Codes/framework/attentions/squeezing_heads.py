"""
Squeezing Heads Attention Re-export Module.
Enables backward compatibility for imports from `framework.attentions.squeezing_heads`.
"""

from framework.attentions.squeezing_heads_distill import (
    SqueezingHeadsDistiller,
    LoRALinear,
    inject_squeezing_heads,
    inject_squeezing_heads as inject_squeezing_heads_distill,
)

__all__ = [
    "SqueezingHeadsDistiller",
    "LoRALinear",
    "inject_squeezing_heads",
    "inject_squeezing_heads_distill",
]
