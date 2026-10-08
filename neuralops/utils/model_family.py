"""Architecture-family detection and common transformer model metadata.

Re-exported from neuralops.core.model_family for backward compatibility.
"""

from __future__ import annotations

from neuralops.core.model_family import (
    CAUSAL_FAMILIES,
    FAMILIES,
    FAMILY_ALIASES,
    GPT_ABSOLUTE_POSITION_FAMILIES,
    ROTARY_FAMILIES,
    ModelAdapter,
    detect_family,
)

__all__ = [
    "CAUSAL_FAMILIES",
    "FAMILIES",
    "FAMILY_ALIASES",
    "GPT_ABSOLUTE_POSITION_FAMILIES",
    "ModelAdapter",
    "ROTARY_FAMILIES",
    "detect_family",
]
