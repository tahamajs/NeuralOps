"""
Model Family Detection

Identifies model architectures, transformer families, and projection naming conventions.

Author: Taha Majlesi
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
