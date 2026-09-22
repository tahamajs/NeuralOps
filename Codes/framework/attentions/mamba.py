"""
Mamba: Faithful Selective State Space Model linear attention replacement.

Faithful implementation of Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling
with Selective State Spaces". Replaces previous non-selective attention fallback.
"""

from __future__ import annotations
from typing import Any, Optional, Tuple

import torch
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.mamba_block import MambaBlock


@ATTENTION_REGISTRY.register("mamba")
@ATTENTION_REGISTRY.register("mamba_attention")
class Mamba(MambaBlock):
    """
    Drop-in alias of MambaBlock for registry backwards compatibility.
    """
    pass


__all__ = ["Mamba", "MambaBlock"]
