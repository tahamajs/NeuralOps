"""
Canonical `neuralops.attention` subpackage.
Re-exports the Attention registry, BaseAttention, and all 8 architecture families.
"""

from __future__ import annotations

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.base import BaseAttention
import neuralops.attentions as _attns

__all__ = ["ATTENTION_REGISTRY", "BaseAttention"]
