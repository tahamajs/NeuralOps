"""
Canonical `neuralops.attention` subpackage.
Re-exports the Attention registry, BaseAttention, and all 8 architecture families.

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

__all__ = ["ATTENTION_REGISTRY", "BaseAttention"]
