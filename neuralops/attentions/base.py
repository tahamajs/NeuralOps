"""
Attentions Base Module Facade

Direct interface to BaseAttention and BaseCompressor module contracts.

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.core.base import *  # noqa: F403
import neuralops.core.base as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
