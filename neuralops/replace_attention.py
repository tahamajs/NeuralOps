"""
Attention Replacement Facade

Provides direct access to attention replacement utilities and wrapper classes.

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.utils.replace_attention import *  # noqa: F403
import neuralops.utils.replace_attention as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
