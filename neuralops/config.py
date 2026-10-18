"""
Unified Configuration Facade

Direct interface to root Config container and specialized sub-configuration classes.

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.config.config import *  # noqa: F403
import neuralops.config.config as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
