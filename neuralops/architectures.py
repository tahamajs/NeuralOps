"""
Architecture Adapters Facade

Direct interface to model family detection, adapter wrappers, and compatibility tools.

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.utils.compat.architectures import *  # noqa: F403
import neuralops.utils.compat.architectures as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
