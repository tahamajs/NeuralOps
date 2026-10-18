"""
Qwen Compatibility Facade

Direct interface to Qwen model detection, configuration proxies, and RoPE bypass utilities.

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.utils.qwen_compat import *  # noqa: F403
import neuralops.utils.qwen_compat as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
