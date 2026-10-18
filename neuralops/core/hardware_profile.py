"""
Hardware Profile Facade

Direct interface to GPU hardware capability profiling, SRAM limits, and architecture detection.

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.core.triton_kernels.hardware_profile import *  # noqa: F403
import neuralops.core.triton_kernels.hardware_profile as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
