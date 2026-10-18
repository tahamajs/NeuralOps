"""
Operator Registry Facade

Direct interface to Triton kernel operator registrations and torch.compile custom ops.

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.core.triton_kernels.op_registry import *  # noqa: F403
import neuralops.core.triton_kernels.op_registry as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
