# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware Profile Facade

Direct interface to GPU hardware capability profiling, SRAM limits, and architecture detection.
"""

from __future__ import annotations

from neuralops.core.triton_kernels.hardware_profile import *  # noqa: F403
import neuralops.core.triton_kernels.hardware_profile as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
