# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Replacement Facade

Provides direct access to attention replacement utilities and wrapper classes.
"""

from __future__ import annotations

from neuralops.utils.replace_attention import *  # noqa: F403
import neuralops.utils.replace_attention as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
