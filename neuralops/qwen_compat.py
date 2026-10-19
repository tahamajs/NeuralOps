#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Qwen Compatibility Facade

Direct interface to Qwen model detection, configuration proxies, and RoPE bypass utilities.
"""

from __future__ import annotations

from neuralops.utils.qwen_compat import *  # noqa: F403
import neuralops.utils.qwen_compat as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
