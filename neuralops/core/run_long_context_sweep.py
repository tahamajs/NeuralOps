#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Backward compatibility shim for neuralops.benchmarks.run_long_context_sweep.
"""

from __future__ import annotations

import warnings
from neuralops.benchmarks.run_long_context_sweep import *  # noqa: F401, F403

warnings.warn(
    "neuralops.core.run_long_context_sweep has moved to neuralops.benchmarks.run_long_context_sweep and will be deprecated in v1.7.0",
    DeprecationWarning,
    stacklevel=2,
)
