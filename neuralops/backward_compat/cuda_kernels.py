#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Backward-compat shim for neuralops.core.cuda_kernels → neuralops.ops.cuda.
"""

from __future__ import annotations

from neuralops.ops.cuda import *  # noqa: F401, F403
