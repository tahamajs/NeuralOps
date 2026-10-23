#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Three-Level Kernel Fallback Dispatcher shim."""

from __future__ import annotations

from neuralops.core.cuda_kernels._fallback import *  # noqa: F401, F403
