#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Optimization Kernels Subpackage.

Hosts unified Triton and CUDA kernel dispatch mechanisms and accelerated primitives.
"""

from neuralops.ops.triton import *  # noqa: F401, F403
