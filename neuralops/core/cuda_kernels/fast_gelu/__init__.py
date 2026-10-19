#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fast GELU Activation Hardware Operators.
"""

from neuralops.core.cuda_kernels.fast_gelu.fast_gelu_autograd import (
    fast_gelu,
    fast_gelu_pytorch_ref,
)

__all__ = [
    "fast_gelu",
    "fast_gelu_pytorch_ref",
]
