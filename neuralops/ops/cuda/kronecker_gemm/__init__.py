#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fused Kronecker GEMM Hardware Operators.
"""

from neuralops.ops.cuda.kronecker_gemm.kronecker_autograd import (
    kronecker_matmul,
    kronecker_matmul_pytorch_ref,
)

__all__ = [
    "kronecker_matmul",
    "kronecker_matmul_pytorch_ref",
]
