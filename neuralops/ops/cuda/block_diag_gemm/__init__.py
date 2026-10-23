#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Block-Diagonal GEMM Hardware Operators.
"""

from neuralops.ops.cuda.block_diag_gemm.block_diag_autograd import (
    block_diag_matmul,
    block_diag_matmul_pytorch_ref,
)

__all__ = [
    "block_diag_matmul",
    "block_diag_matmul_pytorch_ref",
]
